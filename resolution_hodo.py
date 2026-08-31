"""
Energy resolution with the position cut made on the HODOSCOPE instead of the centroid.

The standard selection cuts on the ECAL centroid, |pos_eta - 18| <= 0.2 and
|pos_phi - 6| <= 0.2. The centroid is computed from the same amplitudes whose width
is being measured, so the cut is not independent of the quantity under study. Here
the same geometric cut is applied to the hodoscope, which measures the impact point
independently:

    |hodo_x - x0_x| <= HALF * W_x        |hodo_y - x0_y| <= HALF * W_y

with x0 and W from hodoscope_calib.py: x0 is the vertex of the response parabola in
millimetres, W the crystal width recovered from the ratio of the curvatures in
millimetres and in crystal units. HALF is 0.2, the same half-window as the centroid
cut.

Planes: x is the average of the two x planes, y is the second plane only because the
first is much less efficient. Only events with exactly one cluster in the planes used
are kept, which costs a large fraction of the statistics -- that cost is reported per
point, and it is the price of a cut that does not depend on the amplitudes.

From there the chain is the usual one: double-CB fit per run, weighted mean of the
per-run sigma/peak, then BES and synchrotron subtracted in quadrature.

Usage:
  python3 plot/resolution_hodo.py --base . --outdir plot/hodo --besdir plot/bes \\
      [--resistances 340] [--half 0.2]
"""

import argparse, os, glob, re, csv, math
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import runsets
from uniformita_pos import fit_dcb, rel, wmean, wscatter, design, VERA, A_TOT_MIN
from drift_dcb_all import syst_for_unit_chi2
from hodoscope_calib import hodo_xy, FILES

SYNC_C = 1.92e-7
ETA0, PHI0, SEL = 18., 6., 0.2


PLATEAU_TOL = 0.005          # frazione sotto il massimo che conta ancora come piatta


def window_from_response(h, at, tol, lo, hi, nb=30, nmin=150):
    """Hodoscope window where the response is flat, without any calibration.

    The cut on the centroid selects the region around the crystal centre where the
    response varies little. The same region can be defined directly on the hodoscope,
    with no offset and no scale: profile <A_tot> against the hodoscope coordinate and
    keep the contiguous range around the maximum where the profile stays within tol of
    its plateau value. Nothing here uses the centroid, so the selection does not
    depend on the amplitudes whose width is being measured.

    Ranges: x is used over [-15, 0] mm and y over [0, 8]. The y plane is only usable
    above zero -- below it the profile is jagged, with bin-to-bin jumps of a percent
    and no parabola -- and above 8 mm some energies show a rise of up to 2 % that is
    not the crystal response and that the plateau search would otherwise lock onto.

    Returns (lo, hi, drop_pct, n) with drop_pct the response variation across the
    window, which is what makes windows at different energies comparable.
    """
    m = np.isfinite(h) & (h > lo) & (h < hi)
    if m.sum() < 3000:
        return None
    e = np.linspace(np.percentile(h[m], 1), np.percentile(h[m], 99), nb + 1)
    i = np.clip(np.digitize(h[m], e) - 1, 0, nb - 1)
    xs, ys, ns = [], [], []
    for k in range(nb):
        q = i == k
        if q.sum() < nmin:
            continue
        xs.append(h[m][q].mean()); ys.append(at[m][q].mean()); ns.append(q.sum())
    if len(xs) < 6:
        return None
    xs, ys = np.array(xs), np.array(ys)
    # massimo su tre bin, cosi' una fluttuazione singola non sposta il centro
    sm = np.convolve(ys, np.ones(3) / 3., mode="same")
    sm[0], sm[-1] = ys[0], ys[-1]
    k0 = int(np.argmax(sm))
    thr = sm[k0] * (1 - tol)
    a = k0
    while a > 0 and sm[a - 1] >= thr:
        a -= 1
    b = k0
    while b < len(sm) - 1 and sm[b + 1] >= thr:
        b += 1
    if b <= a:
        return None
    drop = 100 * (1 - min(ys[a], ys[b]) / ys[k0])
    return float(xs[a]), float(xs[b]), float(drop), int(sum(ns[a:b + 1]))


def correct_response(at, u, v, win=None):
    """Correct A_tot event by event for the response non-uniformity in (eta, phi).

    Same surface as uniformita_pos.py, fitted on the events selected here: a full
    quadratic f = a0 + a1 u + a2 v + a3 u^2 + a4 v^2 + a5 uv by linear least squares
    on the individual events, then A -> A * <f> / f. The correction is applied to
    both selections, hodoscope and centroid: the response is not flat inside either
    window -- measured inside the hodoscope window it is in fact less flat, 0.22 to
    0.40 % against 0.11 to 0.22 % inside the centroid one -- so the term has to be
    removed in both cases and not only when cutting on the centroid.
    """
    if len(at) < 500:
        return at
    # la superficie si fitta SOLO sugli eventi dentro la finestra del fit di sigma:
    # includendo le code la mappa viene distorta, perche' quelle stanno ai bordi in
    # (u, v), e la "correzione" allarga la sigma invece di stringerla
    k = (np.ones(len(at), bool) if win is None
         else (at >= win[0]) & (at <= win[1]))
    if k.sum() < 300:
        return at
    X = design(u, v)
    try:
        c = np.linalg.lstsq(X[k], at[k] / np.median(at[k]), rcond=None)[0]
    except np.linalg.LinAlgError:
        return at
    f = X @ c
    fm = float(f[k].mean())
    if not (fm > 0) or (f <= 0).any():
        return at
    return at * fm / f


def load_drift(plotdir, R):
    """Drift systematic on sigma, in points of sigma/mu, scaled until the per-run
    sigma values fitted to a constant give chi2/ndf = 1. Written by drift_dcb_all.py.
    Undefined where a point has a single run, and zero there."""
    out = {}
    f = os.path.join(plotdir, str(R), f"sistematica_drift_{R}ohm.csv")
    if not os.path.exists(f):
        return out
    for r in csv.DictReader(open(f)):
        try:
            p = float(r["peak_medio"]); v = float(r["syst_sigma_ADC"])
        except (ValueError, KeyError):
            continue
        if p > 0 and np.isfinite(v):
            out[int(r["energy"])] = 100 * v / p
    return out


def load_bes(besdir, R):
    b = {}
    f = os.path.join(besdir, f"rereco_{R}_withBES.csv")
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            b[int(float(r["en"]))] = float(r["bes"])
    return b


def reso(x, N, S, C):
    return np.sqrt((100 * N / x) ** 2 + (S / np.sqrt(x)) ** 2 + C ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--outdir", default="plot/hodo")
    ap.add_argument("--besdir", default="plot/bes")
    ap.add_argument("--plotdir", default="plot")
    ap.add_argument("--resistances", nargs="+", type=int, default=[340])
    ap.add_argument("--half", type=float, default=SEL,
                    help="half-window of the centroid cut, in crystal units")
    ap.add_argument("--tol", type=float, default=PLATEAU_TOL,
                    help="how far below the maximum the response may fall inside the "
                         "hodoscope window; 0.005 means the plateau is kept to 0.5%%")
    ap.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    ap.add_argument("--exclude", nargs="*", default=["340:275"])
    runsets.add_argument(ap)
    a = ap.parse_args()
    drop, only = runsets.resolve(a.runset, a.exclude_runs)
    excl = {tuple(int(v) for v in s.split(":")) for s in a.exclude}
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for R in a.resistances:
        bes = load_bes(a.besdir, R)
        drift_syst = load_drift(a.plotdir, R)
        d, pat = FILES[R]
        for f in sorted(glob.glob(os.path.join(a.base, d, pat)),
                        key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1))):
            E = int(re.match(r"(\d+)", os.path.basename(f)).group(1))
            if (R, E) in excl:
                continue
            arr = uproot.open(f)["h4_reco"].arrays(
                ["run", "A_tot", "pos_eta", "pos_phi", "hodo_x1_nclusters", "hodo_x1_pos",
                 "hodo_x2_nclusters", "hodo_x2_pos", "hodo_y2_nclusters", "hodo_y2_pos"],
                library="ak")
            run = ak.to_numpy(arr["run"]); at = ak.to_numpy(arr["A_tot"]).astype(float)
            pe = ak.to_numpy(arr["pos_eta"]); pp = ak.to_numpy(arr["pos_phi"])
            x, y = hodo_xy(arr)
            base = at > A_TOT_MIN
            if drop:
                base &= ~np.isin(run, drop)
            if len(only):
                base &= np.isin(run, only)
            cut_c = base & (np.abs(pe - ETA0) <= a.half) & (np.abs(pp - PHI0) <= a.half)
            # finestra dove la risposta e' piatta, senza calibrazione: nessun offset,
            # nessuna scala, nessun uso del centroide
            core = base & (np.abs(at / np.median(at[base]) - 1) < 0.10)
            wx = window_from_response(x[core], at[core], a.tol, -15., 0.)
            wy = window_from_response(y[core], at[core], a.tol, 0., 8.)
            if wx is None or wy is None:
                print(f"  {R} ohm {E:>4} GeV: plateau non trovato, salto"); continue
            cut_h = (base & np.isfinite(x) & np.isfinite(y)
                     & (x >= wx[0]) & (x <= wx[1]) & (y >= wy[0]) & (y <= wy[1]))
            if cut_h.sum() < 500:
                print(f"  {R} ohm {E:>4} GeV: solo {cut_h.sum()} eventi dopo il taglio hodo, salto")
                continue
            out = dict(resistance=R, energy=E, energy_true=VERA.get(E, float(E)),
                       n_centroid=int(cut_c.sum()), n_hodo=int(cut_h.sum()),
                       x_lo=wx[0], x_hi=wx[1], x_drop=wx[2], n_x=wx[3],
                       y_lo=wy[0], y_hi=wy[1], y_drop=wy[2], n_y=wy[3])
            for tag, m in (("cen", cut_c), ("hodo", cut_h)):
                vals, errs, raws = [], [], []
                for r in sorted(int(v) for v in np.unique(run[m])):
                    q = m & (run == r)
                    if q.sum() < 300:
                        continue
                    f0 = fit_dcb(at[q], E, R)
                    if f0 is None:
                        continue
                    # la correzione della non-uniformita' si applica SOLO quando il
                    # taglio e' sul centroide. Con il taglio sull'odoscopio la
                    # posizione non entra nella selezione, quindi non si corregge e
                    # non si sottrae POS_eff: l'unica sistematica comune alle due
                    # catene e' quella di drift, che sta nelle barre d'errore.
                    if tag == "cen":
                        ac = correct_response(at[q], (pe - ETA0)[q], (pp - PHI0)[q],
                                              (f0["lo"], f0["hi"]))
                        fc = fit_dcb(ac, E, R)
                    else:
                        fc = f0
                    if fc is None:
                        continue
                    v, e = rel(fc)
                    vals.append(v); errs.append(e); raws.append(rel(f0)[0])
                # pesi 1/sigma^2, come chiesto: l'errore statistico e' la sigma della
                # sigma dalla varianza pesata con quei pesi; dove c'e' un solo run non
                # e' definita e si usa l'errore della media pesata
                wts = [1.0 / (e * e) if e > 0 else 0.0 for e in errs]
                if not vals:
                    out[tag] = np.nan; out[tag + "_err"] = np.nan; out["nrun_" + tag] = 0
                    continue
                mu, er = wmean(vals, errs, wts)
                sc = wscatter(vals, wts)
                mu_raw = wmean(raws, errs, wts)[0]
                # POS_eff, solo per la catena col centroide
                out[tag + "_pos"] = (float(np.sqrt(max(mu_raw ** 2 - mu ** 2, 0.)))
                                     if tag == "cen" else 0.)
                # statistico: varianza pesata quando c'e' piu' di un run, altrimenti
                # l'errore del fit; poi in quadratura la sistematica di drift.
                # Nessuna sistematica di posizione: con il taglio sull'odoscopio la
                # non-uniformita' non viene corretta, quindi non c'e' niente da
                # propagare su di essa.
                est = max(er, sc) if np.isfinite(sc) else er
                out[tag] = mu
                out[tag + "_stat"] = est
                # drift: calcolato energia per energia SULLE SIGMA DI QUESTA
                # selezione, non importato da quella col centroide. E' l'errore
                # aggiuntivo che porta a chi2/ndf = 1 il fit delle sigma per run a
                # una costante; con un solo run non c'e' dispersione ed e' zero.
                dr = 0.
                if len(vals) > 1:
                    q = syst_for_unit_chi2(np.array(vals), np.array(errs))
                    dr = float(q[0]) if np.isfinite(q[0]) else 0.
                out[tag + "_drift"] = dr
                out[tag + "_err"] = math.hypot(est, dr)
                out["nrun_" + tag] = len(vals)
            b = bes.get(E, 0.); syn = SYNC_C * out["energy_true"] ** 2.5
            out["bes"] = b; out["sync"] = syn
            for tag in ("cen", "hodo"):
                # sottrazioni in quadratura: BES, sincrotrone, POS_eff (solo con il
                # centroide) e il drift dove esiste, cioe' dove il punto ha >1 run
                v = out[tag]; ps = out.get(tag + "_pos", 0.); dr = out.get(tag + "_drift", 0.)
                out[tag + "_corr"] = (
                    math.sqrt(max(v * v - b * b - syn * syn - ps * ps - dr * dr, 0.))
                    if np.isfinite(v) else np.nan)
            rows.append(out)
            print(f"  {R} ohm {E:>4} GeV: x [{wx[0]:+6.2f},{wx[1]:+6.2f}] mm "
                  f"(calo {wx[2]:.2f}%), y [{wy[0]:+6.2f},{wy[1]:+6.2f}] mm "
                  f"(calo {wy[2]:.2f}%) | centroide {out['cen']:.4f}% su "
                  f"{out['n_centroid']} ev, hodo {out['hodo']:.4f}% su {out['n_hodo']} ev "
                  f"({100*out['n_hodo']/max(out['n_centroid'],1):.0f}%)", flush=True)

    if not rows:
        return
    cols = ("resistance,energy,energy_true,n_centroid,n_hodo,nrun_cen,nrun_hodo,"
            "x_lo,x_hi,x_drop,n_x,y_lo,y_hi,y_drop,n_y,bes,sync,"
            "cen,cen_stat,cen_drift,cen_pos,cen_err,cen_corr,"
            "hodo,hodo_stat,hodo_drift,hodo_pos,hodo_err,hodo_corr")
    p = os.path.join(a.outdir, "resolution_hodo.csv")
    with open(p, "w") as fh:
        fh.write(cols + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                              for c in cols.split(",")) + "\n")
    print("->", p)

    Rs = [R for R in (340, 400, 500) if any(r["resistance"] == R for r in rows)]
    for tag, lab in (("cen", "centroid"), ("hodo", "hodoscope")):
        fig, axs = plt.subplots(2, len(Rs), figsize=(6.4 * len(Rs), 10), sharex=True,
                                gridspec_kw=dict(height_ratios=[2, 1]), squeeze=False)
        for j, R in enumerate(Rs):
            q = [r for r in rows if r["resistance"] == R]
            x = np.array([r["energy_true"] for r in q])
            raw = np.array([r[tag] for r in q])
            y = np.array([r[tag + "_corr"] for r in q])
            e = np.array([r[tag + "_err"] for r in q])
            bes = np.array([r["bes"] for r in q]); syn = np.array([r["sync"] for r in q])
            pos = np.array([r.get(tag + "_pos", 0.) for r in q])
            dri = np.array([r.get(tag + "_drift", 0.) for r in q])
            ax, ax2 = axs[0][j], axs[1][j]
            ax.plot(x, raw, "o-", ms=6, color="0.35", label="$\\sigma/\\mu$")
            ax.errorbar(x, y, yerr=e, fmt="^-", ms=7.5, color="C3", capsize=3,
                        label="$-$ BES $-$ sync" + (" $-$ POS$_{eff}$" if tag == "cen" else "")
                              + " $-$ drift")
            g = np.isfinite(y) & (y > 0) & (e > 0)
            if g.sum() >= 4:
                mi = Minuit(LeastSquares(x[g], y[g], e[g], reso), N=0.3, S=3., C=0.3)
                for k in "NSC":
                    mi.limits[k] = (0, None)
                if R == 500:
                    mi.values["C"] = 0.300; mi.fixed["C"] = True
                mi.migrad(); mi.hesse()
                xs = np.linspace(x[g].min() * .9, x[g].max() * 1.05, 300)
                ax.plot(xs, reso(xs, *mi.values), "--", lw=2.2, color="darkviolet",
                        label="fit  $N/E \\oplus S/\\sqrt{E} \\oplus C$")
                nd = int(g.sum()) - (2 if R == 500 else 3)
                ax.text(.97, .95,
                        f"$N$ {1000*mi.values['N']:6.0f} $\\pm$ {1000*mi.errors['N']:.0f} MeV\n"
                        f"$S$ {mi.values['S']:6.2f} $\\pm$ {mi.errors['S']:.2f} %\n"
                        f"$C$ {mi.values['C']:6.3f}" +
                        (" % (FIXED)" if R == 500 else f" $\\pm$ {mi.errors['C']:.3f} %") +
                        f"\n$\\chi^2$/ndf {mi.fval:6.1f} / {nd}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
                        family="monospace", bbox=dict(fc="white", ec="darkviolet", pad=6))
            ax.set_title(f"{R} $\\Omega$", fontsize=12, fontweight="bold")
            ax.set_ylabel("$\\sigma/E$  [%]"); ax.grid(alpha=.3); ax.legend(fontsize=8)
            ax.set_ylim(0, float(np.nanmax(raw) * 1.08))
            ax2.plot(x, raw, "o-", ms=5, color="0.35", label="$\\sigma/\\mu$")
            ax2.plot(x, bes, "D-.", ms=5, color="C1", label="BES")
            ax2.plot(x, syn, "^-", ms=5, color="C4", label="synchrotron")
            if tag == "cen":
                ax2.plot(x, np.where(pos > 0, pos, np.nan), "v-", ms=5, color="C2",
                         label="POS$_{eff}$")
            ax2.plot(x, np.where(dri > 0, dri, np.nan), "s--", ms=5, color="C0",
                     label="drift")
            t = np.concatenate([raw, bes, syn, pos[pos > 0], dri[dri > 0], [1e-3]])
            ax2.set_yscale("log"); ax2.set_ylim(float(t[t > 0].min() * .4), float(t.max() * 2.5))
            ax2.set_xlabel("True beam energy [GeV]")
            ax2.set_ylabel("size of each term  [%]")
            ax2.grid(alpha=.3, which="both"); ax2.legend(fontsize=8)
        fig.suptitle(f"cut on the {lab}   $\\quad$   $A_{{tot}} > {A_TOT_MIN:.0f}$ ADC",
                     fontsize=12)
        fig.tight_layout()
        p = os.path.join(a.outdir, f"resolution_terms_{tag}.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print("->", p)


if __name__ == "__main__":
    main()
