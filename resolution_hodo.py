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

Planes: x is the average of the two x planes, y is y1 -- y2 fires a single cluster
more often but its profile is jagged, y1 is parabolic over its whole range. Only
events with exactly one cluster in the planes used are kept, which costs a large
fraction of the statistics -- that cost is reported per point, and it is the price of
a cut that does not depend on the amplitudes.

Two ways of setting the window, --window:
  parabola  vertex of the response parabola +- half * W, with the fit range scanned
            and the answer accepted only if it does not depend on the range. Where no
            parabola survives the scan the energy is dropped and the reason printed.
  plateau   the contiguous range around the maximum where the response stays within
            --tol of it. No calibration at all, and no parabola needed.

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
from uniformita_pos import (fit_dcb, rel, wmean, wscatter, design, VERA,
                           A_TOT_MIN, TAILS)
from drift_dcb_all import syst_for_unit_chi2
from hodoscope_calib import (hodo_xy, FILES, crystal_curvature, response_profile,
                             parabola_scan, COORD)

SYNC_C = 1.92e-7
ETA0, PHI0, SEL = 18., 6., 0.2

PLATEAU_TOL = 0.005          # frazione sotto il massimo che conta ancora come piatta


def window_from_response(h, at, tol, lo=None, hi=None, nb=40, nmin=150):
    """Hodoscope window where the response is flat, without any calibration.

    The cut on the centroid selects the region around the crystal centre where the
    response varies little. The same region can be defined directly on the hodoscope,
    with no offset and no scale: profile <A_tot> against the hodoscope coordinate and
    keep the contiguous range around the maximum where the profile stays within tol of
    its plateau value. Nothing here uses the centroid, so the selection does not
    depend on the amplitudes whose width is being measured.

    No range is imposed in either view: the profile covers the whole range the data
    span, from the 0.5th to the 99.5th percentile, and the window is set only by where
    the response is flat. That is safe on y1, whose profile is a clean response curve
    over its whole range; on y2, which is jagged below zero, the plateau search would
    lock onto a fluctuation instead.

    Returns (lo, hi, drop_pct, n) with drop_pct the response variation across the
    window, which is what makes windows at different energies comparable.
    """
    m = np.isfinite(h)
    if lo is not None:
        m &= (h > lo)
    if hi is not None:
        m &= (h < hi)
    if m.sum() < 3000:
        return None
    # nessun limite a priori: il profilo si fa su tutto il range coperto dai dati e
    # la finestra la decide solo la piattezza della risposta
    e = np.linspace(np.percentile(h[m], 0.5), np.percentile(h[m], 99.5), nb + 1)
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
    ap.add_argument("--window", choices=("plateau", "parabola"), default="plateau",
                    help="plateau = range where the response stays within --tol of its "
                         "maximum, no calibration; parabola = vertex of the response "
                         "parabola +- half * W, with W the crystal width from the ratio "
                         "of the curvatures in mm and in crystal units")
    ap.add_argument("--tol", type=float, default=PLATEAU_TOL,
                    help="how far below the maximum the response may fall inside the "
                         "hodoscope window; 0.005 means the plateau is kept to 0.5%%")
    ap.add_argument("--yplane", choices=("y1", "y2"), default="y1",
                    help="hodoscope y plane to cut on; y1 is the one with a clean response profile over its whole range")
    ap.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="R:E points dropped entirely")
    ap.add_argument("--nofit-energies", nargs="*", type=int, default=[250, 275],
                    help="energies kept in the plot but left out of the N/S/C fit")
    ap.add_argument("--tails", choices=("free", "fixed", "both"), default="both",
                    help="DCB tail parameters per run: free, or held at the values of "
                         "the pooled fit of the same energy. 'both' takes free as the "
                         "nominal and carries |free - fixed| as a systematic on the "
                         "fit model, added to the error bar")
    runsets.add_argument(ap)
    a = ap.parse_args()
    drop, only = runsets.resolve(a.runset, a.exclude_runs)
    excl = {tuple(int(v) for v in s.split(":")) for s in a.exclude}
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for R in a.resistances:
        bes = load_bes(a.besdir, R)
        drift_syst = load_drift(a.plotdir, R)
        cry = crystal_curvature(a.plotdir, R)
        d, pat = FILES[R]
        for f in sorted(glob.glob(os.path.join(a.base, d, pat)),
                        key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1))):
            E = int(re.match(r"(\d+)", os.path.basename(f)).group(1))
            if (R, E) in excl:
                continue
            arr = uproot.open(f)["h4_reco"].arrays(
                ["run", "A_tot", "pos_eta", "pos_phi", "hodo_x1_nclusters", "hodo_x1_pos",
                 "hodo_x2_nclusters", "hodo_x2_pos", "hodo_y1_nclusters", "hodo_y1_pos",
                 "hodo_y2_nclusters", "hodo_y2_pos"],
                library="ak")
            run = ak.to_numpy(arr["run"]); at = ak.to_numpy(arr["A_tot"]).astype(float)
            pe = ak.to_numpy(arr["pos_eta"]); pp = ak.to_numpy(arr["pos_phi"])
            x, y = hodo_xy(arr, a.yplane)
            base = at > A_TOT_MIN
            if drop:
                base &= ~np.isin(run, drop)
            if len(only):
                base &= np.isin(run, only)
            cut_c = base & (np.abs(pe - ETA0) <= a.half) & (np.abs(pp - PHI0) <= a.half)
            # finestra dove la risposta e' piatta, senza calibrazione: nessun offset,
            # nessuna scala, nessun uso del centroide
            core = base & (np.abs(at / np.median(at[base]) - 1) < 0.10)
            why, fallback = {}, []
            if a.window == "parabola":
                # offset dal vertice della parabola di risposta, scala dal rapporto
                # fra la curvatura in mm e quella in unita' di cristallo. Il range di
                # fit non e' fissato: viene scansionato, e la parabola si accetta solo
                # se vertice e W non dipendono dal range (vedi hodoscope_calib.py).
                wx = wy = None
                for tag, v in (("x", x), ("y", y)):
                    pr = response_profile(v[core], at[core])
                    sc = parabola_scan(*(pr if pr else (None, None, None)),
                                       cry.get((E, dict(COORD)[tag])), a.half)
                    why[tag] = "" if sc["ok"] else sc["why"]
                    if not sc["ok"]:
                        continue
                    w = (sc["x0"] - a.half * sc["W"], sc["x0"] + a.half * sc["W"],
                         0., 0)
                    if tag == "x":
                        wx = w
                    else:
                        wy = w
                # dove la parabola non esiste il punto non si butta via: si ricade
                # sulla finestra di plateau, che non ha bisogno di nessun fit, e il
                # punto resta nel grafico marcato come tale. Buttarlo lo farebbe
                # sparire dal plot senza dire perche'
                if wx is None:
                    wx = window_from_response(x[core], at[core], a.tol)
                    fallback.append("x")
                if wy is None:
                    wy = window_from_response(y[core], at[core], a.tol)
                    fallback.append("y")
            else:
                wx = window_from_response(x[core], at[core], a.tol)
                wy = window_from_response(y[core], at[core], a.tol)
            if wx is None or wy is None:
                miss = ", ".join(f"{t}: {why[t]}" for t in "xy" if why.get(t)) \
                       or "plateau non trovato"
                print(f"  {R} ohm {E:>4} GeV: nessuna finestra ({miss}), salto")
                continue
            if fallback:
                print(f"  {R} ohm {E:>4} GeV: niente parabola in "
                      f"{'+'.join(fallback)} ({'; '.join(why[t] for t in fallback)}), "
                      f"finestra dal plateau")
            cut_h = (base & np.isfinite(x) & np.isfinite(y)
                     & (x >= wx[0]) & (x <= wx[1]) & (y >= wy[0]) & (y <= wy[1]))
            if cut_h.sum() < 500:
                print(f"  {R} ohm {E:>4} GeV: solo {cut_h.sum()} eventi dopo il taglio hodo, salto")
                continue
            out = dict(resistance=R, energy=E, energy_true=VERA.get(E, float(E)),
                       window="+".join(fallback) + "-plateau" if fallback
                              else a.window,
                       n_centroid=int(cut_c.sum()), n_hodo=int(cut_h.sum()),
                       x_lo=wx[0], x_hi=wx[1], x_drop=wx[2], n_x=wx[3],
                       y_lo=wy[0], y_hi=wy[1], y_drop=wy[2], n_y=wy[3])
            for tag, m in (("cen", cut_c), ("hodo", cut_h)):
                # code del DCB dal fit cumulativo di questa energia e di questa
                # selezione: ad alta statistica sono determinate, run per run no
                tails = None
                if a.tails in ("fixed", "both") and m.sum() >= 500:
                    fp = fit_dcb(at[m], E, R)
                    if fp is not None:
                        tails = fp["tails"]

                def per_run(fix):
                    """sigma/mu con il suo errore, e il picco con il suo, run per run"""
                    V, Er, Rw, Pk, Ep, Rn = [], [], [], [], [], []
                    for r in sorted(int(u) for u in np.unique(run[m])):
                        q = m & (run == r)
                        if q.sum() < 300:
                            continue
                        f0 = fit_dcb(at[q], E, R, fix=fix)
                        if f0 is None:
                            continue
                        # la correzione della non-uniformita' si applica SOLO quando
                        # il taglio e' sul centroide
                        if tag == "cen":
                            ac = correct_response(at[q], (pe - ETA0)[q], (pp - PHI0)[q],
                                                  (f0["lo"], f0["hi"]))
                            fc = fit_dcb(ac, E, R, fix=fix)
                        else:
                            fc = f0
                        if fc is None:
                            continue
                        v, e = rel(fc)
                        V.append(v); Er.append(e); Rw.append(rel(f0)[0])
                        Pk.append(f0["peak"]); Ep.append(f0["err_peak"]); Rn.append(r)
                    return V, Er, Rw, Pk, Ep, Rn

                vals, errs, raws, peaks, epeaks, rns = per_run(
                    tails if a.tails == "fixed" else None)
                # variante col modello alternativo, per la sistematica sul fit
                alt = (per_run(tails)[0] if (a.tails == "both" and tails) else [])

                # Dove nessun run singolo arriva a 300 eventi il punto non si perde:
                # si fa UN fit cumulativo su tutti i run insieme. Succede a 340 ohm
                # 250 GeV, dove il taglio sull'odoscopio lascia 846 eventi su 8 run.
                pooled = False
                fx = tails if a.tails == "fixed" else None
                if not vals and m.sum() >= 500:
                    f0 = fit_dcb(at[m], E, R, fix=fx)
                    if f0 is not None:
                        if tag == "cen":
                            ac = correct_response(at[m], (pe - ETA0)[m], (pp - PHI0)[m],
                                                  (f0["lo"], f0["hi"]))
                            fc = fit_dcb(ac, E, R, fix=fx)
                        else:
                            fc = f0
                        if fc is not None:
                            v, e = rel(fc)
                            vals, errs, raws, rns = [v], [e], [rel(f0)[0]], [0]
                            peaks, epeaks = [f0["peak"]], [f0["err_peak"]]
                            pooled = True
                out[tag + "_pooled"] = int(pooled)
                # pesi 1/sigma^2, come chiesto: l'errore statistico e' la sigma della
                # sigma dalla varianza pesata con quei pesi
                wts = [1.0 / (e * e) if e > 0 else 0.0 for e in errs]
                if not vals:
                    for k, v in ((tag, np.nan), (tag + "_stat", np.nan),
                                 (tag + "_drift", 0.), (tag + "_pos", 0.),
                                 (tag + "_chi2", np.nan), (tag + "_tails", 0.),
                                 (tag + "_err", np.nan)):
                        out[k] = v
                    out["nrun_" + tag] = 0
                    out[tag + "_runs"] = []
                    continue
                mu, er = wmean(vals, errs, wts)
                sc = wscatter(vals, wts)
                mu_raw = wmean(raws, errs, wts)[0]
                # POS_eff, solo per la catena col centroide
                out[tag + "_pos"] = (float(np.sqrt(max(mu_raw ** 2 - mu ** 2, 0.)))
                                     if tag == "cen" else 0.)
                est = max(er, sc) if np.isfinite(sc) else er
                out[tag] = mu
                out[tag + "_stat"] = est
                # sistematica sul modello di fit: quanto si sposta il punto se le code
                # del DCB si tengono fisse invece che libere. Non e' una larghezza da
                # togliere, e' un'ambiguita' sulla misura, quindi va nella barra
                ts = 0.
                if alt:
                    n = min(len(alt), len(errs))
                    aw = [1.0 / (e * e) if e > 0 else 0.0 for e in errs[:n]]
                    ma = wmean(alt[:n], errs[:n], aw)[0]
                    if np.isfinite(ma):
                        ts = abs(mu - ma)
                out[tag + "_tails"] = ts
                # DRIFT SUL PICCO, non sulla sigma. La sigma di ogni run e' misurata
                # attorno al picco di quel run, quindi il drift fra run non la sporca
                # direttamente: quello che si misura sui picchi e' l'instabilita' della
                # risposta, ed e' la stessa instabilita' che, agendo dentro un run,
                # allarga la sigma. Si stima dai picchi per run -- l'errore in piu' che
                # porta il loro fit a una costante a chi2/ndf = 1 -- e si esprime in
                # punti percentuali del picco medio, cosi' e' sottraibile in quadratura
                # da sigma/mu. Con un solo run non c'e' dispersione ed e' zero.
                dr, c0 = 0., np.nan
                if len(peaks) > 1:
                    qq = syst_for_unit_chi2(np.array(peaks), np.array(epeaks))
                    c0 = float(qq[1])
                    if np.isfinite(qq[0]) and qq[2] > 0:
                        dr = float(100 * qq[0] / qq[2])
                out[tag + "_drift"] = dr
                out[tag + "_chi2"] = c0
                # la barra porta lo statistico e la sistematica sul modello di fit;
                # il drift NON sta nella barra, si sottrae
                out[tag + "_err"] = math.hypot(est, ts)
                out["nrun_" + tag] = 0 if pooled else len(vals)
                out[tag + "_runs"] = list(zip(rns, vals, errs, peaks, epeaks))
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
    cols = ("resistance,energy,energy_true,window,n_centroid,n_hodo,nrun_cen,nrun_hodo,"
            "x_lo,x_hi,x_drop,n_x,y_lo,y_hi,y_drop,n_y,bes,sync,"
            "cen,cen_stat,cen_drift,cen_chi2,cen_tails,cen_pos,cen_err,cen_corr,"
            "cen_pooled,hodo,hodo_stat,hodo_drift,hodo_chi2,hodo_tails,hodo_pos,"
            "hodo_err,hodo_corr,hodo_pooled")
    p = os.path.join(a.outdir, "resolution_hodo.csv")
    with open(p, "w") as fh:
        fh.write(cols + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                              for c in cols.split(",")) + "\n")
    print("->", p)

    p = os.path.join(a.outdir, "sigma_per_run.csv")
    with open(p, "w") as fh:
        fh.write("resistance,energy,chain,run,sigma_pct,err_pct,peak,err_peak\n")
        for r in rows:
            for tag in ("cen", "hodo"):
                for rn, v, e, pk, ep in r.get(tag + "_runs", []):
                    fh.write(f"{r['resistance']},{r['energy']},{tag},{rn},"
                             f"{v:.6g},{e:.6g},{pk:.6g},{ep:.6g}\n")
    print("->", p)

    Rs = [R for R in (340, 400, 500) if any(r["resistance"] == R for r in rows)]

    # ---- controllo del drift, energia per energia -------------------------
    # Un pannello per energia: il PICCO dei singoli run, normalizzato al picco medio,
    # con il suo errore di fit, la costante e la banda di drift. Il drift e' l'errore
    # in piu' che porta il chi2/ndf di quel fit a 1: dove il chi2/ndf e' gia' <= 1 il
    # drift e' zero perche' non serve niente, e il pannello lo mostra.
    for tag, lab in (("cen", "centroid"), ("hodo", "hodoscope")):
        for R in Rs:
            q = [r for r in rows if r["resistance"] == R and len(r.get(tag + "_runs", [])) > 1]
            if not q:
                continue
            nc = 4; nr = int(np.ceil(len(q) / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.2 * nr), squeeze=False)
            for k, r in enumerate(q):
                ax = axs[k // nc][k % nc]
                pr = r[tag + "_runs"]
                xs = np.arange(len(pr))
                pk = np.array([p for _, _, _, p, _ in pr])
                ep = np.array([e for _, _, _, _, e in pr])
                w = 1. / ep ** 2
                m0 = float((pk * w).sum() / w.sum())
                ys = 100 * (pk / m0 - 1.); es = 100 * ep / m0
                dr = r.get(tag + "_drift", 0.)
                c0 = r.get(tag + "_chi2", np.nan)
                ax.errorbar(xs, ys, yerr=es, fmt="o", ms=5, capsize=3, color="C0")
                ax.axhline(0., color="C3", lw=1.6,
                           label=f"weighted mean {m0:.1f} ADC")
                if dr > 0:
                    ax.axhspan(-dr, dr, color="C3", alpha=.15,
                               label=f"$\\pm$ drift {dr:.4f} %")
                ax.set_xticks(xs)
                ax.set_xticklabels([str(rn) for rn, *_ in pr], rotation=90, fontsize=6)
                ndf = len(pr) - 1
                if dr > 0:
                    txt = (f"$\\chi^2$/ndf {c0:.2f} ({ndf} ndf) $>$ 1  "
                           f"$\\rightarrow$ drift {dr:.4f} %")
                    col = "C3"
                else:
                    txt = (f"$\\chi^2$/ndf {c0:.2f} ({ndf} ndf) $\\leq$ 1  "
                           f"$\\rightarrow$ drift 0")
                    col = "C2"
                ax.set_title(f"{r['energy']} GeV\n{txt}", fontsize=8.5, color=col,
                             linespacing=1.5)
                ax.set_ylabel("peak / $\\langle$peak$\\rangle$ $-$ 1  [%]", fontsize=8)
                ax.tick_params(axis="y", labelsize=7); ax.grid(alpha=.3, axis="y")
                ax.legend(fontsize=6.5, loc="best")
            for k in range(len(q), nr * nc):
                axs[k // nc][k % nc].set_axis_off()
            fig.suptitle(f"{R} $\\Omega$ — cut on the {lab} — per-run PEAK against a "
                         f"constant.  green: already compatible, drift = 0.  "
                         f"red: extra error needed", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            p = os.path.join(a.outdir, f"drift_check_{tag}_{R}ohm.png")
            fig.savefig(p, dpi=130); plt.close(fig)
            print("->", p)

    # due versioni dello stesso grafico: una con tutti i punti dentro al fit, una in
    # cui --nofit-energies restano disegnati ma fuori dal fit. Cosi' si vede subito
    # quanto pesano quei punti sui parametri
    for tag, lab in (("cen", "centroid"), ("hodo", "hodoscope")):
      for allfit in (False, True):
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
                          alpha=.9, label="$-$ BES $-$ sync" + (" $-$ POS$_{eff}$" if tag == "cen" else "")
                                + " $-$ drift")
              infit = (np.ones(len(q), bool) if allfit else
                       ~np.isin([r["energy"] for r in q], a.nofit_energies))
              g = np.isfinite(y) & (y > 0) & (e > 0) & infit
              shown = np.isfinite(y) & (y > 0) & (e > 0) & ~infit
              if shown.any():
                  ax.errorbar(x[shown], y[shown], yerr=e[shown], fmt="^", ms=7.5,
                              mfc="none", color="C3", capsize=3, label="not in the fit")
              pl = np.array([bool(r.get(tag + "_pooled", 0)) for r in q])
              pl &= np.isfinite(y) & (y > 0)
              if pl.any():
                  ax.plot(x[pl], y[pl], "s", ms=14, mfc="none", mew=1.4, color="C4",
                          label="one pooled fit (no run has enough events)")
              fb = np.array([str(r.get("window", "")).endswith("-plateau") for r in q])
              fb &= np.isfinite(y) & (y > 0)
              if fb.any():
                  ax.plot(x[fb], y[fb], "o", ms=13, mfc="none", mew=1.4, color="0.25",
                          label="no parabola: plateau window")
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
              if np.isfinite(np.nanmax(raw)):
                  ax.set_ylim(0, float(np.nanmax(raw) * 1.08))
              ax2.plot(x, raw, "o-", ms=5, color="0.35", label="$\\sigma/\\mu$")
              ax2.plot(x, bes, "D-.", ms=5, color="C1", label="BES")
              ax2.plot(x, syn, "^-", ms=5, color="C4", label="synchrotron")
              if tag == "cen":
                  ax2.plot(x, np.where(pos > 0, pos, np.nan), "v-", ms=5, color="C2",
                           label="POS$_{eff}$")
              nr = np.array([r.get("nrun_" + tag, 0) for r in q])
              ax2.plot(x, np.where(dri > 0, dri, np.nan), "s--", ms=5, color="C0",
                       label="drift")
              # dove il drift non c'e' si dice perche': un solo run (niente dispersione
              # da misurare) oppure sigma per run gia' compatibili fra loro
              lo2 = float(np.nanmin(np.concatenate([bes, syn, [1e-3]])))
              m1 = (dri <= 0) & (nr <= 1)
              mc = (dri <= 0) & (nr > 1)
              if m1.any():
                  ax2.plot(x[m1], np.full(m1.sum(), lo2), "x", ms=6, color="C0",
                           label="drift n/a (1 run)")
              if mc.any():
                  ax2.plot(x[mc], np.full(mc.sum(), lo2), "s", ms=5, mfc="none",
                           color="C0", label="drift = 0 ($\\chi^2$/ndf $\\leq$ 1)")
              t = np.concatenate([raw, bes, syn, pos[pos > 0], dri[dri > 0], [1e-3]])
              t = t[np.isfinite(t) & (t > 0)]
              ax2.set_yscale("log")
              if len(t):
                  ax2.set_ylim(float(t.min() * .4), float(t.max() * 2.5))
              ax2.set_xlabel("True beam energy [GeV]")
              ax2.set_ylabel("size of each term  [%]")
              ax2.grid(alpha=.3, which="both"); ax2.legend(fontsize=8)
          fig.suptitle(f"cut on the {lab}   $\\quad$   $A_{{tot}} > {A_TOT_MIN:.0f}$ ADC"
                       + ("   $\\quad$   all points in the fit" if allfit else
                          "   $\\quad$   " + ", ".join(str(e) for e in a.nofit_energies)
                          + " GeV shown but not fitted"), fontsize=12)
          fig.tight_layout()
          p = os.path.join(a.outdir,
                           f"resolution_terms_{tag}{'_allpoints' if allfit else ''}.png")
          fig.savefig(p, dpi=150); plt.close(fig)
          print("->", p)


if __name__ == "__main__":
    main()
