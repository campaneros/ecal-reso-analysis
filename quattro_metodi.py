"""
Quattro modi di ricavare sigma/E da un file merged che contiene piu' run, messi
a confronto sullo stesso plot.

  global   fit DCB unico su tutti gli eventi del file, come fa fit.sh
  runmean  fit DCB per run, poi media di sigma_i/picco_i pesata sugli eventi
  norm     ogni evento riscalato per (picco_rif / picco del suo run), poi un fit
           unico: toglie gli sfasamenti di scala fra run e tiene tutta la statistica
  pop      i run vengono divisi in popolazioni separando sul salto piu' grande fra
           i picchi (se supera SPLIT_MIN); ogni popolazione ha il suo fit, e il
           valore riportato e' la media pesata

Uso:
  python3 quattro_metodi.py --base . --outdir plot/metodi --besdir plot/bes \
      [--resistances 340] [--exclude-runs 20592]
"""

import argparse, os, glob, re, json, math
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

SCALE = {340: 3500/150., 400: 1080/40., 500: 3340/100.}
NBINS, XLO, XHI = 8000, 0., 8000.
ETA0, PHI0, SEL = 18., 6., 0.2
A_TOT_MIN = 100.
SPLIT_MIN = 1.0          # salto minimo fra picchi, in %, per dichiarare due popolazioni
SYNC_C = 1.92e-7         # sigma_sincrotrone [%] = SYNC_C * E^2.5

CUT = ("$|\\mathrm{pos\\_eta}-18| \\leq 0.2$,  $|\\mathrm{pos\\_phi}-6| \\leq 0.2$")

VERA = {20:20.00, 30:30.00, 40:39.99, 50:49.98, 60:59.97, 80:79.90, 100:99.75,
        120:119.48, 150:148.73, 175:172.67, 200:196.08, 225:218.82, 250:240.76,
        275:261.77, 300:281.74}

FILES = {340: ("reco_340ohm", "*_merged.root"),
         400: ("reco_400ohm", "*_400_merged.root"),
         500: ("reco_500ohm", "*_500_merged.root")}

PARS = ("alpha_l", "alpha_h", "n_l", "n_h", "mean", "sigma", "N")


def dcb_func(x, alpha_l, alpha_h, n_l, n_h, mean, sigma, N):
    t = (x - mean) / sigma
    out = np.empty_like(t)
    core = (t >= -alpha_l) & (t <= alpha_h)
    low, high = t < -alpha_l, t > alpha_h
    out[core] = np.exp(-0.5 * t[core] ** 2)
    if np.any(low):
        f2 = (n_l/alpha_l) - alpha_l - t[low]
        out[low] = np.exp(-0.5*alpha_l**2) * np.power(np.maximum(alpha_l/n_l*f2, 1e-12), -n_l)
    if np.any(high):
        f2 = (n_h/alpha_h) - alpha_h + t[high]
        out[high] = np.exp(-0.5*alpha_h**2) * np.power(np.maximum(alpha_h/n_h*f2, 1e-12), -n_h)
    return N * out


def hist_stats(c, x, lo, hi):
    m = (x >= lo) & (x <= hi)
    cc, xx = c[m], x[m]
    tot = cc.sum()
    if tot <= 0:
        return np.nan, np.nan, 0.
    mu = (cc*xx).sum()/tot
    return mu, np.sqrt(max((cc*(xx-mu)**2).sum()/tot, 0.)), tot


def fit_window(v, energy, resistance):
    c, e = np.histogram(v, bins=NBINS, range=(XLO, XHI))
    c = c.astype(float); x = 0.5*(e[:-1]+e[1:])
    sc = SCALE[resistance]
    lo, hi = sc*energy*0.95, sc*energy*1.05
    for _ in range(2):
        mu, rms, tot = hist_stats(c, x, lo, hi)
        if not np.isfinite(mu) or rms <= 0:
            return None
        lo, hi = mu-3*rms, mu+3*rms
    return lo, hi


def fd_binwidth(v):
    if len(v) < 20:
        return 1.
    q25, q75 = np.percentile(v, [25, 75])
    return float(max(1., round(2*(q75-q25)/max(len(v), 1)**(1./3.))))


def mode_window(v, energy, resistance):
    """Ripiego per i run in cui il picco non cade nella finestra di fit.sh
    (es. 340 ohm 275 GeV, popolazione bassa a 5996 ADC con finestra da 6095)."""
    nom = SCALE[resistance]*energy
    x = v[(v > 0.5*nom) & (v < 1.3*nom)]
    if len(x) < 100:
        return None
    c, e = np.histogram(x, bins=150)
    mode = 0.5*(e[c.argmax()] + e[c.argmax()+1])
    core = x[np.abs(x-mode) < 0.08*nom]
    if len(core) < 50 or core.std() <= 0:
        return None
    return mode-3*core.std(), mode+3*core.std()


def _win_ok(w, v):
    if w is None:
        return False
    lo, hi = w
    return hi > lo and ((v >= lo) & (v <= hi)).sum() >= 200


def fit_dcb(v, energy, resistance):
    """Fit DCB con finestra di fit.sh e binning Freedman-Diaconis; se la finestra
    non contiene il picco si ripiega su una centrata sulla moda."""
    w = fit_window(v, energy, resistance)
    if not _win_ok(w, v):
        w = mode_window(v, energy, resistance)
    if not _win_ok(w, v):
        return None
    lo, hi = w
    inwin = v[(v >= lo) & (v <= hi)]
    if len(inwin) < 200:
        return None
    bw = fd_binwidth(inwin)
    nb = max(int(round((hi-lo)/bw)), 12)
    edges = np.linspace(lo, hi, nb+1)
    y, _ = np.histogram(v, bins=edges)
    y = y.astype(float)
    x = 0.5*(edges[:-1]+edges[1:])
    sel = y > 0
    if sel.sum() < 12:
        return None
    x, y = x[sel], y[sel]
    ey = np.sqrt(y)
    seed = dict(alpha_l=2., alpha_h=2., n_l=2., n_h=2.,
                mean=(y*x).sum()/y.sum(), sigma=0.5*(hi-lo)/3., N=float(y.max()))
    best = None
    for _ in range(3):
        m = Minuit(LeastSquares(x, y, ey, dcb_func), **seed)
        m.limits["alpha_l"] = (0.1, 10); m.limits["alpha_h"] = (0.1, 10)
        m.limits["n_l"] = (1, 10); m.limits["n_h"] = (1, 10)
        m.limits["mean"] = (lo, hi); m.limits["sigma"] = (0, hi-lo); m.limits["N"] = (0, None)
        m.migrad(); m.hesse(); best = m
        seed = {p: m.values[p] for p in seed}
    if (best.covariance is None or best.errors["sigma"] <= 0 or best.errors["mean"] <= 0):
        m = Minuit(LeastSquares(x, y, ey, dcb_func), **{p: best.values[p] for p in PARS})
        for p in ("alpha_l", "alpha_h", "n_l", "n_h"):
            m.limits[p] = best.limits[p]; m.fixed[p] = True
        m.limits["mean"] = (lo, hi); m.limits["sigma"] = (0, hi-lo); m.limits["N"] = (0, None)
        m.migrad(); m.hesse()
        if m.errors["sigma"] > 0 and m.errors["mean"] > 0:
            best = m
    return dict(peak=float(best.values["mean"]), err_peak=float(best.errors["mean"]),
                sigma=float(best.values["sigma"]), err_sigma=float(best.errors["sigma"]),
                chi2=float(best.fval), ndf=max(len(x)-7, 1), nev=int(len(inwin)))


def rel(r):
    """sigma/mu in % e il suo errore."""
    v = 100*r["sigma"]/r["peak"]
    e = v*math.sqrt((r["err_sigma"]/r["sigma"])**2 + (r["err_peak"]/r["peak"])**2)
    return v, e


def split_pop(peaks, ns):
    """Divide i run sul salto relativo piu' grande fra picchi consecutivi."""
    o = np.argsort(peaks)
    p = np.array(peaks)[o]
    if len(p) < 2:
        return [list(range(len(peaks)))], 0.
    gaps = 100*np.diff(p)/p[:-1]
    k = int(np.argmax(gaps))
    if gaps[k] < SPLIT_MIN:
        return [list(range(len(peaks)))], float(gaps[k])
    return [list(o[:k+1]), list(o[k+1:])], float(gaps[k])


METHODS = ("global", "runmean", "norm", "pop")
MCOL = {"global": "k", "runmean": "C0", "norm": "C3", "pop": "C2"}
MMK = {"global": "o", "runmean": "s", "norm": "^", "pop": "v"}


def analyse(path, energy, resistance, drop):
    t = uproot.open(path)["h4_reco"]
    arr = t.arrays(["run", "A_tot", "pos_eta", "pos_phi"], library="np")
    k = ((np.abs(arr["pos_eta"]-ETA0) <= SEL) & (np.abs(arr["pos_phi"]-PHI0) <= SEL)
         & (arr["A_tot"] > A_TOT_MIN))
    if drop:
        k &= ~np.isin(arr["run"], drop)
    run, at = arr["run"][k], arr["A_tot"][k]
    if len(at) < 500:
        return None
    runs = sorted(np.unique(run))
    out = dict(energy=energy, resistance=resistance, nrun=len(runs), nev=len(at), runs=runs)

    per = []
    for r in runs:
        v = at[run == r]
        f = fit_dcb(v, energy, resistance)
        per.append(None if f is None else dict(run=int(r), n=len(v), **f))
    ok = [p for p in per if p]
    out["per_run"] = per
    if not ok:
        return None

    # ---- global
    g = fit_dcb(at, energy, resistance)
    out["global"] = rel(g) if g else (np.nan, np.nan)
    out["global_chi2ndf"] = g["chi2"]/g["ndf"] if g else np.nan
    out["peak_global"] = g["peak"] if g else np.nan

    # ---- runmean
    x = np.array([rel(p)[0] for p in ok]); ex = np.array([rel(p)[1] for p in ok])
    w = np.array([p["n"] for p in ok], float)
    out["runmean"] = (float((x*w).sum()/w.sum()),
                      float(np.sqrt((w**2*ex**2).sum())/w.sum()))

    # ---- norm: ogni evento riscalato al picco di riferimento
    pref = float((np.array([p["peak"] for p in ok])*w).sum()/w.sum())
    scaled = at.astype(float).copy()
    for p in ok:
        scaled[run == p["run"]] *= pref/p["peak"]
    nf = fit_dcb(scaled, energy, resistance)
    out["norm"] = rel(nf) if nf else (np.nan, np.nan)
    out["norm_chi2ndf"] = nf["chi2"]/nf["ndf"] if nf else np.nan

    # ---- pop: popolazioni separate
    groups, gap = split_pop([p["peak"] for p in ok], w)
    out["gap_pct"] = gap
    out["npop"] = len(groups)
    vals, errs, wts, det = [], [], [], []
    for gidx in groups:
        rs = [ok[i]["run"] for i in gidx]
        v = at[np.isin(run, rs)]
        f = fit_dcb(v, energy, resistance)
        if f is None:
            continue
        a, b = rel(f)
        vals.append(a); errs.append(b); wts.append(float(len(v)))
        det.append(dict(runs=rs, nev=int(len(v)), peak=f["peak"], sigma_over_mu=a,
                        err=b, chi2ndf=f["chi2"]/f["ndf"]))
    out["pop_detail"] = det
    if vals:
        vals = np.array(vals); errs = np.array(errs); wts = np.array(wts)
        out["pop"] = (float((vals*wts).sum()/wts.sum()),
                      float(np.sqrt((wts**2*errs**2).sum())/wts.sum()))
    else:
        out["pop"] = (np.nan, np.nan)
    return out


def load_bes(besdir, R):
    b = {}
    f = os.path.join(besdir, f"rereco_{R}_withBES.csv")
    if not os.path.exists(f):
        return b
    for i, l in enumerate(open(f)):
        if i == 0:
            continue
        p = l.strip().split(",")
        if len(p) >= 7:
            b[int(float(p[0]))] = float(p[6])
    return b


def nsc(x, N, S, C):
    return np.sqrt((100*N/x)**2 + (S/np.sqrt(x))**2 + C**2)


def fit_nsc(E, y, ey):
    m = Minuit(LeastSquares(E, y, ey, nsc), N=0.3, S=3., C=0.3)
    m.limits["N"] = (0, 2); m.limits["S"] = (0, 20); m.limits["C"] = (0, 5)
    m.migrad(); m.hesse()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--outdir", default="plot/metodi")
    ap.add_argument("--besdir", default="plot/bes")
    ap.add_argument("--resistances", nargs="+", type=int, default=[340, 400, 500])
    ap.add_argument("--energies", nargs="*", type=int, default=None)
    ap.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    ap.add_argument("--only-collect", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cache = os.path.join(a.outdir, "_cache"); os.makedirs(cache, exist_ok=True)

    for R in a.resistances:
        d, pat = FILES[R]
        files = sorted(glob.glob(os.path.join(a.base, d, pat)),
                       key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)))
        for f in files:
            E = int(re.match(r"(\d+)", os.path.basename(f)).group(1))
            cf = os.path.join(cache, f"{R}_{E}.json")
            if a.only_collect or (a.energies is not None and E not in a.energies):
                continue
            print(f"  {R} ohm {E:4d} GeV", flush=True)
            res = analyse(f, E, R, a.exclude_runs)
            if res is None:
                print("      salto"); continue
            json.dump(res, open(cf, "w"), default=float)
            print(f"      global {res['global'][0]:.4f}  runmean {res['runmean'][0]:.4f}  "
                  f"norm {res['norm'][0]:.4f}  pop {res['pop'][0]:.4f}  "
                  f"({res['npop']} pop, salto {res['gap_pct']:.2f}%)", flush=True)

    # ----------------------------------------------------------- assemblaggio
    rows = []
    for R in a.resistances:
        bes = load_bes(a.besdir, R)
        for cf in sorted(glob.glob(os.path.join(cache, f"{R}_*.json")),
                         key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0])):
            c = json.load(open(cf))
            E = int(c["energy"]); Et = VERA.get(E, float(E))
            syn = SYNC_C*Et**2.5
            b = bes.get(E, 0.)
            r = dict(resistance=R, energy=E, energy_true=Et, nrun=c["nrun"], nev=c["nev"],
                     npop=c["npop"], gap_pct=c["gap_pct"], bes=b, sync=syn)
            for m in METHODS:
                v, e = c[m]
                r[m] = v; r[m+"_err"] = e
                r[m+"_corr"] = float(np.sqrt(max(v*v - b*b - syn*syn, 0.))) if np.isfinite(v) else np.nan
            rows.append(r)

    csv = os.path.join(a.outdir, "quattro_metodi.csv")
    with open(csv, "w") as fh:
        cols = (["resistance","energy","energy_true","nrun","nev","npop","gap_pct","bes","sync"]
                + [c for m in METHODS for c in (m, m+"_err", m+"_corr")])
        fh.write(",".join(cols)+"\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                              for c in cols)+"\n")
    print("->", csv)

    # ----------------------------------------------------------------- plot
    fitres = {}
    for R in a.resistances:
        rr = [r for r in rows if r["resistance"] == R]
        if not rr:
            continue
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                gridspec_kw=dict(height_ratios=[2, 1]))
        box = []
        for m in METHODS:
            E = np.array([r["energy_true"] for r in rr])
            y = np.array([r[m+"_corr"] for r in rr])
            ey = np.array([r[m+"_err"] for r in rr])
            good = np.isfinite(y) & (y > 0) & np.isfinite(ey) & (ey > 0)
            axs[0].errorbar(E[good], y[good], yerr=ey[good], fmt=MMK[m]+"-",
                            color=MCOL[m], capsize=3, ms=6, label=m)
            if good.sum() >= 4:
                mi = fit_nsc(E[good], y[good], ey[good])
                fitres[(R, m)] = mi
                box.append(f"{m:<8s} N={1000*mi.values['N']:5.0f} MeV  "
                           f"S={mi.values['S']:5.2f}%  C={mi.values['C']:5.3f}%  "
                           f"$\\chi^2$/ndf={mi.fval:6.1f}/{good.sum()-3}")
                xs = np.linspace(E[good].min(), E[good].max(), 300)
                axs[0].plot(xs, nsc(xs, *mi.values), "--", color=MCOL[m], lw=1, alpha=.6)
        axs[0].set_ylabel("$\\sigma/E$ [%]  ($\\ominus$ BES $\\ominus$ sincrotrone)")
        axs[0].set_title(f"{R} $\\Omega$   $\\quad$   $A_{{tot}} > {A_TOT_MIN:.0f}$ ADC,  {CUT}",
                         fontsize=12)
        axs[0].legend(fontsize=10); axs[0].grid(alpha=.3)
        if box:
            axs[0].text(.98, .95, "\n".join(box), transform=axs[0].transAxes,
                        ha="right", va="top", fontsize=8.5, family="monospace",
                        bbox=dict(fc="white", ec="0.5"))
        ref = np.array([r["runmean_corr"] for r in rr])
        for m in METHODS:
            if m == "runmean":
                continue
            E = np.array([r["energy_true"] for r in rr])
            y = np.array([r[m+"_corr"] for r in rr])
            g = np.isfinite(y) & np.isfinite(ref) & (ref > 0)
            axs[1].plot(E[g], 100*(y[g]/ref[g]-1), MMK[m]+"-", color=MCOL[m], ms=6, label=m)
        axs[1].axhline(0, color=MCOL["runmean"], lw=1.2)
        axs[1].set_ylabel("rispetto a runmean [%]")
        axs[1].set_xlabel("$E_{true}$ [GeV]")
        axs[1].grid(alpha=.3); axs[1].legend(fontsize=9)
        fig.tight_layout()
        p = os.path.join(a.outdir, f"quattro_metodi_{R}ohm.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print("->", p)

    with open(os.path.join(a.outdir, "quattro_metodi_fit.csv"), "w") as fh:
        fh.write("resistance,metodo,N_MeV,err_N_MeV,S_pct,err_S,C_pct,err_C,chi2,ndf\n")
        for (R, m), mi in sorted(fitres.items()):
            n = sum(1 for r in rows if r["resistance"] == R and np.isfinite(r[m+"_corr"]))
            fh.write(f"{R},{m},{1000*mi.values['N']:.1f},{1000*mi.errors['N']:.1f},"
                     f"{mi.values['S']:.4f},{mi.errors['S']:.4f},{mi.values['C']:.4f},"
                     f"{mi.errors['C']:.4f},{mi.fval:.2f},{n-3}\n")
    print("-> quattro_metodi_fit.csv")


if __name__ == "__main__":
    main()
