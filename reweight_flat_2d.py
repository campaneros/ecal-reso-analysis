"""
Ripesamento 2D in (pos_eta, pos_phi) a riferimento PIATTO.

Motivo: la risposta A_tot(eta, phi) e' una parabola, non una costante. Il fascio
illumina la finestra di selezione in modo non uniforme e la sua forma cambia da
run a run e da energia a energia, quindi la sigma misurata contiene un pezzo che
dipende dal fascio e non dal calorimetro. Pesando ogni evento con 1/occupancy del
suo bin (eta, phi) la distribuzione di posizione diventa piatta per costruzione e
quel pezzo viene tolto.

Peso:   w_i = c / H(bin_i),   c scelto perche' sum(w) = N
Errori: e_b = sqrt(sum w^2) nel bin b   (istogramma pesato)
Entries efficaci: N_eff = (sum w)^2 / sum(w^2)

Binning del fit: Freedman-Diaconis, h = 2*IQR/N^(1/3), arrotondato a multipli
interi di 1 ADC (il binning di fit.sh) con minimo 1 ADC. Lo STESSO binning e la
STESSA finestra si usano per l'istogramma originale e per quello pesato, cosi'
i due chi2 hanno lo stesso ndf e sono confrontabili.

Uso:
  python3 reweight_flat_2d.py --file reco_340ohm/100_merged.root \
      --energy 100 --resistance 340 --outdir plot/reweight
"""

import argparse, os, math
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALE = {340: 3500 / 150., 400: 1080 / 40., 500: 3340 / 100.}
NBINS, XLO, XHI = 8000, 0., 8000.
ETA0, PHI0, SEL = 18., 6., 0.2          # finestra di selezione di fit.sh
CUT = ("$|\\mathrm{pos\\_eta}-18| \\leq 0.2$,  $|\\mathrm{pos\\_phi}-6| \\leq 0.2$")

# soglia di rumore, identica a drift_dcb_all.py: serve perche' una frazione degli
# eventi ha A_tot ~ 0 (trigger vuoti) e il loro centroide e' rumore puro; lasciarli
# dentro sporca sia l'occupancy che determina i pesi sia la mappa di risposta.
A_TOT_MIN = 100.


def a_tot_min(energy, resistance):
    return A_TOT_MIN


# ------------------------------------------------------------------ DCB
def dcb_func(x, alpha_l, alpha_h, n_l, n_h, mean, sigma, N):
    t = (x - mean) / sigma
    out = np.empty_like(t)
    core = (t >= -alpha_l) & (t <= alpha_h)
    low, high = t < -alpha_l, t > alpha_h
    out[core] = np.exp(-0.5 * t[core] ** 2)
    if np.any(low):
        f2 = (n_l / alpha_l) - alpha_l - t[low]
        out[low] = np.exp(-0.5 * alpha_l ** 2) * np.power(
            np.maximum(alpha_l / n_l * f2, 1e-12), -n_l)
    if np.any(high):
        f2 = (n_h / alpha_h) - alpha_h + t[high]
        out[high] = np.exp(-0.5 * alpha_h ** 2) * np.power(
            np.maximum(alpha_h / n_h * f2, 1e-12), -n_h)
    return N * out


PARS = ("alpha_l", "alpha_h", "n_l", "n_h", "mean", "sigma", "N")


# ------------------------------------------------------------- finestra
def hist_stats(counts, centers, lo, hi):
    m = (centers >= lo) & (centers <= hi)
    c, x = counts[m], centers[m]
    tot = c.sum()
    if tot <= 0:
        return np.nan, np.nan, 0.
    mean = (c * x).sum() / tot
    return mean, np.sqrt(max((c * (x - mean) ** 2).sum() / tot, 0.)), tot


def fit_window(values, energy, resistance):
    """Finestra di fit.sh: scale*E*(0.95,1.05) -> mean +- 3 RMS, due volte."""
    counts, edges = np.histogram(values, bins=NBINS, range=(XLO, XHI))
    counts = counts.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sc = SCALE[resistance]
    lo, hi = sc * energy * 0.95, sc * energy * 1.05
    for _ in range(2):
        mean, rms, tot = hist_stats(counts, centers, lo, hi)
        if not np.isfinite(mean) or rms <= 0:
            return None
        lo, hi = mean - 3 * rms, mean + 3 * rms
    return lo, hi


def fd_binwidth(v, n_eff=None):
    """Freedman-Diaconis sugli eventi in finestra, arrotondato a interi ADC."""
    if len(v) < 20:
        return 1.
    q25, q75 = np.percentile(v, [25, 75])
    n = n_eff if n_eff else len(v)
    h = 2 * (q75 - q25) / max(n, 1) ** (1. / 3.)
    return float(max(1., round(h)))


# ----------------------------------------------------------------- fit
def fit_dcb(v, w, lo, hi, binw, seed_sigma):
    """Fit DCB su istogramma (eventualmente pesato) con binning dato."""
    from iminuit import Minuit
    from iminuit.cost import LeastSquares

    nb = max(int(round((hi - lo) / binw)), 10)
    edges = np.linspace(lo, hi, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if w is None:
        y, _ = np.histogram(v, bins=edges)
        y = y.astype(float)
        ey = np.sqrt(np.maximum(y, 1.))
    else:
        y, _ = np.histogram(v, bins=edges, weights=w)
        y2, _ = np.histogram(v, bins=edges, weights=w ** 2)
        ey = np.sqrt(np.maximum(y2, 0.))
        ey = np.where(ey > 0, ey, np.nan)

    sel = (y > 0) & np.isfinite(ey) & (ey > 0)
    x, yy, eyy = centers[sel], y[sel], ey[sel]
    if len(x) < 12:
        return None

    mean0 = (yy * x).sum() / yy.sum()
    seed = dict(alpha_l=2., alpha_h=2., n_l=2., n_h=2.,
                mean=mean0, sigma=seed_sigma, N=float(yy.max()))
    best = None
    for _ in range(3):
        m = Minuit(LeastSquares(x, yy, eyy, dcb_func), **seed)
        m.limits["alpha_l"] = (0.1, 10)
        m.limits["alpha_h"] = (0.1, 10)
        m.limits["n_l"] = (1, 10)
        m.limits["n_h"] = (1, 10)
        m.limits["mean"] = (lo, hi)
        m.limits["sigma"] = (0, hi - lo)
        m.limits["N"] = (0, None)
        m.migrad(); m.hesse()
        best = m
        seed = {p: m.values[p] for p in seed}

    # HESSE singolare quando le code non hanno bin che le vincolano: blocco le
    # code al valore fittato e rifitto (mean, sigma, N).
    TAILS = ("alpha_l", "alpha_h", "n_l", "n_h")
    if (best.covariance is None or best.errors["sigma"] <= 0
            or best.errors["mean"] <= 0):
        m = Minuit(LeastSquares(x, yy, eyy, dcb_func),
                   **{p: best.values[p] for p in PARS})
        for p in TAILS:
            m.limits[p] = best.limits[p]
            m.fixed[p] = True
        m.limits["mean"] = (lo, hi)
        m.limits["sigma"] = (0, hi - lo)
        m.limits["N"] = (0, None)
        m.migrad(); m.hesse()
        if m.errors["sigma"] > 0 and m.errors["mean"] > 0:
            best = m

    ndf = max(len(x) - 7, 1)
    return dict(minuit=best, edges=edges, x=x, y=yy, ey=eyy, lo=lo, hi=hi,
                chi2=float(best.fval), ndf=ndf, nbin=len(x), binw=binw,
                vals={p: float(best.values[p]) for p in PARS},
                errs={p: float(best.errors[p]) for p in PARS})


# --------------------------------------------------------------- pesi
def flat_weights(eta, phi, ngrid):
    """w ∝ 1/occupancy del bin (eta, phi), normalizzato a sum(w) = N."""
    rng = [[ETA0 - SEL, ETA0 + SEL], [PHI0 - SEL, PHI0 + SEL]]
    H, xe, ye = np.histogram2d(eta, phi, bins=ngrid, range=rng)
    ix = np.clip(np.digitize(eta, xe) - 1, 0, ngrid - 1)
    iy = np.clip(np.digitize(phi, ye) - 1, 0, ngrid - 1)
    occ = H[ix, iy]
    w = np.where(occ > 0, 1. / np.maximum(occ, 1), 0.)
    w *= len(eta) / w.sum()
    return w, H, xe, ye


def chi2_prob(chi2, ndf):
    """P(chi2 > osservato) senza scipy: serie incompleta gamma regolarizzata."""
    x, k = 0.5 * chi2, 0.5 * ndf
    if x <= 0:
        return 1.
    if x < k + 1:                       # serie
        term = 1. / k
        s, n = term, 0
        while abs(term) > 1e-14 * abs(s) and n < 10000:
            n += 1
            term *= x / (k + n)
            s += term
        return float(max(0., min(1., 1. - s * math.exp(-x + k * math.log(x) - math.lgamma(k)))))
    # frazione continua per Q(k, x)
    tiny = 1e-300
    b, c, d = x + 1 - k, 1. / tiny, 1. / (x + 1 - k)
    h = d
    for i in range(1, 10000):
        an = -i * (i - k)
        b += 2
        d = an * d + b
        if abs(d) < tiny: d = tiny
        c = b + an / c
        if abs(c) < tiny: c = tiny
        d = 1. / d
        de = d * c
        h *= de
        if abs(de - 1.) < 1e-14:
            break
    return float(max(0., min(1., math.exp(-x + k * math.log(x) - math.lgamma(k)) * h)))


# ------------------------------------------------------------- disegno
def _box(ax, res, name, color, side, nent):
    v, e = res["vals"], res["errs"]
    lines = [(name, ""),
             ("Entries", f"{nent:.0f}"),
             (r"$\chi^2$ / ndf", f"{res['chi2']:.2f} / {res['ndf']}"),
             (r"$\alpha_l$", f"{v['alpha_l']:.3g} $\\pm$ {e['alpha_l']:.2g}"),
             (r"$\alpha_r$", f"{v['alpha_h']:.3g} $\\pm$ {e['alpha_h']:.2g}"),
             (r"$n_l$", f"{v['n_l']:.3g} $\\pm$ {e['n_l']:.2g}"),
             (r"$n_r$", f"{v['n_h']:.3g} $\\pm$ {e['n_h']:.2g}"),
             (r"$\mu$", f"{v['mean']:.1f} $\\pm$ {e['mean']:.1f}"),
             (r"$\sigma$", f"{v['sigma']:.2f} $\\pm$ {e['sigma']:.2f}"),
             ("N", f"{v['N']:.1f} $\\pm$ {e['N']:.1f}")]
    x0 = 0.03 if side == "left" else 0.60
    for i, (a, b) in enumerate(lines):
        y = 0.965 - i * 0.042
        if b == "":
            ax.text(x0, y, a, transform=ax.transAxes, color=color, ha="left",
                    va="top", fontsize=10, fontweight="bold")
        else:
            ax.text(x0, y, a, transform=ax.transAxes, color=color, ha="left",
                    va="top", fontsize=9.5)
            ax.text(x0 + 0.37, y, b, transform=ax.transAxes, color=color,
                    ha="right", va="top", fontsize=9.5)


def overlay_figure(orig, wght, n_orig, n_eff, energy, resistance, ngrid, outdir, tag):
    fig, ax = plt.subplots(figsize=(11, 7))
    for res, color in ((orig, "red"), (wght, "blue")):
        ax.errorbar(res["x"], res["y"], yerr=res["ey"], fmt="s", ms=2.6,
                    lw=0.9, color=color, capsize=0)
        xs = np.linspace(res["lo"], res["hi"], 800)
        ax.plot(xs, dcb_func(xs, *[res["vals"][p] for p in PARS]), "-",
                color=color, lw=1.5)
    _box(ax, wght, "hEnergyWeighted", "blue", "left", n_eff)
    _box(ax, orig, "hEnergyOriginal", "red", "right", n_orig)
    ax.set_xlabel("$A_{tot}$ [ADC]", fontsize=11)
    ax.set_ylabel("Events", fontsize=11)
    ax.set_xlim(orig["lo"], orig["hi"])
    ax.set_ylim(0, 1.75 * max(orig["y"].max(), wght["y"].max()))
    ax.set_title(f"{resistance} $\\Omega$,  {energy} GeV   $\\quad$   "
                 f"$A_{{tot}} > {a_tot_min(energy, resistance):.0f}$ ADC,  {CUT}", fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, f"reweight_overlay_{energy}GeV_{resistance}ohm{tag}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def sanity_figure(eta, phi, atot, w, H, xe, ye, energy, resistance, ngrid, outdir, tag):
    rng = [[ETA0 - SEL, ETA0 + SEL], [PHI0 - SEL, PHI0 + SEL]]
    fine = 2 * ngrid
    Hw, xf, yf = np.histogram2d(eta, phi, bins=fine, range=rng, weights=w)
    Hf, _, _ = np.histogram2d(eta, phi, bins=fine, range=rng)
    Hw = np.where(Hf > 0, Hw, np.nan)
    S, _, _ = np.histogram2d(eta, phi, bins=ngrid, range=rng, weights=atot)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(H > 0, S / np.maximum(H, 1), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    im = axes[0].pcolormesh(xe, ye, H.T, cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="events / bin")
    axes[0].set_title(f"$N_{{bin}} \\in [{H.min():.0f}, {H.max():.0f}]$", fontsize=10)

    im = axes[1].pcolormesh(xf, yf, Hw.T, cmap="viridis")
    fig.colorbar(im, ax=axes[1], label="$\\sum w$ / bin")
    axes[1].set_title(f"${fine}\\times{fine}$,   RMS/mean $= "
                      f"{100*np.nanstd(Hw)/np.nanmean(Hw):.1f}\\%$", fontsize=10)

    incl = atot.mean()
    im = axes[2].pcolormesh(xe, ye, M.T, cmap="plasma",
                            vmin=incl * 0.97, vmax=incl * 1.03)
    fig.colorbar(im, ax=axes[2], extend="both",
                 label=f"$\\langle A_{{tot}} \\rangle$ [ADC] (inclusive {incl:.0f} $\\pm$ 3%)")
    lo_m, hi_m = np.nanmin(M), np.nanmax(M)
    axes[2].set_title(f"$\\langle A_{{tot}} \\rangle / \\overline{{A_{{tot}}}} \\in "
                      f"[{lo_m/incl:.3f}, {hi_m/incl:.3f}]$", fontsize=10)
    for a in axes:
        a.set_xlabel("pos_eta"); a.set_ylabel("pos_phi")
    fig.suptitle(f"{resistance} $\\Omega$,  {energy} GeV   $\\quad$   "
                 f"$A_{{tot}} > {a_tot_min(energy, resistance):.0f}$ ADC,  {CUT}", fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, f"reweight_maps_{energy}GeV_{resistance}ohm{tag}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--energy", type=int, required=True)
    ap.add_argument("--resistance", type=int, required=True)
    ap.add_argument("--outdir", default="plot/reweight")
    ap.add_argument("--grid", type=int, default=10)
    ap.add_argument("--tag", default="")
    ap.add_argument("--csv", default="reweight_summary.csv")
    ap.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    t = uproot.open(a.file)["h4_reco"]
    arr = t.arrays(["run", "spill", "evt", "A_tot", "pos_eta", "pos_phi"], library="np")
    amin = a_tot_min(a.energy, a.resistance)
    inbox = ((np.abs(arr["pos_eta"] - ETA0) <= SEL) & (np.abs(arr["pos_phi"] - PHI0) <= SEL))
    keep = inbox & (arr["A_tot"] > amin)
    if a.exclude_runs:
        drop = np.isin(arr["run"], a.exclude_runs)
        if (keep & drop).sum():
            print(f"  scartati {int((keep & drop).sum())} eventi dei run {a.exclude_runs}")
        keep &= ~drop
    eta, phi, atot, run = (arr[k][keep] for k in ("pos_eta", "pos_phi", "A_tot", "run"))
    print(f"[{a.resistance} ohm {a.energy} GeV] {inbox.sum()} / {len(inbox)} eventi in finestra, "
          f"{keep.sum()} sopra la soglia A_tot > {amin:.0f} ADC "
          f"({100*(1-keep.sum()/max(inbox.sum(),1)):.1f}% scartati), "
          f"{len(np.unique(run))} run")

    w, H, xe, ye = flat_weights(eta, phi, a.grid)
    n_orig = float(len(atot))
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    print(f"  pesi: min {w.min():.3f} max {w.max():.3f} media {w.mean():.3f}  "
          f"| occupancy min {H.min():.0f} max {H.max():.0f}  "
          f"| N = {n_orig:.0f} -> N_eff = {n_eff:.0f} ({100*n_eff/n_orig:.1f}%)")

    win = fit_window(atot, a.energy, a.resistance)
    if win is None:
        print("  finestra non trovata"); return
    lo, hi = win
    inwin = atot[(atot >= lo) & (atot <= hi)]
    binw = fd_binwidth(inwin)
    binw_w = fd_binwidth(inwin, n_eff=n_eff * len(inwin) / n_orig)
    print(f"  finestra [{lo:.1f}, {hi:.1f}]  bin FD = {binw:.0f} ADC "
          f"(sul pesato sarebbe {binw_w:.0f} ADC; uso {binw:.0f} per entrambi)")

    seed_sigma = 0.5 * (hi - lo) / 3.
    orig = fit_dcb(atot, None, lo, hi, binw, seed_sigma)
    wght = fit_dcb(atot, w, lo, hi, binw, seed_sigma)
    if orig is None or wght is None:
        print("  fit fallito"); return

    for nm, r, ne in (("original", orig, n_orig), ("weighted", wght, n_eff)):
        p = chi2_prob(r["chi2"], r["ndf"])
        print(f"  {nm:9s}: mu = {r['vals']['mean']:.2f} +- {r['errs']['mean']:.2f}   "
              f"sigma = {r['vals']['sigma']:.3f} +- {r['errs']['sigma']:.3f}   "
              f"sigma/mu = {100*r['vals']['sigma']/r['vals']['mean']:.4f}%   "
              f"chi2/ndf = {r['chi2']:.1f}/{r['ndf']} = {r['chi2']/r['ndf']:.3f}   "
              f"p = {p:.3g}   entries = {ne:.0f}")

    d = 100 * (wght["vals"]["sigma"] / wght["vals"]["mean"]) / \
        (orig["vals"]["sigma"] / orig["vals"]["mean"]) - 100
    print(f"  --> sigma/mu ripesata / originale - 1 = {d:+.2f}%")

    p1 = overlay_figure(orig, wght, n_orig, n_eff, a.energy, a.resistance, a.grid, a.outdir, a.tag)
    p2 = sanity_figure(eta, phi, atot, w, H, xe, ye, a.energy, a.resistance, a.grid, a.outdir, a.tag)

    csv = os.path.join(a.outdir, a.csv)
    new = not os.path.exists(csv)
    with open(csv, "a") as f:
        if new:
            f.write("resistance,energy,nrun,nev,n_eff,grid,binw_adc,nbin,"
                    "peak_orig,err_peak_orig,sigma_orig,err_sigma_orig,chi2ndf_orig,p_orig,"
                    "peak_wgt,err_peak_wgt,sigma_wgt,err_sigma_wgt,chi2ndf_wgt,p_wgt,"
                    "d_sigma_over_mu_pct\n")
        f.write(f"{a.resistance},{a.energy},{len(np.unique(run))},{n_orig:.0f},{n_eff:.0f},"
                f"{a.grid},{binw:.0f},{orig['nbin']},"
                f"{orig['vals']['mean']:.4f},{orig['errs']['mean']:.4f},"
                f"{orig['vals']['sigma']:.4f},{orig['errs']['sigma']:.4f},"
                f"{orig['chi2']/orig['ndf']:.4f},{chi2_prob(orig['chi2'],orig['ndf']):.4g},"
                f"{wght['vals']['mean']:.4f},{wght['errs']['mean']:.4f},"
                f"{wght['vals']['sigma']:.4f},{wght['errs']['sigma']:.4f},"
                f"{wght['chi2']/wght['ndf']:.4f},{chi2_prob(wght['chi2'],wght['ndf']):.4g},"
                f"{d:.4f}\n")
    print("  ->", p1); print("  ->", p2); print("  ->", csv)


if __name__ == "__main__":
    main()
