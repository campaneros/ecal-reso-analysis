"""
Perche' il profilo <A_tot> vs centroide a 340 ohm 150 GeV e' cambiato dopo il re-merge.

Prima il file conteneva il solo run 20535. Ora contiene anche 20592 e 20598.
  20592  mediana A_tot 1883: non e' a 150 GeV, non entra nella finestra di fit
  20598  mediana A_tot 3519 contro 3408, cioe' +3.3%: e' la popolazione ALTA,
         e per di piu' e' centrato altrove, <pos_eta-18> = +0.039 contro -0.022

Quindi <A_tot>(eta) non e' piu' una curva di risposta ma la composizione dei run:
dove 20598 pesa di piu' la media sale del 3.3%, e nasce la rampa al posto della
parabola. Misurato con una quadratica a + b x + c x^2 su |x| < 0.3, curvatura
relativa c/a in % per cristallo^2, finestra picco +- 10 sigma:

  20535 solo             -10.76      chi2/ndf   5.3     <- il vecchio plot
  20535 + 20598 grezzi    +2.20      chi2/ndf  71.8     <- il nuovo plot, senza senso
  20535 + 20598 normal.   -9.55      chi2/ndf   5.1     <- rimedio

Rimedio: normalizzare ogni evento alla mediana del proprio run prima del profilo,
che e' quello che fa uniformita_pos.py per costruire la superficie di risposta.

Nota sul fit: qui la parabola e' scritta a + b x + c x^2 e NON p1 + p2 (x - p0)^2.
La seconda forma ha vertice e curvatura degeneri -- uno spostamento del vertice si
compensa con un cambio di curvatura -- e su
questi dati scappa: sul profilo normalizzato in eta dava p0 = +7.6 e p2 = -10.9,
cioe' un fit senza significato, mentre la quadratica semplice da' -9.55%.

Uso: python3 plot/diagnosi_profilo_150.py --file reco_340ohm/150_merged.root \
        --energy 150 --resistance 340 --outdir plot/metodi
"""
import argparse, os
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETA0, PHI0, SEL = 18., 6., 0.2
HALF, NB, FIT_HALF, NMIN = 0.6, 96, 0.3, 20
A_TOT_MIN, NSIG = 100., 10.


def profile(x, y):
    H, e = np.histogram(x, bins=NB, range=(-HALF, HALF))
    S, _ = np.histogram(x, bins=NB, range=(-HALF, HALF), weights=y)
    S2, _ = np.histogram(x, bins=NB, range=(-HALF, HALF), weights=y ** 2)
    c = 0.5 * (e[:-1] + e[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = S / np.maximum(H, 1)
        var = S2 / np.maximum(H, 1) - m ** 2
        er = np.sqrt(np.maximum(var, 0) / np.maximum(H, 1))
    ok = (H >= NMIN) & (er > 0)
    return c[ok], m[ok], er[ok]


def quad_fit(c, m, er):
    """a + b x + c x^2 con minimi quadrati lineari: nessuna degenerazione."""
    k = np.abs(c) <= FIT_HALF
    if k.sum() < 8:
        return None
    X = np.column_stack([np.ones(k.sum()), c[k], c[k] ** 2]) / er[k][:, None]
    y = m[k] / er[k]
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    cov = np.linalg.inv(X.T @ X)
    res = y - X @ co
    return dict(co=co, err=np.sqrt(np.diag(cov)), chi2=float((res ** 2).sum()),
                ndf=int(k.sum()) - 3, rel=100 * co[2] / co[0],
                err_rel=100 * np.sqrt(cov[2, 2]) / co[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="reco_340ohm/150_merged.root")
    ap.add_argument("--energy", type=int, default=150)
    ap.add_argument("--resistance", type=int, default=340)
    ap.add_argument("--outdir", default="plot/metodi")
    ap.add_argument("--window", nargs=2, type=float, default=[3275., 3573.],
                    help="finestra su A_tot, per difetto quella del plot originale")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    lo, hi = a.window

    t = uproot.open(a.file)["h4_reco"]
    arr = t.arrays(["run", "A_tot", "pos_eta", "pos_phi"], library="np")
    run = arr["run"]
    at = arr["A_tot"].astype(float)
    u = arr["pos_eta"] - ETA0
    v = arr["pos_phi"] - PHI0

    runs = sorted(int(r) for r in np.unique(run))
    med = {r: float(np.median(at[run == r])) for r in runs}
    inwin = {r: int(((run == r) & (at > lo) & (at < hi)).sum()) for r in runs}
    good = [r for r in runs if inwin[r] > 500]
    for r in runs:
        print(f"  run {r}  mediana A_tot {med[r]:8.1f}  in finestra {inwin[r]:7d}"
              + ("" if r in good else "   -> fuori dalla finestra, ignorato"))

    scale = np.ones(len(at))
    for r in good:
        m = (run == r) & (at > lo) & (at < hi) & (np.abs(u) <= SEL) & (np.abs(v) <= SEL)
        scale[run == r] = np.median(at[m])
    ref = np.median(at[(run == good[0]) & (at > lo) & (at < hi)])

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    for j, (x, lab, oth) in enumerate(((u, "pos\\_eta - 18", v), (v, "pos\\_phi - 6", u))):
        base = (at > lo) & (at < hi) & (np.abs(oth) <= SEL) & np.isin(run, good)
        for i, (yy, tag) in enumerate((
                (at, "$\\langle A_{tot} \\rangle$  [ADC]"),
                (at / scale * ref, "$\\langle A_{tot} / \\mathrm{med}(A_{tot})_{run} "
                                   "\\rangle \\times$ ref  [ADC]"))):
            ax = axes[i][j]
            for r in good:
                b = base & (run == r)
                cc, mm, ee = profile(x[b], yy[b])
                ax.plot(cc, mm, "-", lw=1.0, alpha=.8, label=f"run {r}")
            c, m, er = profile(x[base], yy[base])
            ax.errorbar(c, m, yerr=er, fmt="o", ms=3.2, lw=.9, color="C0",
                        label="all runs", zorder=4)
            f = quad_fit(c, m, er)
            if f:
                xs = np.linspace(-FIT_HALF, FIT_HALF, 200)
                ax.plot(xs, f["co"][0] + f["co"][1] * xs + f["co"][2] * xs ** 2,
                        "r-", lw=2.2, zorder=5)
                ax.text(.5, .03,
                        f"$c$ = {f['co'][2]:.1f} $\\pm$ {f['err'][2]:.1f} ADC/cr$^2$   "
                        f"$c/a$ = {f['rel']:.2f} $\\pm$ {f['err_rel']:.2f} %/cr$^2$   "
                        f"$\\chi^2$/ndf {f['chi2']:.1f} / {f['ndf']}",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5,
                        bbox=dict(fc="white", ec="0.5"))
            for s in (-SEL, SEL):
                ax.axvline(s, color="k", lw=1)
            ax.set_xlabel(f"${lab}$  [crystal units]")
            ax.set_ylabel(tag)
            ax.set_title(("$|\\mathrm{pos\\_phi}-6| \\leq 0.2$" if j == 0
                          else "$|\\mathrm{pos\\_eta}-18| \\leq 0.2$")
                         + f",   ${lo:.0f} < A_{{tot}} < {hi:.0f}$ ADC"
                         + ("" if i == 0 else ",   $A_{tot} \\to A_{tot} / \\mathrm{med}(A_{tot})_{run}$"),
                         fontsize=10)
            ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle(f"{a.resistance} $\\Omega$,  {a.energy} GeV   $\\quad$   "
                 f"runs {', '.join(str(r) for r in good)}   $\\quad$   "
                 f"fit $a + bx + cx^2$,  $|x| \\leq {FIT_HALF}$", fontsize=12)
    fig.tight_layout()
    p = os.path.join(a.outdir, f"profilo_{a.energy}GeV_{a.resistance}ohm_pernorm.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print("->", p)


if __name__ == "__main__":
    main()
