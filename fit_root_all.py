"""
Prepara i punti per i fit in ROOT e rifa' gli stessi fit in Python come riscontro.

Quattro fit, due per ciascuno dei due insiemi di punti:

  insiemi   nopos   (sigma/E)^2 = (sigma/mu)^2 - BES^2 - sincrotrone^2
            pos     come sopra, meno anche POS_eff^2, la sistematica sul centroide
                    (vedi plot/uniformita_pos.py e plot/note/mattermost_uniformita.md)

  fit       indep   N, S, C liberi per ciascuna resistenza. A 500 ohm C e' fissato
                    a 0.300 %: i dati arrivano solo a 150 GeV e non lo vincolano
            common  S e C comuni alle tre resistenze, N libero per resistenza

In entrambi i casi l'errore e' stat (+) drift: la non-uniformita' e' una correzione,
non una barra, quindi non entra negli errori.

Scrive plot/root/points.csv, che la macro plot/root/fit_resolution.C legge per
rifare gli stessi fit e salvare TGraphErrors, TF1 e TCanvas in un file .root.

Uso: python3 plot/fit_root_all.py --plotdir plot --outdir plot/root
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

RES = (340, 400, 500)
COL = {340: "C0", 400: "C1", 500: "C2"}
C_FIX_500 = 0.300


def reso(x, N, S, C):
    return np.sqrt((100 * N / x) ** 2 + (S / np.sqrt(x)) ** 2 + C ** 2)


def load(path):
    d = {}
    for r in csv.DictReader(open(path)):
        y = float(r["final_pct"])
        e = float(r["err_tot_pct"])
        if not (np.isfinite(y) and y > 0 and e > 0):
            continue
        d.setdefault(int(r["resistance"]), []).append(
            (float(r["energy_true"]), int(float(r["energy_nom"])), y, e))
    return {R: sorted(v) for R, v in d.items()}


def fit_indep(x, y, e, R):
    m = Minuit(LeastSquares(x, y, e, reso), N=0.3, S=3., C=0.3)
    m.limits["N"] = (0, None); m.limits["S"] = (0, None); m.limits["C"] = (0, None)
    if R == 500:
        m.values["C"] = C_FIX_500; m.fixed["C"] = True
    m.migrad(); m.hesse()
    return m, len(x) - (2 if R == 500 else 3)


def fit_common(d):
    def chi2(S, C, N340, N400, N500):
        Np = {340: N340, 400: N400, 500: N500}
        t = 0.
        for R in RES:
            if R not in d:
                continue
            x, y, e = (np.array(v) for v in zip(*[(q[0], q[2], q[3]) for q in d[R]]))
            t += (((y - reso(x, Np[R], S, C)) / e) ** 2).sum()
        return t
    chi2.errordef = 1
    m = Minuit(chi2, S=2.5, C=0.35, N340=.30, N400=.30, N500=.25)
    for k in ("S", "C", "N340", "N400", "N500"):
        m.limits[k] = (0, None)
    m.migrad(); m.hesse()
    npt = sum(len(d[R]) for R in RES if R in d)
    return m, npt - 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plotdir", default="plot")
    ap.add_argument("--outdir", default="plot/root")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    sets = {"nopos": load(os.path.join(a.plotdir, "resolution_final_unif_raw.csv")),
            "pos": load(os.path.join(a.plotdir, "resolution_final_unif.csv"))}

    pts = os.path.join(a.outdir, "points.csv")
    with open(pts, "w") as fh:
        fh.write("dataset,resistance,energy_nom,energy_true,sigma_over_E_pct,err_pct\n")
        for name, d in sets.items():
            for R in RES:
                for et, en, y, e in d.get(R, []):
                    fh.write(f"{name},{R},{en},{et:.4f},{y:.6f},{e:.6f}\n")
    print("->", pts)

    rows = []
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for i, (name, d) in enumerate(sets.items()):
        mc, ndfc = fit_common(d)
        rows.append((name, "common", 0, np.nan, np.nan,
                     mc.values["S"], mc.errors["S"], mc.values["C"], mc.errors["C"],
                     mc.fval, ndfc))
        print(f"=== {name} / common: S = {mc.values['S']:.4f} +- {mc.errors['S']:.4f} %, "
              f"C = {mc.values['C']:.5f} +- {mc.errors['C']:.5f} %, "
              f"chi2/ndf = {mc.fval:.2f}/{ndfc}")
        for j, R in enumerate(RES):
            if R not in d:
                continue
            x, y, e = (np.array(v) for v in zip(*[(q[0], q[2], q[3]) for q in d[R]]))
            mi, ndf = fit_indep(x, y, e, R)
            rows.append((name, "indep", R, 1000 * mi.values["N"], 1000 * mi.errors["N"],
                         mi.values["S"], mi.errors["S"], mi.values["C"], mi.errors["C"],
                         mi.fval, ndf))
            NR = mc.values[f"N{R}"]
            rows.append((name, "common", R, 1000 * NR, 1000 * mc.errors[f"N{R}"],
                         mc.values["S"], mc.errors["S"], mc.values["C"], mc.errors["C"],
                         np.nan, np.nan))
            ax = axs[i][j]
            ax.errorbar(x, y, yerr=e, fmt="o", ms=6, color=COL[R], capsize=3,
                        label="$\\sigma/E$")
            xs = np.linspace(x.min() * .9, x.max() * 1.05, 300)
            ax.plot(xs, reso(xs, *mi.values), "--", lw=2, color="darkviolet",
                    label="$N$, $S$, $C$ per resistance")
            ax.plot(xs, reso(xs, NR, mc.values["S"], mc.values["C"]), "-", lw=2,
                    color="C3", label="$S$, $C$ common")
            tab = (f"indep  N {1000*mi.values['N']:5.0f} $\\pm$ {1000*mi.errors['N']:.0f} MeV  "
                   f"S {mi.values['S']:5.2f}  C {mi.values['C']:.3f}  "
                   f"$\\chi^2$/ndf {mi.fval:5.1f}/{ndf}\n"
                   f"common N {1000*NR:5.0f} $\\pm$ {1000*mc.errors[f'N{R}']:.0f} MeV  "
                   f"S {mc.values['S']:5.2f}  C {mc.values['C']:.3f}  "
                   f"$\\chi^2$/ndf {mc.fval:5.1f}/{ndfc} (global)")
            ax.text(.97, .95, tab, transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, family="monospace", bbox=dict(fc="white", ec="0.5"))
            ax.set_xscale("log")
            ax.set_xlim(0.85 * x.min(), 1.18 * x.max())
            ax.set_title(f"{R} $\\Omega$   $\\quad$   {name}", fontsize=11)
            ax.set_xlabel("$E_{true}$ [GeV]"); ax.set_ylabel("$\\sigma/E$ [%]")
            ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(a.outdir, "fit_root_all.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p)

    fs = os.path.join(a.outdir, "fit_summary.csv")
    with open(fs, "w") as fh:
        fh.write("dataset,mode,resistance,N_MeV,err_N_MeV,S_pct,err_S,C_pct,err_C,chi2,ndf\n")
        for r in rows:
            fh.write(",".join("nan" if (isinstance(v, float) and not np.isfinite(v))
                              else (f"{v:.6g}" if isinstance(v, float) else str(v))
                              for v in r) + "\n")
    print("->", fs)


if __name__ == "__main__":
    main()
