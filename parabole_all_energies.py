"""
Tutte le parabole di risposta in eta e phi, per tutte le energie di una
resistenza, sullo stesso plot -- ciascuna normalizzata al valore al vertice
della propria parabola, cosi' le curvature sono confrontabili tra energie.

Profilo: <A_tot> vs (pos_eta - 18) con |pos_phi - 6| < 0.2, e viceversa.
Parabola  y = p1 + p2*(x - p0)^2  fittata su |x| <= 0.3 (convenzione di
drift_dcb_all.py). Normalizzazione: y -> y / p1.

Uso:  python3 parabole_all_energies.py --base . --outdir plot/parabole
"""

import argparse, os, glob, re
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
from iminuit import Minuit
from iminuit.cost import LeastSquares

ETA0, PHI0 = 18., 6.
SEL_HALF = 0.2                       # taglio sull'altra coordinata
PROF_HALF, PROF_BIN, PROF_NMIN = 0.6, 0.0125, 20
FIT_HALF = 0.3                       # intervallo su cui si fitta la parabola
A_TOT_MIN = 100.
NSIG = 10.                           # finestra su A_tot: picco +- 10 sigma

FILES = {340: ("reco_340ohm", "{E}_merged.root"),
         400: ("reco_400ohm", "{E}_400_merged.root"),
         500: ("reco_500ohm", "{E}_500_merged.root")}


def _parab(x, p0, p1, p2):
    return p1 + p2 * (x - p0) ** 2


def profile(x, y, nb):
    H, edges = np.histogram(x, bins=nb, range=(-PROF_HALF, PROF_HALF))
    S, _ = np.histogram(x, bins=nb, range=(-PROF_HALF, PROF_HALF), weights=y)
    S2, _ = np.histogram(x, bins=nb, range=(-PROF_HALF, PROF_HALF), weights=y ** 2)
    c = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = S / np.maximum(H, 1)
        var = S2 / np.maximum(H, 1) - mean ** 2
        err = np.sqrt(np.maximum(var, 0) / np.maximum(H, 1))
    return c, mean, err, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--outdir", default="plot/parabole")
    ap.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    ap.add_argument("--summary", default="plot/reweight/reweight_summary_all.csv",
                    help="da qui prendo picco e sigma per la finestra su A_tot")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    S = np.genfromtxt(os.path.join(a.base, a.summary), delimiter=",", names=True)
    pk = {(int(r["resistance"]), int(r["energy"])): (r["peak_orig"], r["sigma_orig"]) for r in S}

    nb = int(2 * PROF_HALF / PROF_BIN)
    rows = []
    for R, (d, pat) in FILES.items():
        files = sorted(glob.glob(os.path.join(a.base, d, pat.format(E="*"))),
                       key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)))
        energies = [int(re.match(r"(\d+)", os.path.basename(f)).group(1)) for f in files]
        base = list(cm.tab20.colors) + list(cm.Dark2.colors)
        cols = {E: base[i % len(base)] for i, E in enumerate(energies)}
        handles = []

        fig, axs = plt.subplots(1, 2, figsize=(16, 6.6))
        for f, E in zip(files, energies):
            if (R, E) not in pk:
                print(f"  [!] {R} ohm {E} GeV: manca picco/sigma, salto"); continue
            p, s = pk[(R, E)]
            lo, hi = p - NSIG * s, p + NSIG * s
            t = uproot.open(f)["h4_reco"]
            arr = t.arrays(["A_tot", "pos_eta", "pos_phi", "run"], library="np")
            good = (arr["A_tot"] > A_TOT_MIN) & (arr["A_tot"] > lo) & (arr["A_tot"] < hi)
            if a.exclude_runs:
                good &= ~np.isin(arr["run"], a.exclude_runs)
            eta, phi, at = (arr[k][good] for k in ("pos_eta", "pos_phi", "A_tot"))
            col = cols[E]

            for ax, (xv, ov, xl) in zip(axs, (
                    (eta - ETA0, phi - PHI0, "pos_eta - 18"),
                    (phi - PHI0, eta - ETA0, "pos_phi - 6"))):
                m = np.abs(ov) < SEL_HALF
                c, mean, err, H = profile(xv[m], at[m], nb)
                ok = (H >= PROF_NMIN) & (err > 0)
                fitm = ok & (np.abs(c) <= FIT_HALF)
                if fitm.sum() < 5:
                    continue
                mi = Minuit(LeastSquares(c[fitm], mean[fitm], err[fitm], _parab),
                            p0=0., p1=float(mean[fitm].max()), p2=-100.)
                mi.migrad(); mi.hesse()
                p1 = mi.values["p1"]
                if not np.isfinite(p1) or p1 <= 0:
                    continue
                ax.errorbar(c[ok], mean[ok] / p1, yerr=err[ok] / p1, fmt="o-", ms=3.0,
                            lw=1.0, color=col, alpha=.9, capsize=0, zorder=2)
                rows.append((R, E, xl.split()[0], float(mi.values["p0"]), float(mi.errors["p0"]),
                             float(p1), float(mi.errors["p1"]),
                             float(mi.values["p2"]), float(mi.errors["p2"]),
                             float(mi.values["p2"] / p1), float(mi.fval), int(fitm.sum() - 3)))

        for ax, xl, ol in zip(axs, ("pos\\_eta - 18", "pos\\_phi - 6"),
                              ("pos\\_phi - 6", "pos\\_eta - 18")):
            for v in (-SEL_HALF, SEL_HALF):
                ax.axvline(v, color="k", lw=1)
            ax.axhline(1, color="grey", lw=.8, ls=":")
            ax.set_xlim(-PROF_HALF, PROF_HALF)
            ax.set_ylim(0.955, 1.008)
            ax.set_xlabel(f"$\\mathrm{{{xl}}}$")
            ax.set_ylabel("$\\langle A_{tot} \\rangle \\, / \\, p_1$")
            ax.set_title(f"$|\\mathrm{{{ol}}}| < {SEL_HALF}$", fontsize=11)
            ax.grid(alpha=.3)
        handles = [Line2D([], [], color=cols[E], marker="o", ms=5, lw=1.4,
                          label=f"{E} GeV") for E in energies]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.905, 0.5),
                   frameon=True, fontsize=9.5, title="Beam energy", title_fontsize=10)
        fig.subplots_adjust(right=0.895)
        fig.subplots_adjust(top=0.88, right=0.895)
        fig.suptitle(f"{R} $\\Omega$   $\\quad$   $A_{{tot}} > {A_TOT_MIN:.0f}$ ADC   "
                     f"$\\quad$   $|A_{{tot}} - \\mu| < {NSIG:.0f}\\sigma$",
                     fontsize=13, y=0.97)
        p = os.path.join(a.outdir, f"parabole_all_energies_{R}ohm.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("->", p)

    csv = os.path.join(a.outdir, "parabole_all_energies.csv")
    with open(csv, "w") as f:
        f.write("resistance,energy,coord,p0_vertex,err_p0,p1_max,err_p1,"
                "p2_curv,err_p2,p2_over_p1,chi2,ndf\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r) + "\n")
    print("->", csv)

    # curvatura relativa vs energia
    d = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, coord in zip(axs, ("pos_eta", "pos_phi")):
        for R, c in ((340, "C0"), (400, "C1"), (500, "C2")):
            m = (d["resistance"] == R) & (d["coord"] == coord)
            if m.sum() == 0:
                continue
            o = np.argsort(d["energy"][m])
            ax.errorbar(d["energy"][m][o], -100 * d["p2_over_p1"][m][o],
                        yerr=100 * d["err_p2"][m][o] / d["p1_max"][m][o],
                        fmt="o-", color=c, capsize=3, label=f"{R} $\\Omega$")
        ax.set_xscale("log")
        ax.set_xlabel("Beam energy [GeV]")
        ax.set_ylabel("$-p_2/p_1$  [% / crystal$^2$]")
        ax.set_title(f"$\\mathrm{{{coord.replace(chr(95), chr(92)+chr(95))}}}$", fontsize=11)
        ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    p = os.path.join(a.outdir, "parabole_curvature_vs_energy.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p)


if __name__ == "__main__":
    main()
