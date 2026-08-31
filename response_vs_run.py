"""
Response per GeV against run number: where the readout state changes.

The question this answers is what happens *between* the runs that drift. Comparing
runs taken at different energies needs a common yardstick, so what is plotted is

    median(A_tot) / E_true      [ADC / GeV]

one point per run, ordered by run number. The 3x3 sum is uncalibrated and not
perfectly linear, so this quantity carries the residual non-linearity as well
(about +-1 % peak to peak, see linearity.py); anything beyond that is a change of
state of the readout, not of the beam.

Run 20592 is dropped: it is labelled 150 GeV but its median A_tot matches 80 GeV,
so E_true is not the one on the label and the ratio would be meaningless.

Usage: python3 plot/response_vs_run.py --base . --outdir plot [--resistances 340]
"""

import argparse, os, glob, re
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import runsets

VERA = {20: 20.00, 30: 30.00, 40: 39.99, 50: 49.98, 60: 59.97, 80: 79.90,
        100: 99.75, 120: 119.48, 150: 148.73, 175: 172.67, 200: 196.08,
        225: 218.82, 250: 240.76, 275: 261.77}
FILES = {340: ("reco_340ohm", "*_merged.root"),
         400: ("reco_400ohm", "*_400_merged.root"),
         500: ("reco_500ohm", "*_500_merged.root")}
BAD_LABEL = (20592,)          # nominal energy does not match the data
NMIN = 50


def scan(base, R):
    d, pat = FILES[R]
    out = []
    for f in sorted(glob.glob(os.path.join(base, d, pat)),
                    key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1))):
        E = int(re.match(r"(\d+)", os.path.basename(f)).group(1))
        a = uproot.open(f)["h4_reco"].arrays(["run", "A_tot"], library="np")
        for r in sorted(int(x) for x in np.unique(a["run"])):
            if r in BAD_LABEL:
                continue
            m = a["run"] == r
            if m.sum() < NMIN:
                continue
            out.append((r, E, float(np.median(a["A_tot"][m])) / VERA[E], int(m.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--outdir", default="plot")
    ap.add_argument("--resistances", nargs="+", type=int, default=[340])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    f50, h275 = set(runsets.FILTER_50MHZ), set(runsets.HIGH_275)
    for R in a.resistances:
        rows = scan(a.base, R)
        if not rows:
            continue
        run = np.array([x[0] for x in rows])
        E = np.array([x[1] for x in rows])
        y = np.array([x[2] for x in rows])
        ref = float(np.median(y[run < min(f50)])) if (run < min(f50)).any() else float(np.median(y))
        dev = 100 * (y / ref - 1)

        fig, ax = plt.subplots(figsize=(13, 5.5))
        norm = matplotlib.colors.LogNorm(vmin=E.min(), vmax=E.max())
        sc = ax.scatter(run, dev, c=E, cmap="viridis", norm=norm, s=55,
                        edgecolor="k", linewidth=.4, zorder=3)
        fig.colorbar(sc, ax=ax, label="$E_{nom}$ [GeV]")
        for s, lab, col in ((f50, "50 MHz filter", "C3"), (h275, "275 GeV high population", "C1")):
            m = np.isin(run, list(s))
            if m.any():
                ax.scatter(run[m], dev[m], facecolor="none", edgecolor=col, s=170,
                           linewidth=1.8, label=lab, zorder=4)
        ax.axhline(0, color="0.4", lw=1)
        for lim in (-1, 1):
            ax.axhline(lim, color="0.7", lw=.8, ls=":")
        ax.set_xlabel("run number")
        ax.set_ylabel("median$(A_{tot})/E_{true}$  $-$ 1   [%]")
        ax.set_title(f"{R} $\\Omega$   $\\quad$   one point per run,  "
                     f"$N_{{run}} \\geq {NMIN}$,  dotted lines: $\\pm 1\\%$", fontsize=11)
        ax.grid(alpha=.3); ax.legend(fontsize=9)
        fig.tight_layout()
        p = os.path.join(a.outdir, f"response_vs_run_{R}ohm.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print("->", p)

        csv = os.path.join(a.outdir, f"response_vs_run_{R}ohm.csv")
        with open(csv, "w") as fh:
            fh.write("run,energy_nom,energy_true,nev,adc_per_gev,dev_pct\n")
            for (r, e, v, n), dv in zip(rows, dev):
                fh.write(f"{r},{e},{VERA[e]},{n},{v:.4f},{dv:.4f}\n")
        print("->", csv)


if __name__ == "__main__":
    main()
