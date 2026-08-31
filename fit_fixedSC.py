"""
Fit of N/E (+) S/sqrt(E) (+) C with S and C FIXED at the values measured at 340 ohm.

Idea: 340 ohm is the resistance with the most points (12 energies, from 20 to
250 GeV) and the best conditioned fit. S and C are properties of the crystal and of
the geometry, not of the readout chain; the CATIA resistance changes the gain and
therefore the noise, that is N. If that is true, N alone should be enough as a free
parameter at 400 and 500 ohm.

Two sets of points:
  runmean  mean of the per-run sigmas, plot/resolution_final_runmean.csv
           error = stat (+) drift systematic
  corr     as above but with the response corrected event by event,
           plot/resolution_final_unif.csv
           error = stat (+) drift (+) non-uniformity systematic

For each set: 340 ohm free (N, S, C), then 400 and 500 with S and C frozen at the
values just found at 340 and N as the only free parameter.

Writes plot/root/points_resolution.csv, which the macro plot/root/fit_fixedSC.C
reads to repeat the same fits in ROOT and store TGraphErrors, TF1 and TCanvas in a
versionable .root file.

Usage: python3 plot/fit_fixedSC.py --plotdir plot --outdir plot/root
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

COL = {340: "C0", 400: "C1", 500: "C2"}


def reso(x, N, S, C):
    return np.sqrt((100 * N / x) ** 2 + (S / np.sqrt(x)) ** 2 + C ** 2)


def load(path, ecols):
    d = {}
    for r in csv.DictReader(open(path)):
        y = float(r["final_pct"])
        if not np.isfinite(y) or y <= 0:
            continue
        e = float(np.sqrt(sum(float(r[c]) ** 2 for c in ecols)))
        d.setdefault(int(r["resistance"]), []).append(
            (float(r["energy_true"]), int(float(r["energy_nom"])), y, e))
    return {R: sorted(v) for R, v in d.items()}


def do_fit(x, y, e, S=None, C=None):
    m = Minuit(LeastSquares(x, y, e, reso), N=0.3, S=(S if S else 3.), C=(C if C else 0.3))
    m.limits["N"] = (0, None)
    if S is None:
        m.limits["S"] = (0, None); m.limits["C"] = (0, None)
    else:
        m.fixed["S"] = True; m.fixed["C"] = True
    m.migrad(); m.hesse()
    nfree = 1 if S is not None else 3
    return m, len(x) - nfree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plotdir", default="plot")
    ap.add_argument("--outdir", default="plot/root")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    sets = {
        "runmean": load(os.path.join(a.plotdir, "resolution_final_runmean.csv"),
                        ["err_stat_pct", "err_syst_pct"]),
        "corr": load(os.path.join(a.plotdir, "resolution_final_unif.csv"),
                     ["err_stat_pct", "err_drift_pct", "err_unif_pct"]),
    }

    # ------------------------------------------------------- CSV for the ROOT macro
    pts = os.path.join(a.outdir, "points_resolution.csv")
    with open(pts, "w") as fh:
        fh.write("dataset,resistance,energy_nom,energy_true,sigma_over_E_pct,err_pct\n")
        for name, d in sets.items():
            for R in sorted(d):
                for et, en, y, e in d[R]:
                    fh.write(f"{name},{R},{en},{et:.4f},{y:.6f},{e:.6f}\n")
    print("->", pts)

    rows = []
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    for i, (name, d) in enumerate(sets.items()):
        x0, y0, e0 = (np.array(v) for v in zip(*[(t[0], t[2], t[3]) for t in d[340]]))
        m340, ndf340 = do_fit(x0, y0, e0)
        S0, C0 = float(m340.values["S"]), float(m340.values["C"])
        print(f"=== {name}: 340 ohm libero -> S = {S0:.4f} %, C = {C0:.5f} % "
              f"(chi2/ndf {m340.fval:.2f}/{ndf340})")
        for j, R in enumerate((340, 400, 500)):
            if R not in d:
                continue
            x, y, e = (np.array(v) for v in zip(*[(t[0], t[2], t[3]) for t in d[R]]))
            free, ndf_f = do_fit(x, y, e)
            if R == 340:
                fix, ndf_x = free, ndf_f
            else:
                fix, ndf_x = do_fit(x, y, e, S=S0, C=C0)
            ax = axs[i][j]
            ax.errorbar(x, y, yerr=e, fmt="o", ms=6, color=COL[R], capsize=3,
                        label="$\\sigma/E$")
            xs = np.linspace(x.min() * .9, x.max() * 1.05, 300)
            ax.plot(xs, reso(xs, *free.values), "--", lw=1.6, color="0.35",
                    label="$N$, $S$, $C$ free")
            if R != 340:
                ax.plot(xs, reso(xs, *fix.values), "-", lw=2.2, color="darkviolet",
                        label="$S$, $C$ fixed at 340 $\\Omega$")
            tab = (f"free   N {1000*free.values['N']:5.0f} $\\pm$ {1000*free.errors['N']:.0f} MeV  "
                   f"S {free.values['S']:5.2f}  C {free.values['C']:.3f}  "
                   f"$\\chi^2$/ndf {free.fval:6.1f}/{ndf_f}")
            if R != 340:
                tab += (f"\nfixed  N {1000*fix.values['N']:5.0f} $\\pm$ {1000*fix.errors['N']:.0f} MeV  "
                        f"S {S0:5.2f}  C {C0:.3f}  "
                        f"$\\chi^2$/ndf {fix.fval:6.1f}/{ndf_x}")
            ax.text(.97, .95, tab, transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, family="monospace",
                    bbox=dict(fc="white", ec="0.5"))
            ax.set_xscale("log")
            ax.set_xlim(0.85*x.min(), 1.18*x.max())
            ax.set_title(f"{R} $\\Omega$   $\\quad$   {name}", fontsize=11)
            ax.set_xlabel("$E_{true}$ [GeV]"); ax.set_ylabel("$\\sigma/E$ [%]")
            ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
            modes = [("free", free, ndf_f)]
            if R != 340:                 # a 340 ohm il fit "fixedSC" e' il fit libero
                modes.append(("fixedSC", fix, ndf_x))
            for tag, mi, nd in modes:
                rows.append((name, R, tag, 1000 * mi.values["N"], 1000 * mi.errors["N"],
                             mi.values["S"], mi.errors["S"], mi.values["C"],
                             mi.errors["C"], mi.fval, nd, len(x)))
    fig.tight_layout()
    p = os.path.join(a.outdir, "fit_fixedSC.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("->", p)

    fs = os.path.join(a.outdir, "fit_fixedSC_summary.csv")
    with open(fs, "w") as fh:
        fh.write("dataset,resistance,mode,N_MeV,err_N_MeV,S_pct,err_S,C_pct,err_C,"
                 "chi2,ndf,npoints\n")
        for r in rows:
            fh.write(",".join(f"{v:.5g}" if isinstance(v, float) else str(v)
                              for v in r) + "\n")
    print("->", fs)
    print()
    print(f'{"dataset":>8} {"R":>5} {"mode":>8} {"N (MeV)":>16} {"S (%)":>8} {"C (%)":>8} {"chi2/ndf":>12}')
    for r in rows:
        print(f"{r[0]:>8} {r[1]:>5} {r[2]:>8} {r[3]:8.1f} +- {r[4]:4.1f} {r[5]:8.3f} "
              f"{r[7]:8.4f} {r[9]:8.2f}/{r[10]:<3d}")


if __name__ == "__main__":
    main()
