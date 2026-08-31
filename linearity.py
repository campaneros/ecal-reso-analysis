#!/usr/bin/env python3
"""
Energy linearity of the uncalibrated 3x3 amplitude
==================================================
Fits peak_ADC = p0 + p1 * E to the double-CB peaks and shows the residuals in
percent underneath. Two versions:

  linearity_by_energy.png   one point per energy (the ALL row of drift_per_run,
                            i.e. all runs of that energy fitted together).
                            Errors: statistical from the fit, added in quadrature
                            to the run-to-run drift systematic where it exists.

  linearity_by_run.png      one point per RUN, to spot individual runs that fall
                            off the line. The line is the one fitted on the
                            per-energy points, so the runs are compared against
                            a common reference and not re-fitted.

Reads the CSVs produced by drift_dcb_all.py; writes linearity_fit.csv with the
fit parameters and every residual.

Usage:  python3 linearity.py --plotdir plot
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {340: "C0", 400: "C1", 500: "C2"}

# TRUE beam energy ("Final Energy" column of beamfiles/CMS_ECAL_energies_June26.xlsx)
# against the nominal setting. Above 100 GeV the difference reaches 5%, and it is the
# reason why the fit looked non-linear when the nominal energies were used.
E_TRUE = {20: 20.00, 30: 30.00, 40: 39.99, 50: 49.98, 60: 59.97, 80: 79.90,
          100: 99.75, 120: 119.48, 150: 148.73, 175: 172.67, 200: 196.08,
          225: 218.82, 250: 240.76, 275: 261.77, 300: 281.74}
TOL = 0.5        # requisito di linearita', in percento


def load(plotdir, R):
    per_energy, per_run = {}, []
    f = os.path.join(plotdir, str(R), f"drift_per_run_{R}ohm.csv")
    if not os.path.exists(f):
        return None, None
    for r in csv.DictReader(open(f)):
        E = int(r["energy"])
        if r["run"] == "ALL":
            per_energy[E] = (float(r["peak_abs"]), float(r["err_peak_abs"]))
        elif r["fit_ok"] == "1":
            per_run.append((E, int(r["run"]), float(r["peak_abs"]),
                            float(r["err_peak_abs"]), int(r["nev"])))
    syst = {}
    g = os.path.join(plotdir, str(R), f"sistematica_drift_{R}ohm.csv")
    if os.path.exists(g):
        for r in csv.DictReader(open(g)):
            syst[int(r["energy"])] = float(r["syst_picco_ADC"])
    return per_energy, per_run, syst


def fit_line(E, y, ey):
    """Minimi quadrati pesati per y = p0 + p1*E, con errori sui parametri."""
    w = 1 / ey ** 2
    S, Sx, Sy = w.sum(), (w * E).sum(), (w * y).sum()
    Sxx, Sxy = (w * E * E).sum(), (w * E * y).sum()
    d = S * Sxx - Sx ** 2
    p1 = (S * Sxy - Sx * Sy) / d
    p0 = (Sxx * Sy - Sx * Sxy) / d
    e1, e0 = np.sqrt(S / d), np.sqrt(Sxx / d)
    chi2 = (((y - (p0 + p1 * E)) / ey) ** 2).sum()
    return p0, e0, p1, e1, chi2, len(E) - 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plotdir", default="plot")
    ap.add_argument("--fitmax", type=float, default=100.,
                    help="energia massima usata per la retta di riferimento a bassa E")
    ap.add_argument("--exclude", nargs="*", default=["340:275"],
                    help="coppie R:E da escludere dal fit (default: 340:275, "
                         "dove i run sono due popolazioni distinte)")
    a = ap.parse_args()
    excl = {tuple(int(v) for v in s.split(":")) for s in a.exclude}

    data, fits = {}, {}
    for R in (340, 400, 500):
        got = load(a.plotdir, R)
        if not got or not got[0]:
            continue
        data[R] = got

    # ---------------------------------------------------- per energy
    fig, axs = plt.subplots(2, len(data), figsize=(6.2 * len(data), 8.6),
                            sharex="col", gridspec_kw=dict(height_ratios=[2, 1]))
    if len(data) == 1:
        axs = axs.reshape(2, 1)
    rowsout = []
    for j, R in enumerate(sorted(data)):
        per_energy, per_run, syst = data[R]
        Enom = np.array(sorted(per_energy))
        E = np.array([E_TRUE.get(int(e), float(e)) for e in Enom])
        y = np.array([per_energy[e][0] for e in Enom])
        est = np.array([per_energy[e][1] for e in Enom])
        esy = np.array([syst.get(int(e), 0.) for e in Enom])
        ey = np.hypot(est, esy)
        keep = np.array([(R, int(e)) not in excl for e in Enom])
        p0, e0, p1, e1, chi2, ndf = fit_line(E[keep], y[keep], ey[keep])
        fits[R] = (p0, p1)
        res = 100 * (y - (p0 + p1 * E)) / (p0 + p1 * E)
        eres = 100 * ey / (p0 + p1 * E)
        for e, yy, ee, rr, er, k in zip(Enom, y, ey, res, eres, keep):
            rowsout.append((R, int(e), "ALL", yy, ee, rr, er, int(k)))

        ax, ax2 = axs[0][j], axs[1][j]
        ax.errorbar(E[keep], y[keep], yerr=ey[keep], fmt="o", ms=6, color=COL[R],
                    capsize=3, label="fitted peak")
        if (~keep).any():
            ax.errorbar(E[~keep], y[~keep], yerr=ey[~keep], fmt="x", ms=9, mew=2,
                        color="red", capsize=3, label="excluded from the fit")
        xx = np.linspace(0, E.max() * 1.05, 10)
        ax.plot(xx, p0 + p1 * xx, "k-", lw=1.3)
        ax.set_ylabel("double-CB peak  [ADC]")
        ax.set_title(f"{R} $\\Omega$\n"
                     f"slope = {p1:.4f} $\\pm$ {e1:.4f} ADC/GeV,  "
                     f"intercept = {p0:+.1f} $\\pm$ {e0:.1f} ADC\n"
                     f"$\\chi^2$/ndf = {chi2:.0f}/{ndf}", fontsize=10)
        ax.grid(alpha=.3); ax.legend(fontsize=8)

        ax2.errorbar(E[keep], res[keep], yerr=eres[keep], fmt="o", ms=6,
                     color=COL[R], capsize=3)
        if (~keep).any():
            ax2.errorbar(E[~keep], res[~keep], yerr=eres[~keep], fmt="x", ms=9,
                         mew=2, color="red", capsize=3)
        ax2.axhline(0, color="k", lw=1.2)
        ax2.axhspan(-TOL, TOL, color="green", alpha=.12)
        for lev in (-TOL, TOL):
            ax2.axhline(lev, color="green", lw=1.2)
        ax2.set_xlabel("True beam energy [GeV]")
        ax2.set_ylabel("residual  (data - fit)/fit  [%]")
        ax2.grid(alpha=.3)
        nbad = int((np.abs(res[keep]) > TOL).sum())
        ax2.set_title(f"$\\max|\\Delta| = {np.abs(res[keep]).max():.2f}\\%$,   "
                      f"${nbad}/{int(keep.sum())}$ outside $\\pm{TOL}\\%$", fontsize=9)
    fig.suptitle("$A_{tot}^{peak} = p_0 + p_1 E_{true}$", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(a.plotdir, "linearity_by_energy.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------- per run
    fig, axs = plt.subplots(2, len(data), figsize=(6.2 * len(data), 8.6),
                            sharex="col", gridspec_kw=dict(height_ratios=[2, 1]))
    if len(data) == 1:
        axs = axs.reshape(2, 1)
    for j, R in enumerate(sorted(data)):
        per_energy, per_run, syst = data[R]
        p0, p1 = fits[R]
        ax, ax2 = axs[0][j], axs[1][j]
        Enr = np.array([r[0] for r in per_run], float)
        E = np.array([E_TRUE.get(int(e), float(e)) for e in Enr])
        y = np.array([r[2] for r in per_run])
        ey = np.array([r[3] for r in per_run])
        pred = p0 + p1 * E
        res = 100 * (y - pred) / pred
        for e, run, yy, ee, rr in zip(E, [r[1] for r in per_run], y, ey, res):
            rowsout.append((R, int(e), run, yy, ee, rr, 100 * ee / (p0 + p1 * e), 1))
        ax.errorbar(E, y, yerr=ey, fmt="o", ms=4, color=COL[R], capsize=2, alpha=.8)
        xx = np.linspace(0, E.max() * 1.05, 10)
        ax.plot(xx, p0 + p1 * xx, "k-", lw=1.3)
        ax.set_ylabel("double-CB peak per run  [ADC]")
        ax.set_title(f"{R} $\\Omega$,  $N_{{run}} = {len(E)}$", fontsize=11)
        ax.grid(alpha=.3)
        # deviation from the MEAN OF ITS OWN ENERGY: this removes the non-linearity
        # and leaves only the run-to-run spread, which is what is needed to spot a
        # run that is out of place.
        runs = np.array([r[1] for r in per_run])
        dev = np.zeros_like(y)
        for e in np.unique(E):
            k = E == e
            dev[k] = 100 * (y[k] / y[k].mean() - 1)
        ax2.errorbar(E, dev, yerr=100 * ey / y, fmt="o", ms=4, color=COL[R],
                     capsize=2, alpha=.8)
        bad = np.abs(dev) > TOL
        if bad.any():
            ax2.plot(E[bad], dev[bad], "o", ms=11, mfc="none", mec="red", mew=2)
            for e, dd, run in zip(E[bad], dev[bad], runs[bad]):
                ax2.annotate(str(run), (e, dd), fontsize=6.5, color="red",
                             xytext=(5, 3), textcoords="offset points")
        ax2.axhline(0, color="k", lw=1.2)
        ax2.axhspan(-TOL, TOL, color="green", alpha=.12)
        for lev in (-TOL, TOL):
            ax2.axhline(lev, color="green", lw=1.2)
        ax2.set_xlabel("True beam energy [GeV]")
        ax2.set_ylabel("run - mean of that energy  [%]")
        ax2.set_title(f"${int(bad.sum())}$ runs outside $\\pm{TOL}\\%$", fontsize=9)
        ax2.grid(alpha=.3)
    fig.suptitle("$A_{tot}^{peak} = p_0 + p_1 E_{true}$,  run by run", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(a.plotdir, "linearity_by_run.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(a.plotdir, "linearity_fit.csv"), "w") as fh:
        fh.write("resistance,energy,run,peak_ADC,err_peak_ADC,residual_pct,"
                 "err_residual_pct,used_in_fit\n")
        for r in rowsout:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:+.4f},"
                     f"{r[6]:.4f},{r[7]}\n")
    # ------------------------------- reference line from the low-E points only
    fig, axs = plt.subplots(2, 1, figsize=(9, 9), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1]))
    ax, ax2 = axs
    for R in sorted(data):
        per_energy, per_run, syst = data[R]
        E = np.array(sorted(per_energy))
        y = np.array([per_energy[e][0] for e in E])
        ey = np.hypot(np.array([per_energy[e][1] for e in E]),
                      np.array([syst.get(int(e), 0.) for e in E]))
        keep = np.array([(R, int(e)) not in excl for e in E])
        low = keep & (E <= a.fitmax)
        if low.sum() < 3:
            continue
        q0, _, q1, _, _, _ = fit_line(E[low], y[low], ey[low])
        r = 100 * (y - (q0 + q1 * E)) / (q0 + q1 * E)
        ax.errorbar(E[keep], r[keep], yerr=100 * ey[keep] / (q0 + q1 * E[keep]),
                    fmt="o-", ms=6, color=COL[R], capsize=3,
                    label=f"{R} $\\Omega$  (slope {q1:.3f} ADC/GeV)")
        if (~keep).any():
            ax.errorbar(E[~keep], r[~keep], fmt="x", ms=9, mew=2, color="red")
        ax2.errorbar(y[keep], r[keep], yerr=100 * ey[keep] / (q0 + q1 * E[keep]),
                     fmt="o-", ms=6, color=COL[R], capsize=3, label=f"{R} $\\Omega$")
    for A in (ax, ax2):
        A.axhline(0, color="k", lw=1.2)
        A.grid(alpha=.3); A.legend(fontsize=9)
        A.set_ylabel("deviation from the low-energy line  [%]")
    ax.axvline(a.fitmax, color="grey", ls="--", lw=1)
    ax.set_xlim(0, 290)
    ax.set_xlabel("True beam energy [GeV]")
    ax.set_title(f"$p_0 + p_1 E$ fitted on $E \\leq {a.fitmax:.0f}$ GeV", fontsize=11)
    ax2.set_xlabel("double-CB peak  [ADC]")
    ax2.set_title("", fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(a.plotdir, "linearity_lowE_reference.png"), dpi=150)
    plt.close(fig)

    for R in sorted(fits):
        p0, p1 = fits[R]
        rr = [x[5] for x in rowsout if x[0] == R and x[2] == "ALL" and x[7] == 1]
        print(f"{R} ohm: pendenza {p1:.4f} ADC/GeV, intercetta {p0:+.1f} ADC, "
              f"residui per energia: max |{max(abs(np.array(rr))):.2f}|%, RMS {np.std(rr):.2f}%")
    print("scritto linearity_by_energy.png, linearity_by_run.png, linearity_fit.csv")


if __name__ == "__main__":
    main()
