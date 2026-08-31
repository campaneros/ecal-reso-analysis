#!/usr/bin/env python3
"""
Final resolution curve: subtracting one contribution at a time
==============================================================
Starts from sigma/mu of the double-CB fit and removes, IN QUADRATURE and one at
a time, the two beam contributions that are not detector resolution:

  BES        beam energy spread, per (resistance, energy), from the `bes` column
             of rereco_<R>_withBES.csv.
             Already in percent.
  SYNCHROTRON  1.92e-7 * E^2.5  [percent], the term in fit_plot.sh
             (there written as pow(1.92e-7*pow(en,2.5),2)*1e-4 because it is
             subtracted from a squared FRACTION).

So, working in percent throughout,

    (sigma/E)^2_corr = (sigma/mu)^2 - BES^2 - SYNC^2

The x axis and the synchrotron term use the TRUE beam energy (Final Energy from
the beam log), not the nominal setting -- the two differ by up to 5% above
100 GeV, and using the nominal one fakes a 3-4% non-linearity.

Also overlaid: sigma/mu obtained by AVERAGING THE PER-RUN sigmas (weighted mean
with the drift systematic included) with its band, instead of fitting all the
runs merged together. The two differ when the runs are not homogeneous.

Error bars: statistical from the fit, added in quadrature to the run-to-run
drift systematic on sigma.

Usage:  python3 resolution_final.py --plotdir plot --besdir <dir with rereco_*_withBES.csv>
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

COL = {340: "C0", 400: "C1", 500: "C2"}
E_TRUE = {20: 20.00, 30: 30.00, 40: 39.99, 50: 49.98, 60: 59.97, 80: 79.90,
          100: 99.75, 120: 119.48, 150: 148.73, 175: 172.67, 200: 196.08,
          225: 218.82, 250: 240.76, 275: 261.77, 300: 281.74}


def sync_pct(E):
    """Synchrotron term in percent: 1.92e-7 * E^2.5 (as in fit_plot.sh)."""
    return 1.92e-7 * np.power(E, 2.5)


def reso(x, N, S, C):
    """sigma/E in PERCENT, with the same convention as fit_plot.sh:
       fraction = N/E (+) (S/100)/sqrt(E) (+) C/100   ->   N in GeV, S and C in %.
    Since everything here is in percent, the noise term has to be multiplied by 100."""
    return np.sqrt((100 * N / x) ** 2 + (S / np.sqrt(x)) ** 2 + C ** 2)


def load(plotdir, besdir, R):
    out = {}
    f = os.path.join(plotdir, str(R), f"drift_per_run_{R}ohm.csv")
    per_run = {}
    for r in csv.DictReader(open(f)):
        E = int(r["energy"]) if r["run"] != "ALL" else None
        if r["run"] == "ALL":
            E = int(r["energy"])
            p, s = float(r["peak_abs"]), float(r["sigma_abs"])
            out[E] = dict(peak=p, sigma=s, err_sigma=float(r["err_sigma_abs"]),
                          err_peak=float(r["err_peak_abs"]), rel=100 * s / p)
        elif r["fit_ok"] == "1":
            per_run.setdefault(E, []).append(
                (int(r["nev"]), float(r["sigma_abs"]), float(r["peak_abs"])))
    # mean of the per-run sigmas WEIGHTED BY THE NUMBER OF EVENTS in the run
    for E, v in per_run.items():
        if E not in out:
            continue
        n = np.array([x[0] for x in v], float)
        out[E]["sigma_runmean"] = float((n * [x[1] for x in v]).sum() / n.sum())
        out[E]["peak_runmean"] = float((n * [x[2] for x in v]).sum() / n.sum())
        out[E]["n_run_used"] = len(v)
        out[E]["n_ev_tot"] = int(n.sum())
    g = os.path.join(plotdir, str(R), f"sistematica_drift_{R}ohm.csv")
    if os.path.exists(g):
        for r in csv.DictReader(open(g)):
            E = int(r["energy"])
            if E in out:
                out[E]["syst_rel"] = float(r["syst_sigma_pct"])
                out[E]["drift_pct"] = float(r["syst_picco_pct"])
                out[E]["syst_sigma_abs"] = float(r["syst_sigma_ADC"])
                out[E]["n_run"] = int(r["n_run"])
    b = os.path.join(besdir, f"rereco_{R}_withBES.csv")
    if os.path.exists(b):
        for r in csv.DictReader(open(b)):
            E = int(r["en"])
            if E in out:
                out[E]["bes"] = float(r["bes"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plotdir", default="plot")
    ap.add_argument("--besdir", required=True)
    ap.add_argument("--exclude", nargs="*", default=["340:275"])
    ap.add_argument("--source", choices=("global", "runmean"), default="global",
                    help="global = sigma dal fit su tutti i run insieme (default); "
                         "runmean = media delle sigma per run, con la sua banda di drift. "
                         "Con runmean il drift NON va sottratto: mediando le sigma dei "
                         "singoli run lo spostamento dei picchi non entra mai.")
    ap.add_argument("--suffix", default="", help="suffisso per i file di output")
    a = ap.parse_args()
    excl = {tuple(int(v) for v in s.split(":")) for s in a.exclude}

    data = {R: load(a.plotdir, a.besdir, R) for R in (340, 400, 500)}
    fig, axs = plt.subplots(2, 3, figsize=(19, 11), sharex=True,
                            gridspec_kw=dict(height_ratios=[2, 1]))
    rows, rowsfit, store = [], [], {}
    S_340 = None
    for j, R in enumerate((340, 400, 500)):
        d = data[R]
        Es = [e for e in sorted(d) if (R, e) not in excl and "bes" in d[e]]
        if not Es:
            continue
        x = np.array([E_TRUE.get(e, e) for e in Es])
        if a.source == "runmean":
            # mean of the per-run sigmas where it exists (>=2 runs), else the single run
            raw = np.array([100 * d[e]["sigma_runmean"] / d[e]["peak_runmean"]
                            if "sigma_runmean" in d[e] else d[e]["rel"] for e in Es])
        else:
            raw = np.array([d[e]["rel"] for e in Es])
        bes = np.array([d[e]["bes"] for e in Es])
        syn = sync_pct(x)
        estat = np.array([100 * d[e]["err_sigma"] / d[e]["peak"] for e in Es])
        if a.source == "runmean":
            # the band is the drift systematic on sigma, in ADC -> percent
            esyst = np.array([100 * d[e]["syst_sigma_abs"] / d[e]["peak_runmean"]
                              if "syst_sigma_abs" in d[e] else 0. for e in Es])
        else:
            esyst = np.array([raw[i] * d[Es[i]].get("syst_rel", 0.) / 100
                              for i in range(len(Es))])
        etot = np.hypot(estat, esyst)

        # drift = systematic on the PEAK: runs with different peaks, averaged
        # together, widen the distribution. It has to be subtracted in quadrature, it
        # is not an error. With the per-run mean drift never enters: nothing to subtract
        dr = (np.zeros(len(Es)) if a.source == "runmean"
              else np.array([d[e].get("drift_pct", 0.) for e in Es]))

        def _sub(v):
            return np.where(v > 0, np.sqrt(np.abs(v)), np.nan)
        step1 = _sub(raw ** 2 - dr ** 2)
        step2 = _sub(raw ** 2 - dr ** 2 - bes ** 2)
        step3 = _sub(raw ** 2 - dr ** 2 - bes ** 2 - syn ** 2)

        # comparison curve: the sigma of the global fit with the drift subtracted.
        # If the interpretation is right it has to coincide with the mean of the
        # per-run sigmas, which does not see the drift at all.
        glob_raw = np.array([d[e]["rel"] for e in Es])
        glob_dr = np.array([d[e].get("drift_pct", 0.) for e in Es])
        glob_nodrift = _sub(glob_raw ** 2 - glob_dr ** 2)

        # y limits computed once and reused identically in the two figures
        top_hi = float(np.nanmax(np.concatenate([raw, step3])) * 1.08)
        terms = np.concatenate([raw, bes, syn, dr[dr > 0]]) if (dr > 0).any() \
            else np.concatenate([raw, bes, syn])
        bot_lo = float(np.nanmin(terms[terms > 0]) * 0.4)
        bot_hi = float(np.nanmax(terms) * 2.5)

        ax, ax2 = axs[0][j], axs[1][j]
        ax.errorbar(x, raw, yerr=etot, fmt="o-", ms=6, color="0.35", capsize=3,
                    label=("mean of the per-run $\\sigma$ $\\pm$ drift band"
                           if a.source == "runmean" else "no correction"))
        if a.source == "runmean":
            ax.plot(x, glob_nodrift, "--", lw=1.4, color="C0", marker="s", ms=4,
                    label="global $\\sigma$ $-$ drift   (comparison)")
        ax.errorbar(x, step3, yerr=etot, fmt="^-", ms=7.5, color="C3", capsize=3,
                    label=("$-$ BES $-$ synchrotron   (fitted)" if a.source == "runmean"
                           else "$-$ drift $-$ BES $-$ synchrotron   (fitted)"))

        # mean of the per-run sigmas, with its band
        xr = [E_TRUE.get(e, e) for e in Es if "sigma_runmean" in d[e]]
        yr = np.array([100 * d[e]["sigma_runmean"] / d[e]["peak_runmean"]
                       for e in Es if "sigma_runmean" in d[e]])
        br = np.array([100 * d[e].get("syst_sigma_abs", 0.) / d[e]["peak_runmean"]
                       for e in Es if "sigma_runmean" in d[e]])
        if len(xr) and a.source != "runmean":
            # bars and not a band: the energies with >=2 runs are scattered, and a
            # fill_between would interpolate them, drawing a wedge that is not there
            ax.errorbar(xr, yr, yerr=br, fmt="v", ms=9, mfc="none", mec="C2",
                        mew=2, ecolor="C2", elinewidth=2.5, capsize=5,
                        label="mean of the per-run $\\sigma$ $\\pm$ drift band",
                        zorder=5, ls="none")

        ok = np.isfinite(step3) & (etot > 0)
        if ok.sum() >= 4:
            S_start = S_340 if (R == 500 and S_340 is not None) else 5.
            mi = Minuit(LeastSquares(x[ok], step3[ok], etot[ok], reso),
                        N=0.3, S=S_start, C=0.3)
            mi.limits["N"] = (0, None); mi.limits["S"] = (0, None); mi.limits["C"] = (0, None)
            if R == 500:
                mi.fixed["C"] = True          # C fissato a 0.3: i dati arrivano
                                              # only to 150 GeV and do not constrain it
            mi.migrad(); mi.hesse()
            if R == 340:
                S_340 = mi.values["S"]
            xx = np.linspace(x.min() * .9, x.max() * 1.05, 300)
            ax.plot(xx, reso(xx, *mi.values), "--", lw=2.2, color="darkviolet",
                    label="fit  $N/E \\oplus S/\\sqrt{E} \\oplus C$", zorder=6)
            ax.set_title(f"{R} $\\Omega$", fontsize=12, fontweight="bold")
            tab = (f"$N$ (noise)   {1000*mi.values['N']:6.0f} $\\pm$ {1000*mi.errors['N']:.0f}  MeV\n"
                   f"$S$ (stochastic) {mi.values['S']:6.2f} $\\pm$ {mi.errors['S']:.2f}  %\n"
                   + (f"$C$ (constant)   {mi.values['C']:6.3f}  % (FIXED)\n"
                      if R == 500 else
                      f"$C$ (constant)   {mi.values['C']:6.3f} $\\pm$ {mi.errors['C']:.3f}  %\n") +
                   f"$\\chi^2$/ndf        {mi.fval:6.1f} / {int(ok.sum())-3}")
            ax.text(.97, .60, tab, transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, family="monospace",
                    bbox=dict(fc="white", ec="darkviolet", lw=1.2, alpha=.95, pad=6))
            rowsfit.append((R, mi.values["N"], mi.errors["N"], mi.values["S"],
                            mi.errors["S"], mi.values["C"], mi.errors["C"],
                            mi.fval, int(ok.sum()) - (2 if R == 500 else 3)))
        ax.set_ylabel("$\\sigma/E$  [%]")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_ylim(0, top_hi)

        # ---- panel: the size of each subtracted contribution
        ax2.plot(x, raw, "o-", ms=5, color="0.35", label="$\\sigma/\\mu$")
        # zero points are not drawn: on a log scale matplotlib would draw a
        # vertical line down to the bottom of the axis
        if a.source != "runmean":
            ax2.plot(x, np.where(dr > 0, dr, np.nan), "s--", ms=5, color="C0",
                     label="drift (on the peak)")
        ax2.plot(x, bes, "D-.", ms=5, color="C1", label="BES")
        ax2.plot(x, syn, "^-", ms=5, color="C4", label="synchrotron")

        ax2.set_yscale("log")
        ax2.set_ylim(bot_lo, bot_hi)
        ax2.set_xlabel("True beam energy [GeV]")
        ax2.set_ylabel("size of each term  [%]")
        ax2.grid(alpha=.3, which="both"); ax2.legend(fontsize=8)

        store[R] = dict(x=x, raw=raw, dr=dr, bes=bes, syn=syn, final=step3,
                        etot=etot, xr=xr, yr=yr, br=br, glob_nodrift=glob_nodrift,
                        top=(0, top_hi), bot=(bot_lo, bot_hi))
        for i, e in enumerate(Es):
            rows.append((R, e, x[i], raw[i], dr[i], bes[i], syn[i], estat[i],
                         esyst[i], step1[i], step2[i], step3[i]))
    fig.tight_layout()
    fig.savefig(os.path.join(a.plotdir, f"resolution_final{a.suffix}.png"), dpi=150)

    with open(os.path.join(a.plotdir, f"resolution_final{a.suffix}.csv"), "w") as fh:
        fh.write("resistance,energy_nom,energy_true,sigma_over_mu_pct,drift_pct,"
                 "bes_pct,synchrotron_pct,err_stat_pct,err_syst_pct,"
                 "after_drift_pct,after_drift_bes_pct,final_pct\n")
        for r in rows:
            fh.write(",".join(f"{v:.5f}" if isinstance(v, float) else str(v)
                              for v in r) + "\n")
    # ------------------------------------------------- simultaneous fit, S and C common
    Rs = [R for R in (340, 400, 500) if R in store]
    if len(Rs) >= 2:
        def chi2(S, C, N340, N400, N500):
            Np = {340: N340, 400: N400, 500: N500}
            t = 0.
            for R in Rs:
                d = store[R]
                k = np.isfinite(d["final"]) & (d["etot"] > 0)
                t += (((d["final"][k] - reso(d["x"][k], Np[R], S, C)) / d["etot"][k]) ** 2).sum()
            return t
        chi2.errordef = 1
        ms = Minuit(chi2, S=2.5, C=0.38, N340=.30, N400=.31, N500=.21)
        for k in ("S", "C", "N340", "N400", "N500"):
            ms.limits[k] = (0, None)
        ms.migrad(); ms.hesse()
        npt = sum(int((np.isfinite(store[R]["final"]) & (store[R]["etot"] > 0)).sum())
                  for R in Rs)
        ndf = npt - 5
        fig2, ax2s = plt.subplots(2, len(Rs), figsize=(6.4 * len(Rs), 11), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 1]))
        for j, R in enumerate(Rs):
            d = store[R]
            ax, axb = ax2s[0][j], ax2s[1][j]
            ax.errorbar(d["x"], d["raw"], yerr=d["etot"], fmt="o-", ms=6, color="0.35",
                        capsize=3, label=("mean of the per-run $\\sigma$ $\\pm$ drift band"
                           if a.source == "runmean" else "no correction"))
            if a.source == "runmean":
                ax.plot(d["x"], d["glob_nodrift"], "--", lw=1.4, color="C0", marker="s",
                        ms=4, label="global $\\sigma$ $-$ drift   (comparison)")
            ax.errorbar(d["x"], d["final"], yerr=d["etot"], fmt="^-", ms=7.5, color="C3",
                        capsize=3, label="$-$ drift $-$ BES $-$ synchrotron   (fitted)")
            if len(d["xr"]) and a.source != "runmean":
                ax.errorbar(d["xr"], d["yr"], yerr=d["br"], fmt="v", ms=9, mfc="none",
                            mec="C2", mew=2, ecolor="C2", elinewidth=2.5, capsize=5,
                            ls="none", label="mean of the per-run $\\sigma$ $\\pm$ drift band")
            xx = np.linspace(d["x"].min() * .9, d["x"].max() * 1.05, 300)
            NR = ms.values[f"N{R}"]
            ax.plot(xx, reso(xx, NR, ms.values["S"], ms.values["C"]), "--", lw=2.2,
                    color="darkviolet", label="simultaneous fit ($S$, $C$ common)")
            ax.set_title(f"{R} $\\Omega$", fontsize=12, fontweight="bold")
            tab = (f"$N$ (noise)   {1000*NR:6.0f} $\\pm$ {1000*ms.errors[f'N{R}']:.0f}  MeV\n"
                   f"$S$ (stochastic) {ms.values['S']:6.2f} $\\pm$ {ms.errors['S']:.2f}  % (common)\n"
                   f"$C$ (constant)   {ms.values['C']:6.3f} $\\pm$ {ms.errors['C']:.3f}  % (common)\n"
                   f"$\\chi^2$/ndf (global) {ms.fval:6.1f} / {ndf}")
            ax.text(.97, .60, tab, transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, family="monospace",
                    bbox=dict(fc="white", ec="darkviolet", lw=1.2, alpha=.95, pad=6))
            ax.set_ylabel("$\\sigma/E$  [%]"); ax.grid(alpha=.3)
            ax.legend(fontsize=8); ax.set_ylim(*d["top"])
            axb.plot(d["x"], d["raw"], "o-", ms=5, color="0.35", label="$\\sigma/\\mu$")
            if a.source != "runmean":
                axb.plot(d["x"], np.where(d["dr"] > 0, d["dr"], np.nan), "s--", ms=5,
                         color="C0", label="drift (on the peak)")
            axb.plot(d["x"], d["bes"], "D-.", ms=5, color="C1", label="BES")
            axb.plot(d["x"], d["syn"], "^-", ms=5, color="C4", label="synchrotron")
            axb.set_yscale("log"); axb.set_ylim(*d["bot"])
            axb.grid(alpha=.3, which="both")
            axb.set_xlabel("True beam energy [GeV]")
            axb.set_ylabel("size of each term  [%]"); axb.legend(fontsize=8)
        fig2.tight_layout()
        fig2.savefig(os.path.join(a.plotdir, f"resolution_final_common{a.suffix}.png"), dpi=150)
        plt.close(fig2)
        print()
        print(f"FIT SIMULTANEO (S e C comuni): chi2/ndf = {ms.fval:.1f}/{ndf}")
        print(f"  S = {ms.values['S']:.2f} +- {ms.errors['S']:.2f} %   "
              f"C = {ms.values['C']:.3f} +- {ms.errors['C']:.3f} %")
        for R in Rs:
            print(f"  {R} ohm: N = {1000*ms.values[f'N{R}']:.0f} +- "
                  f"{1000*ms.errors[f'N{R}']:.0f} MeV")
        with open(os.path.join(a.plotdir, f"resolution_fit_common{a.suffix}.csv"), "w") as fh:
            fh.write("parameter,value,error,note\n")
            fh.write(f"S_pct,{ms.values['S']:.4f},{ms.errors['S']:.4f},common\n")
            fh.write(f"C_pct,{ms.values['C']:.5f},{ms.errors['C']:.5f},common\n")
            for R in Rs:
                fh.write(f"N_GeV_{R},{ms.values[f'N{R}']:.4f},"
                         f"{ms.errors[f'N{R}']:.4f},per resistance\n")
            fh.write(f"chi2,{ms.fval:.3f},0,global\nndf,{ndf},0,global\n")

    with open(os.path.join(a.plotdir, f"resolution_fit_terms{a.suffix}.csv"), "w") as fh:
        fh.write("resistance,N_GeV,err_N,S_pct,err_S,C_pct,err_C,chi2,ndf\n")
        for r in rowsfit:
            fh.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},"
                     f"{r[5]:.5f},{r[6]:.5f},{r[7]:.3f},{r[8]}\n")
    print()
    print(f'{"R":>5} {"N (GeV)":>16} {"S (%)":>16} {"C (%)":>16} {"chi2/ndf":>11}')
    for r in rowsfit:
        print(f"{r[0]:>5} {r[1]:8.2f} +- {r[2]:4.2f} {r[3]:8.2f} +- {r[4]:4.2f} "
              f"{r[5]:8.3f} +- {r[6]:5.3f} {r[7]:7.1f}/{r[8]}")
    print()
    print("scritto resolution_final.png, resolution_final.csv, resolution_fit_terms.csv")
    for R in (340, 400, 500):
        s = [r for r in rows if r[0] == R]
        if s:
            print(f"{R} ohm: {len(s)} punti | medie: drift {np.mean([r[4] for r in s]):.3f}%, "
                  f"BES {np.mean([r[5] for r in s]):.3f}%, sincr {np.mean([r[6] for r in s]):.3f}%"
                  f"  -> sottrazione totale {np.nanmean([r[3]-r[11] for r in s]):.4f} punti %")


if __name__ == "__main__":
    main()
