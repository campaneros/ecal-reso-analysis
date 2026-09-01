#!/usr/bin/env python3
"""
Final resolution curve with the response non-uniformity corrected.

Separate copy of resolution_final.py: that one is NOT touched. The difference is the
central value and one extra systematic.

Central value: sigma/E after the EVENT-BY-EVENT CORRECTION of the response in
(pos_eta, pos_phi), already net of BES and synchrotron:

    (sigma/E)^2 = (sigma_corr/mu_corr)^2 - BES^2 - SYNC^2

where sigma_corr comes from the per-run DCB fit of A -> A * <f>_run / f(u_i, v_i),
with f the quadratic response surface, and the average over runs is weighted by the
number of events.

Drift is NOT subtracted: starting from the mean of the per-run sigmas it never
enters.

Error bars, in quadrature:
  stat    weighted variance of the per-run sigmas, which already contains both the
          noise of the individual fits and the run-to-run spread. Where a point has
          a single run it is undefined and the fit error is used instead.
  drift   run-to-run systematic on sigma, syst_sigma_ADC / peak_medio from
          plot/<R>/sistematica_drift_<R>ohm.csv, scaled until chi2/ndf = 1
  map     centroid systematic: the spread of sigma/mu between the per-run and the
          per-energy response map (see uniformita_maps.py). It is an uncertainty on
          HOW the surface is estimated, so it belongs in the error bars, unlike the
          size of the correction itself.

Usage:
  python3 plot/resolution_final_uniforme.py --plotdir plot \
      --unif plot/uniformita/uniformita_pos.csv --suffix _unif [--exclude 340:275]
"""
import argparse, csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

COL = {340: "C0", 400: "C1", 500: "C2"}
CUT = ("$A_{tot} > 100$ ADC,  $|\\mathrm{pos\\_eta}-18| \\leq 0.2$,  "
       "$|\\mathrm{pos\\_phi}-6| \\leq 0.2$")


def reso(x, N, S, C):
    return np.sqrt((100 * N / x) ** 2 + (S / np.sqrt(x)) ** 2 + C ** 2)


def load_drift(plotdir, R):
    """syst_sigma relativa in punti percentuali di sigma/mu."""
    out = {}
    f = os.path.join(plotdir, str(R), f"sistematica_drift_{R}ohm.csv")
    if not os.path.exists(f):
        return out
    for r in csv.DictReader(open(f)):
        try:
            p = float(r["peak_medio"]); s = float(r["syst_sigma_ADC"])
        except (ValueError, KeyError):
            continue
        if p > 0 and np.isfinite(s):
            out[int(r["energy"])] = 100 * s / p
    return out


def wscatter(vals, wts):
    """Error on the weighted mean from the weighted variance of the values: it already
    contains both the noise of the individual fits and the run-to-run spread.
    n_eff = (sum w)^2 / sum w^2; with a single run it is undefined."""
    v = np.asarray(vals, float); w = np.asarray(wts, float)
    g = np.isfinite(v) & (w > 0)
    v, w = v[g], w[g]
    if len(v) < 2:
        return np.nan
    V1, V2 = w.sum(), (w ** 2).sum()
    neff = V1 ** 2 / V2
    if neff <= 1:
        return np.nan
    mu = (w * v).sum() / V1
    return float(np.sqrt((w * (v - mu) ** 2).sum() / V1 / (neff - 1)))


def load_scatter(cachedir, key="corr"):
    """Scatter per (R, E) from the per-run values already in plot/uniformita/_cache."""
    out = {}
    for f in glob.glob(os.path.join(cachedir, "*.json")):
        c = json.load(open(f))
        ok = [p for p in c["per_run"] if key in p]
        if ok:
            out[(int(c["resistance"]), int(c["energy"]))] = \
                wscatter([p[key][0] for p in ok], [p["n"] for p in ok])
    return out


def load_unif(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        R, E = int(r["resistance"]), int(r["energy"])
        rows[(R, E)] = {k: (float(v) if v not in ("", "nan") else np.nan)
                        for k, v in r.items()}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plotdir", default="plot")
    ap.add_argument("--unif", default="plot/uniformita/uniformita_pos.csv")
    ap.add_argument("--exclude", nargs="*", default=["340:275"])
    ap.add_argument("--central", choices=("run", "corr", "raw"), default="raw",
                    help="run  = sigma corretta con la parabola del singolo run "
                         "(nominale); corr = corretta con la parabola dell'energia; "
                         "raw = non corretta")
    ap.add_argument("--maps", default="plot/uniformita_maps/uniformita_maps.csv",
                    help="CSV di uniformita_maps.py, per la sistematica sulla mappa")
    ap.add_argument("--syst", choices=("none", "map", "raw_corr", "corr_pos"), default="map",
                    help="map = spread di sigma/mu fra mappa per run, per energia e "
                         "media: e' l'incertezza su COME si stima la superficie, e va "
                         "nelle barre; none = solo stat e drift; "
                         "raw_corr = |raw - corr|, la dimensione dell'effetto che si "
                         "lascia dentro quando NON si corregge; "
                         "corr_pos = |corr - pos|, da usare solo per studio: sommarlo "
                         "a una sigma gia' corretta conta due volte lo stesso effetto")
    ap.add_argument("--cache", default="plot/uniformita/_cache")
    ap.add_argument("--suffix", default="_unif")
    a = ap.parse_args()
    excl = {tuple(int(v) for v in s.split(":")) for s in a.exclude}

    U = load_unif(a.unif)
    SCAT = load_scatter(a.cache, "corr" if a.central != "raw" else "raw")
    MAPS = {}
    if os.path.exists(a.maps):
        for r in csv.DictReader(open(a.maps)):
            MAPS[(int(r["resistance"]), int(r["energy"]))] = {
                k: (float(r[k]) if r[k] not in ("", "nan") else np.nan)
                for k in ("s_run", "err_run", "scat_run", "s_energy", "syst_pct")}
    drift = {R: load_drift(a.plotdir, R) for R in (340, 400, 500)}

    fig, axs = plt.subplots(2, 3, figsize=(19, 11), sharex=True,
                            gridspec_kw=dict(height_ratios=[2, 1]))
    rows, rowsfit, store = [], [], {}
    S_340 = None
    for j, R in enumerate((340, 400, 500)):
        Es = sorted(e for (r, e) in U if r == R and (R, e) not in excl)
        # without BES the point cannot be subtracted: the same criterion as
        # resolution_final.py, which requires the row in rereco_<R>_withBES.csv
        nobes = [e for e in Es if not (U[(R, e)]["bes"] > 0)]
        if nobes:
            print(f"  {R} ohm: escludo {nobes} GeV, BES assente in rereco_{R}_withBES.csv")
        Es = [e for e in Es if e not in nobes]
        if not Es:
            continue
        u = [U[(R, e)] for e in Es]
        x = np.array([q["energy_true"] for q in u])
        raw = np.array([q["raw"] for q in u])            # sigma/mu senza trattamento
        cen = np.array([q["corr_corr"] for q in u])      # centrale, gia' - BES - sync
        rawc = np.array([q["raw_corr"] for q in u])      # raw, gia' - BES - sync
        posc = np.array([q["pos_corr"] for q in u])
        flatc = np.array([q["flat_corr"] for q in u])
        bes = np.array([q["bes"] for q in u])
        syn = np.array([q["sync"] for q in u])
        pos_term = np.array([q["pos_term"] for q in u])          # POS naive, std(f)/mean(f)
        # POS_eff: how much the event-by-event correction ACTUALLY removes from sigma.
        # By construction  sqrt(raw^2 - bes^2 - syn^2 - POS_eff^2) = sqrt(corr^2 - bes^2 - syn^2),
        # i.e. the central value is the corrected sigma, but the term is explicit and
        # subtractable like BES and synchrotron. It does not assume the term is Gaussian:
        # the naive POS does assume it, and indeed overestimates the term.
        cor = np.array([q["corr"] for q in u])
        pos_eff = np.sqrt(np.maximum(raw ** 2 - cor ** 2, 0.))

        if a.central == "raw":
            cen = rawc
            efit = np.array([q["raw_err"] for q in u])
            esc = np.array([SCAT.get((R, e), np.nan) for e in Es])
        elif a.central == "run":
            # nominal: weighted mean of the per-run sigmas, each corrected with the
            # parabola of its own run. BES and synchrotron are subtracted here.
            srun = np.array([MAPS.get((R, e), {}).get("s_run", np.nan) for e in Es])
            cen = np.sqrt(np.maximum(srun ** 2 - bes ** 2 - syn ** 2, 0.))
            efit = np.array([MAPS.get((R, e), {}).get("err_run", np.nan) for e in Es])
            esc = np.array([MAPS.get((R, e), {}).get("scat_run", np.nan) for e in Es])
            pos_eff = np.sqrt(np.maximum(raw ** 2 - srun ** 2, 0.))
        else:
            efit = np.array([q["corr_err"] for q in u])
            esc = np.array([SCAT.get((R, e), np.nan) for e in Es])
        # statistical error: weighted variance of the per-run sigmas when there is
        # more than one run (it already contains the fit error), else the fit error
        estat = np.where(np.isfinite(esc), np.maximum(efit, esc), efit)
        edrift = np.array([drift[R].get(e, 0.) for e in Es])
        if a.syst == "map":
            # spread between the maps, on sigma/mu; propagated through the
            # subtractions: sigma_fin^2 = sigma^2 - const  =>  d_fin = sigma*d/sigma_fin
            dsig = np.array([MAPS.get((R, e), {}).get("syst_pct", 0.) for e in Es])
            eunif = np.where(cen > 0, raw * dsig / np.maximum(cen, 1e-9), 0.)
        elif a.syst == "raw_corr":
            eunif = np.abs(rawc - np.array([q["corr_corr"] for q in u]))
        elif a.syst == "corr_pos":
            eunif = np.abs(np.array([q["corr_corr"] for q in u]) - posc)
        else:
            eunif = np.zeros(len(Es))
        etot = np.sqrt(estat ** 2 + edrift ** 2 + eunif ** 2)

        ax, ax2 = axs[0][j], axs[1][j]
        CENLAB = ("$-$ BES $-$ synchrotron" if a.central == "raw"
                  else "$-$ BES $-$ synchrotron $-$ POS$_{eff}$")
        good = np.isfinite(cen) & (cen > 0)
        cen = np.where(good, cen, np.nan)
        ax.plot(x, raw, "o-", ms=6, color="0.35", label="$\\sigma/\\mu$")
        ax.errorbar(x, cen, yerr=etot, fmt="^-", ms=7.5, color="C3", capsize=3,
                    label=CENLAB)

        ok = np.isfinite(cen) & (etot > 0)
        if ok.sum() >= 4:
            S_start = S_340 if (R == 500 and S_340 is not None) else 5.
            mi = Minuit(LeastSquares(x[ok], cen[ok], etot[ok], reso),
                        N=0.3, S=S_start, C=0.3)
            mi.limits["N"] = (0, None); mi.limits["S"] = (0, None); mi.limits["C"] = (0, None)
            if R == 500:
                mi.fixed["C"] = True
            mi.migrad(); mi.hesse()
            if R == 340:
                S_340 = mi.values["S"]
            xx = np.linspace(x.min() * .9, x.max() * 1.05, 300)
            ax.plot(xx, reso(xx, *mi.values), "--", lw=2.2, color="darkviolet",
                    label="fit  $N/E \\oplus S/\\sqrt{E} \\oplus C$", zorder=6)
            ndf = int(ok.sum()) - (2 if R == 500 else 3)
            tab = (f"$N$ (noise)   {1000*mi.values['N']:6.0f} $\\pm$ {1000*mi.errors['N']:.0f}  MeV\n"
                   f"$S$ (stochastic) {mi.values['S']:6.2f} $\\pm$ {mi.errors['S']:.2f}  %\n"
                   + (f"$C$ (constant)   {mi.values['C']:6.3f}  % (FIXED)\n" if R == 500 else
                      f"$C$ (constant)   {mi.values['C']:6.3f} $\\pm$ {mi.errors['C']:.3f}  %\n")
                   + f"$\\chi^2$/ndf        {mi.fval:6.1f} / {ndf}")
            ax.text(.97, .60, tab, transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, family="monospace",
                    bbox=dict(fc="white", ec="darkviolet", lw=1.2, alpha=.95, pad=6))
            rowsfit.append((R, mi.values["N"], mi.errors["N"], mi.values["S"],
                            mi.errors["S"], mi.values["C"], mi.errors["C"],
                            mi.fval, ndf))
        ax.set_title(f"{R} $\\Omega$", fontsize=12, fontweight="bold")
        ax.set_ylabel("$\\sigma/E$  [%]")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_ylim(0, float(np.nanmax(raw) * 1.08))

        ax2.plot(x, raw, "o-", ms=5, color="0.35", label="$\\sigma/\\mu$")
        ax2.plot(x, bes, "D-.", ms=5, color="C1", label="BES")
        ax2.plot(x, syn, "^-", ms=5, color="C4", label="synchrotron")
        # zero points must be set to nan: on a log scale matplotlib would draw a
        # vertical line down to the bottom of the axis
        if a.central != "raw":
            ax2.plot(x, np.where(pos_eff > 0, pos_eff, np.nan), "v-", ms=5, color="C2",
                     label="POS$_{eff}$")
        terms = np.concatenate([raw, bes, syn, pos_eff[pos_eff > 0], np.array([1e-3])])
        ax2.set_yscale("log")
        ax2.set_ylim(float(np.nanmin(terms[terms > 0]) * 0.4), float(np.nanmax(terms) * 2.5))
        ax2.set_xlabel("True beam energy [GeV]")
        ax2.set_ylabel("size of each term  [%]")
        ax2.grid(alpha=.3, which="both"); ax2.legend(fontsize=8)

        store[R] = dict(x=x, raw=raw, final=cen, etot=etot, bes=bes, syn=syn,
                        pos=pos_term, pos_eff=pos_eff, eunif=eunif, edrift=edrift, rawc=rawc,
                        top=(0, float(np.nanmax(raw) * 1.08)))
        for i, e in enumerate(Es):
            rows.append((R, e, x[i], raw[i], bes[i], syn[i], pos_eff[i], pos_term[i],
                         estat[i], edrift[i], eunif[i], etot[i], rawc[i], posc[i],
                         flatc[i], cen[i]))

    fig.suptitle(CUT, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.plotdir, f"resolution_final{a.suffix}.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(a.plotdir, f"resolution_final{a.suffix}.csv"), "w") as fh:
        fh.write("resistance,energy_nom,energy_true,sigma_over_mu_pct,bes_pct,"
                 "synchrotron_pct,pos_eff_pct,pos_naive_pct,err_stat_pct,err_drift_pct,"
                 "err_unif_pct,err_tot_pct,raw_corr_pct,pos_corr_pct,flat_corr_pct,"
                 "final_pct\n")
        for r in rows:
            fh.write(",".join(f"{v:.5f}" if isinstance(v, float) else str(v)
                              for v in r) + "\n")

    # ---------------------------------------------- simultaneous fit, S and C common
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
        ms = Minuit(chi2, S=2.5, C=0.35, N340=.30, N400=.30, N500=.21)
        for k in ("S", "C", "N340", "N400", "N500"):
            ms.limits[k] = (0, None)
        ms.migrad(); ms.hesse()
        npt = sum(int((np.isfinite(store[R]["final"]) & (store[R]["etot"] > 0)).sum())
                  for R in Rs)
        ndf = npt - 5
        fig2, ax2s = plt.subplots(1, len(Rs), figsize=(6.4 * len(Rs), 6), sharey=False)
        for j, R in enumerate(Rs):
            d = store[R]; ax = ax2s[j]
            ax.plot(d["x"], d["raw"], "o-", ms=5, color="0.35", label="$\\sigma/\\mu$")
            ax.errorbar(d["x"], d["final"], yerr=d["etot"], fmt="^-", ms=7, color="C3",
                        capsize=3, label="$-$ BES $-$ synchrotron $-$ POS$_{eff}$")
            xx = np.linspace(d["x"].min() * .9, d["x"].max() * 1.05, 300)
            NR = ms.values[f"N{R}"]
            ax.plot(xx, reso(xx, NR, ms.values["S"], ms.values["C"]), "--", lw=2.2,
                    color="darkviolet", label="simultaneous fit ($S$, $C$ common)")
            ax.set_title(f"{R} $\\Omega$", fontsize=12, fontweight="bold")
            tab = (f"$N$   {1000*NR:6.0f} $\\pm$ {1000*ms.errors[f'N{R}']:.0f}  MeV\n"
                   f"$S$   {ms.values['S']:6.2f} $\\pm$ {ms.errors['S']:.2f}  % (common)\n"
                   f"$C$   {ms.values['C']:6.3f} $\\pm$ {ms.errors['C']:.3f}  % (common)\n"
                   f"$\\chi^2$/ndf {ms.fval:6.1f} / {ndf}")
            ax.text(.97, .95, tab, transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, family="monospace",
                    bbox=dict(fc="white", ec="darkviolet", lw=1.2, alpha=.95, pad=6))
            ax.set_xlabel("True beam energy [GeV]"); ax.set_ylabel("$\\sigma/E$  [%]")
            ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_ylim(*d["top"])
        fig2.suptitle(CUT, fontsize=11)
        fig2.tight_layout()
        fig2.savefig(os.path.join(a.plotdir, f"resolution_final_common{a.suffix}.png"), dpi=150)
        plt.close(fig2)
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
                fh.write(f"N_GeV_{R},{ms.values[f'N{R}']:.4f},{ms.errors[f'N{R}']:.4f},"
                         f"per resistance\n")
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
    for R in (340, 400, 500):
        s = [r for r in rows if r[0] == R]
        if s:
            print(f"{R} ohm: {len(s)} punti | BES {np.mean([r[4] for r in s]):.3f}%, "
                  f"sincr {np.mean([r[5] for r in s]):.3f}%, "
                  f"POS_eff {np.mean([r[6] for r in s]):.4f}% (max {max(r[6] for r in s):.4f}), "
                  f"POS naive {np.mean([r[7] for r in s]):.4f}%")


if __name__ == "__main__":
    main()
