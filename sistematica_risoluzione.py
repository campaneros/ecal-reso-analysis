#!/usr/bin/env python3
"""
Sistematica di drift da sommare alla risoluzione energetica
===========================================================
Legge i CSV di drift_dcb_all.py e produce, energia per energia, l'errore
sistematico relativo da sommare in quadratura a sigma/E.

Metodo (lo stesso della slide "Systematic on centroid selection", applicato ai
run invece che ai bin del centroide): i valori di sigma dei run alla stessa
energia vengono fittati con una costante; l'errore relativo da aggiungere
perche' chi2/ndf = 1 e' la sistematica. Se i run sono gia' compatibili -> 0.

Le energie con un solo run non permettono la misura: prendono un valore di
ripiego (default: la mediana di quelle misurate, esclusi i punti patologici).

Uso: python3 sistematica_risoluzione.py --plotdir . [--fallback 4.0] [--escludi 340:275]
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {340: "C0", 400: "C1", 500: "C2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plotdir", default=".")
    ap.add_argument("--fallback", type=float, default=None,
                    help="syst relativa %% per le energie con 1 solo run (default: mediana misurata)")
    ap.add_argument("--soglia-patologica", type=float, default=10.0,
                    help="syst relativa oltre la quale il punto e' considerato patologico "
                         "(due popolazioni di run, non una sistematica gaussiana)")
    a = ap.parse_args()

    data = {}
    for R in (340, 400, 500):
        f1 = os.path.join(a.plotdir, str(R), f"drift_per_run_{R}ohm.csv")
        f2 = os.path.join(a.plotdir, str(R), f"sistematica_drift_{R}ohm.csv")
        if not (os.path.exists(f1) and os.path.exists(f2)):
            continue
        syst = {int(r["energy"]): r for r in csv.DictReader(open(f2))}
        rows = []
        for r in csv.DictReader(open(f1)):
            if r["run"] != "ALL":
                continue
            e = int(r["energy"])
            p, s, es = float(r["peak_abs"]), float(r["sigma_abs"]), float(r["err_sigma_abs"])
            d = syst.get(e)
            rows.append(dict(E=e, n=int(d["n_run"]) if d else 1,
                             rel=100 * s / p, stat=100 * es / p,
                             sy=float(d["syst_sigma_pct"]) if d else np.nan,
                             sy_peak=float(d["syst_picco_pct"]) if d else np.nan))
        data[R] = sorted(rows, key=lambda x: x["E"])

    misurate = [r["sy"] for R in data for r in data[R]
                if np.isfinite(r["sy"]) and r["sy"] < a.soglia_patologica]
    fb = a.fallback if a.fallback is not None else float(np.median(misurate))
    print(f"syst misurate: N={len(misurate)}  mediana={np.median(misurate):.2f}%  "
          f"max={max(misurate):.2f}%   -> ripiego usato = {fb:.2f}%")

    for R in data:
        out = os.path.join(a.plotdir, str(R), f"sistematica_risoluzione_{R}ohm.csv")
        with open(out, "w") as fh:
            fh.write("energy,n_run,sigma_su_mu_pct,err_stat_pct,syst_drift_rel_pct,"
                     "origine,syst_applicata_rel_pct,err_syst_pct,err_tot_pct\n")
            for r in data[R]:
                if np.isfinite(r["sy"]) and r["sy"] < a.soglia_patologica:
                    sy, org = r["sy"], "misurata"
                elif np.isfinite(r["sy"]):
                    sy, org = r["sy"], "PATOLOGICA"
                else:
                    sy, org = fb, "ripiego"
                r["sy_app"], r["org"] = sy, org
                esy = r["rel"] * sy / 100
                r["esy"] = esy
                r["etot"] = np.hypot(r["stat"], esy)
                m = f"{r['sy']:.2f}" if np.isfinite(r["sy"]) else ""
                fh.write(f"{r['E']},{r['n']},{r['rel']:.4f},{r['stat']:.4f},{m},"
                         f"{org},{sy:.2f},{esy:.4f},{r['etot']:.4f}\n")
        print("scritto", out)

    fig, axs = plt.subplots(1, len(data), figsize=(6 * len(data), 5.5), squeeze=False)
    for ax, R in zip(axs[0], sorted(data)):
        rows = [r for r in data[R] if r["org"] != "PATOLOGICA"]
        bad = [r for r in data[R] if r["org"] == "PATOLOGICA"]
        E = [r["E"] for r in rows]
        y = [r["rel"] for r in rows]
        ax.errorbar(E, y, yerr=[r["etot"] for r in rows], fmt="none",
                    ecolor=COL[R], elinewidth=6, alpha=.35, capsize=0,
                    label="stat $\\oplus$ drift syst")
        ax.errorbar(E, y, yerr=[r["stat"] for r in rows], fmt="o", ms=5,
                    color=COL[R], capsize=3, label="stat only")
        for r in rows:
            if r["org"] == "ripiego":
                ax.plot(r["E"], r["rel"], "o", mfc="none", ms=11, color="grey")
        for r in bad:
            ax.plot(r["E"], r["rel"], "x", ms=11, mew=2.5, color="red",
                    label="pathological drift, to be redone")
        ax.set_xlabel("Beam energy [GeV]")
        ax.set_ylabel("$\\sigma/\\mu$ (3x3, uncalibrated) [%]")
        ax.set_title(f"{R} $\\Omega$", fontsize=12, fontweight="bold")
        ax.grid(alpha=.3)
        h, l = ax.get_legend_handles_labels()
        seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=8)
    fig.suptitle("$\\sigma/E$  with  $\\Delta_{syst}$  such that  $\\chi^2/\\mathrm{ndf} = 1$",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(a.plotdir, "sistematica_risoluzione.png"), dpi=150)
    print("scritto sistematica_risoluzione.png")


if __name__ == "__main__":
    main()
