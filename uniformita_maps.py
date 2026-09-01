"""
Systematic on the non-uniformity correction: per-RUN map against an AVERAGE map.

Rather than measuring the systematic as "corrected against uncorrected" -- which is
the size of the effect, not an uncertainty -- it is measured as the sensitivity to
HOW the response surface is estimated. Three estimates, always the same quadratic
form

    f(u,v) = a0 + a1 u + a2 v + a3 u^2 + a4 v^2 + a5 uv,   u = pos_eta-18, v = pos_phi-6

  run      one surface per run, from the events of that run only
  energy   one surface per (resistance, energy), from all the runs of that point
           normalised to their own peak
  mean     a single surface per resistance, from all the runs of all energies

The nominal point uses the per-RUN map; the centroid systematic is
|sigma(per-run map) - sigma(per-energy map)|. The 'mean' variant is kept as a
diagnostic: it shows how much it costs to ignore that the curvature changes with
energy.

How the fit is done. There is no binning: the weighted linear fit on a fine-grained
binned map coincides with the regression on individual events, and the sufficient
statistics are the moment matrix M = sum_i x_i x_i' and b = sum_i x_i a_i, with
x_i = [1, u, v, u^2, v^2, uv] and a_i = A_i / peak(run). That is 6x6 + 6 numbers per
run: they are summed over runs for the per-energy map, and over all energies for the
average one, without re-reading the events. The three estimates are then exactly the
same estimator on different sets, and differ only by what one wants to measure.

Two passes, because the average map needs all energies before it can be applied:

  --stage moments   reads the files, DCB fit per run, computes and stores M, b, n
  --stage apply     builds the three surfaces, re-applies the correction and redoes
                    the DCB fits, once per variant
  --stage collect   assembles CSV and plots

Usage:
  python3 plot/uniformita_maps.py --stage moments --resistances 340 --energies 20 30 40
  python3 plot/uniformita_maps.py --stage apply   --resistances 340 --energies 20 30 40
  python3 plot/uniformita_maps.py --stage collect
"""

import argparse, os, sys, glob, re, json, math
import runsets
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uniformita_pos import (ETA0, PHI0, SEL, A_TOT_MIN, FILES, VERA,
                            fit_dcb, rel, wmean, wscatter, design)

VARIANTS = ("run", "energy", "mean")
VCOL = {"run": "C0", "energy": "C3", "mean": "C2"}
VMK = {"run": "s", "energy": "^", "mean": "v"}
NMIN_RUN = 200          # eventi minimi perche' un run abbia la sua superficie


def moments(u, v, a):
    X = design(u, v)
    return X.T @ X, X.T @ a, int(len(a))


def solve(M, b):
    try:
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return None


# ------------------------------------------------------------------ momenti
def stage_moments(path, E, R, drop, outdir, only=()):
    t = uproot.open(path)["h4_reco"]
    arr = t.arrays(["run", "A_tot", "pos_eta", "pos_phi"], library="np")
    k = ((np.abs(arr["pos_eta"] - ETA0) <= SEL) & (np.abs(arr["pos_phi"] - PHI0) <= SEL)
         & (arr["A_tot"] > A_TOT_MIN))
    if drop:
        k &= ~np.isin(arr["run"], drop)
    if len(only):
        k &= np.isin(arr["run"], only)
    run = arr["run"][k]
    at = arr["A_tot"][k].astype(float)
    u = arr["pos_eta"][k].astype(float) - ETA0
    v = arr["pos_phi"][k].astype(float) - PHI0
    if len(at) < 500:
        return None

    out = []
    for r in sorted(int(x) for x in np.unique(run)):
        m = run == r
        f = fit_dcb(at[m], E, R)
        if f is None:
            continue
        w = (at[m] >= f["lo"]) & (at[m] <= f["hi"])
        if w.sum() < 100:
            continue
        M, b, n = moments(u[m][w], v[m][w], at[m][w] / f["peak"])
        sm, se = rel(f)
        out.append(dict(run=r, n=int(m.sum()), n_win=n, peak=f["peak"],
                        sigma=f["sigma"], lo=f["lo"], hi=f["hi"],
                        raw=[sm, se], chi2ndf=f["chi2"] / f["ndf"],
                        M=[float(x) for x in M.ravel()], b=[float(x) for x in b]))
    if not out:
        return None
    res = dict(resistance=R, energy=E, per_run=out)
    json.dump(res, open(os.path.join(outdir, f"{R}_{E}.json"), "w"), default=float)
    return res


# ----------------------------------------------------------------- applica
def load_moments(momdir, R, E=None):
    pat = f"{R}_*.json" if E is None else f"{R}_{E}.json"
    out = []
    for f in sorted(glob.glob(os.path.join(momdir, pat))):
        out.append(json.load(open(f)))
    return out


def sum_moments(cachelist):
    M = np.zeros((6, 6)); b = np.zeros(6); n = 0
    for c in cachelist:
        for p in c["per_run"]:
            M += np.array(p["M"]).reshape(6, 6)
            b += np.array(p["b"])
            n += p["n_win"]
    return M, b, n


def stage_apply(path, E, R, drop, momdir, outdir, only=()):
    here = load_moments(momdir, R, E)
    if not here:
        print("      niente momenti, salto"); return None
    allR = load_moments(momdir, R)
    coef_mean = solve(*sum_moments(allR)[:2])
    coef_ene = solve(*sum_moments(here)[:2])
    if coef_mean is None or coef_ene is None:
        print("      superficie singolare, salto"); return None

    per = {p["run"]: p for p in here[0]["per_run"]}
    t = uproot.open(path)["h4_reco"]
    arr = t.arrays(["run", "A_tot", "pos_eta", "pos_phi"], library="np")
    k = ((np.abs(arr["pos_eta"] - ETA0) <= SEL) & (np.abs(arr["pos_phi"] - PHI0) <= SEL)
         & (arr["A_tot"] > A_TOT_MIN))
    if drop:
        k &= ~np.isin(arr["run"], drop)
    if len(only):
        k &= np.isin(arr["run"], only)
    run = arr["run"][k]
    at = arr["A_tot"][k].astype(float)
    u = arr["pos_eta"][k].astype(float) - ETA0
    v = arr["pos_phi"][k].astype(float) - PHI0

    rows, nfall = [], 0
    for r, p in per.items():
        m = run == r
        if m.sum() == 0:
            continue
        row = dict(run=r, n=p["n"], n_win=p["n_win"], raw=p["raw"], peak=p["peak"])
        w = (at[m] >= p["lo"]) & (at[m] <= p["hi"])
        X = design(u[m], v[m])
        for var in VARIANTS:
            if var == "run":
                c = (solve(np.array(p["M"]).reshape(6, 6), np.array(p["b"]))
                     if p["n_win"] >= NMIN_RUN else None)
                if c is None:
                    c = coef_ene; nfall += 1; row["fallback"] = True
            elif var == "energy":
                c = coef_ene
            else:
                c = coef_mean
            fev = X @ c
            fm = float(fev[w].mean())
            if not (fm > 0):
                continue
            fc = fit_dcb(at[m] * fm / np.maximum(fev, 1e-9), E, R)
            if fc is not None:
                row[var] = list(rel(fc))
                row["chi2ndf_" + var] = fc["chi2"] / fc["ndf"]
        rows.append(row)

    ok = [r for r in rows if "raw" in r]
    wts = [r["n"] for r in ok]
    out = dict(resistance=R, energy=E, nrun=len(ok), nev=int(sum(wts)),
               n_fallback=nfall, per_run=rows,
               coef_energy=[float(x) for x in coef_ene],
               coef_mean=[float(x) for x in coef_mean])
    out["raw"] = wmean([r["raw"][0] for r in ok], [r["raw"][1] for r in ok], wts)
    # key s_<var>: "energy" alone would already be the beam energy
    for var in VARIANTS:
        s = [r for r in ok if var in r]
        out["s_" + var] = (wmean([r[var][0] for r in s], [r[var][1] for r in s],
                                 [r["n"] for r in s]) if s else (np.nan, np.nan))
    json.dump(out, open(os.path.join(outdir, f"{R}_{E}.json"), "w"), default=float)
    return out


# ----------------------------------------------------------------- raccolta
# in the CSV the variants are named s_<v>: "energy" alone would be the beam energy
COLS = ("resistance,energy,energy_true,nrun,nev,n_fallback,raw,err_raw,scat_run,"
        + ",".join(f"s_{v},err_{v}" for v in VARIANTS)
        + ",syst_pct,syst_rel_pct,"
        + ",".join(f"curv_eta_{v},curv_phi_{v}" for v in ("energy", "mean")))


def stage_collect(outdir, resistances):
    rows = []
    for f in sorted(glob.glob(os.path.join(outdir, "_apply", "*.json"))):
        c = json.load(open(f))
        R, E = int(c["resistance"]), int(c["energy"])
        if R not in resistances:
            continue
        # Nominal: correction with the parabola of the SINGLE RUN.
        # Centroid systematic: against the parabola of the ENERGY, i.e. the one
        # obtained by pooling the runs of that point. On points with a single run
        # the two coincide by construction and the systematic is zero: there is no
        # ambiguity to measure there. The per-resistance 'mean' variant stays in the
        # CSV as a diagnostic and does not enter the number.
        sr, se, sm = c["s_run"][0], c["s_energy"][0], c["s_mean"][0]
        syst = abs(se - sm) if (np.isfinite(sm) and np.isfinite(se)) else np.nan # passing to the diff w.r.t. mean (Ruben)
        ce, cm = np.array(c["coef_energy"]), np.array(c["coef_mean"])
        r = dict(resistance=R, energy=E, energy_true=VERA.get(E, float(E)),
                 nrun=c["nrun"], nev=c["nev"], n_fallback=c["n_fallback"],
                 raw=c["raw"][0], err_raw=c["raw"][1],
                 scat_run=wscatter([p["run"][0] for p in c["per_run"] if "run" in p],
                                   [p["n"] for p in c["per_run"] if "run" in p]),
                 syst_pct=syst, syst_rel_pct=100 * syst / c["raw"][0],
                 curv_eta_energy=100 * ce[3] / ce[0], curv_phi_energy=100 * ce[4] / ce[0],
                 curv_eta_mean=100 * cm[3] / cm[0], curv_phi_mean=100 * cm[4] / cm[0])
        for v in VARIANTS:
            r["s_" + v] = c["s_" + v][0]; r["err_" + v] = c["s_" + v][1]
        rows.append(r)
    rows.sort(key=lambda r: (r["resistance"], r["energy"]))
    csv = os.path.join(outdir, "uniformita_maps.csv")
    with open(csv, "w") as fh:
        fh.write(COLS + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                              for c in COLS.split(",")) + "\n")
    print("->", csv, f"({len(rows)} righe)")
    if not rows:
        return

    for R in resistances:
        rr = [r for r in rows if r["resistance"] == R]
        if len(rr) < 2:
            continue
        x = np.array([r["energy_true"] for r in rr])
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                                gridspec_kw=dict(height_ratios=[2, 1, 1]))
        axs[0].plot(x, [r["raw"] for r in rr], "o-", color="0.35", label="no correction")
        for v in VARIANTS:
            axs[0].errorbar(x, [r["s_" + v] for r in rr], yerr=[r["err_" + v] for r in rr],
                            fmt=VMK[v] + "-", color=VCOL[v], ms=6, capsize=3,
                            label=f"map per {v}")
        axs[0].set_ylabel("$\\sigma/\\mu$  [%]")
        axs[0].set_title(f"{R} $\\Omega$   $\\quad$   $A_{{tot}} > {A_TOT_MIN:.0f}$ ADC,  "
                         f"$|\\mathrm{{pos\\_eta}}-18| \\leq 0.2$,  "
                         f"$|\\mathrm{{pos\\_phi}}-6| \\leq 0.2$", fontsize=11)
        axs[0].set_xscale("log"); axs[0].grid(alpha=.3); axs[0].legend(fontsize=9)

        ref = np.array([r["s_energy"] for r in rr])
        for v in VARIANTS:
            y = np.array([r["s_" + v] for r in rr])
            axs[1].plot(x, 100 * (y / ref - 1), VMK[v] + "-", color=VCOL[v], ms=6,
                        label=f"map per {v}")
        axs[1].axhline(0, color=VCOL["energy"], lw=1.2)
        axs[1].set_ylabel("rispetto a map per energy  [%]")
        axs[1].grid(alpha=.3); axs[1].legend(fontsize=9)

        axs[2].plot(x, [r["syst_pct"] for r in rr], "d-", color="C4", ms=6,
                    label="spread fra le tre mappe")
        axs[2].set_ylabel("[%]"); axs[2].set_xlabel("$E_{true}$ [GeV]")
        axs[2].grid(alpha=.3); axs[2].legend(fontsize=9)
        fig.tight_layout()
        p = os.path.join(outdir, f"uniformita_maps_{R}ohm.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print("->", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("moments", "apply", "collect"), required=True)
    ap.add_argument("--base", default=".")
    ap.add_argument("--outdir", default="plot/uniformita_maps")
    ap.add_argument("--resistances", nargs="+", type=int, default=[340, 400, 500])
    ap.add_argument("--energies", nargs="*", type=int, default=None)
    runsets.add_argument(ap)
    ap.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    a = ap.parse_args()
    drop, only = runsets.resolve(a.runset, a.exclude_runs)
    momdir = os.path.join(a.outdir, "_mom")
    appdir = os.path.join(a.outdir, "_apply")
    for d in (a.outdir, momdir, appdir):
        os.makedirs(d, exist_ok=True)

    if a.stage == "collect":
        stage_collect(a.outdir, a.resistances)
        return

    for R in a.resistances:
        d, pat = FILES[R]
        files = sorted(glob.glob(os.path.join(a.base, d, pat)),
                       key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)))
        for f in files:
            E = int(re.match(r"(\d+)", os.path.basename(f)).group(1))
            if a.energies is not None and E not in a.energies:
                continue
            print(f"  {R} ohm {E:4d} GeV", flush=True)
            if a.stage == "moments":
                r = stage_moments(f, E, R, drop, momdir, only)
                if r:
                    print(f"      {len(r['per_run'])} run con momenti", flush=True)
            else:
                r = stage_apply(f, E, R, drop, momdir, appdir, only)
                if r:
                    print(f"      raw {r['raw'][0]:.4f}   run {r['s_run'][0]:.4f}   "
                          f"energy {r['s_energy'][0]:.4f}   mean {r['s_mean'][0]:.4f}"
                          + (f"   ({r['n_fallback']} run senza mappa propria)"
                             if r["n_fallback"] else ""), flush=True)


if __name__ == "__main__":
    main()
