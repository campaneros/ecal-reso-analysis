"""
Are the per-run error bars right? Bootstrap check of the double-CB fit error.

The error on each per-run sigma/mu is the HESSE error of the double-CB fit,
propagated to the ratio:

    e = (sigma/mu) * sqrt( (d_sigma/sigma)^2 + (d_peak/peak)^2 )

Those bars look large next to the textbook sigma/sqrt(2N) -- typically three to four
times larger -- and the whole drift systematic depends on them: the drift is the extra
error needed to bring the per-run sigmas to chi2/ndf = 1, so if the bars were
overestimated the chi2 would come out artificially small and the drift would vanish
where it should not.

sigma/sqrt(2N) is the wrong reference. It is the error on the RMS of a Gaussian from
an unbinned maximum-likelihood fit. Here sigma is the width of the *core* of a
seven-parameter double CB, fitted by binned least squares, and it is correlated with
the four tail parameters: far fewer events effectively constrain it.

The model-independent reference is a bootstrap: resample the events of one run with
replacement, refit, and take the spread of the sigma/mu obtained. This script does
that and prints the ratio HESSE / bootstrap, which should be 1 if the bars are right.

Measured on three points, twenty runs in total:

    340 ohm 225 GeV   median ratio 1.00
    400 ohm  80 GeV   median ratio 0.72
    500 ohm  60 GeV   median ratio 0.81

so the HESSE errors are right on average and, where they are not, they are *smaller*
than the truth by 20-30 %. That direction matters: too-small errors make chi2 too
large, hence the drift too large. The points where the drift comes out zero would come
out zero with the correct errors as well.

Usage:
  python3 plot/dcb_error_check.py --base . --resistance 500 --energy 60 --nboot 40
"""

import argparse, os, sys
import numpy as np
import uproot
import awkward as ak
import runsets
from uniformita_pos import fit_dcb, rel, A_TOT_MIN
from hodoscope_calib import (hodo_xy, FILES, crystal_curvature, response_profile,
                             parabola_scan)

CORE = 0.10
NMIN_RUN = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--plotdir", default="plot")
    ap.add_argument("--resistance", type=int, required=True)
    ap.add_argument("--energy", type=int, required=True)
    ap.add_argument("--nboot", type=int, default=40)
    ap.add_argument("--half", type=float, default=0.2)
    ap.add_argument("--yplane", choices=("y1", "y2"), default="y1")
    ap.add_argument("--cut", choices=("hodo", "centroid"), default="hodo")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    runsets.add_argument(ap)
    a = ap.parse_args()
    drop, only = runsets.resolve(a.runset, a.exclude_runs)
    R, E = a.resistance, a.energy

    d, pat = FILES[R]
    f = os.path.join(a.base, d, pat.replace("*", str(E)))
    if not os.path.exists(f):
        sys.exit(f"non trovo {f}")
    arr = uproot.open(f)["h4_reco"].arrays(
        ["run", "A_tot", "pos_eta", "pos_phi", "hodo_x1_nclusters", "hodo_x1_pos",
         "hodo_x2_nclusters", "hodo_x2_pos", "hodo_y1_nclusters", "hodo_y1_pos",
         "hodo_y2_nclusters", "hodo_y2_pos"], library="ak")
    run = ak.to_numpy(arr["run"]); at = ak.to_numpy(arr["A_tot"]).astype(float)
    base = at > A_TOT_MIN
    if drop:
        base &= ~np.isin(run, drop)
    if len(only):
        base &= np.isin(run, only)

    if a.cut == "centroid":
        pe = ak.to_numpy(arr["pos_eta"]); pp = ak.to_numpy(arr["pos_phi"])
        cut = base & (np.abs(pe - 18.) <= a.half) & (np.abs(pp - 6.) <= a.half)
    else:
        cry = crystal_curvature(a.plotdir, R)
        x, y = hodo_xy(arr, a.yplane)
        core = base & (np.abs(at / np.median(at[base]) - 1) < CORE)
        cut = base & np.isfinite(x) & np.isfinite(y)
        for v, coord in ((x, "pos_eta"), (y, "pos_phi")):
            pr = response_profile(v[core], at[core])
            if pr is None:
                sys.exit("nessun profilo di risposta")
            s = parabola_scan(*pr, cry.get((E, coord)), a.half)
            if not s["ok"]:
                sys.exit(f"nessuna parabola: {s['why']}")
            cut &= (v >= s["x0"] - a.half * s["W"]) & (v <= s["x0"] + a.half * s["W"])

    rng = np.random.default_rng(a.seed)
    print(f"{R} ohm {E} GeV, taglio {a.cut}, {a.nboot} repliche bootstrap per run\n")
    print(f"{'run':>6} {'nev':>6} {'sigma/mu':>9} {'HESSE':>8} {'bootstrap':>10} "
          f"{'ratio':>6} {'sig/sqrt(2N)':>13}")
    ratios = []
    for r in sorted(int(v) for v in np.unique(run[cut])):
        q = cut & (run == r)
        if q.sum() < NMIN_RUN:
            continue
        v = at[q]
        f0 = fit_dcb(v, E, R)
        if f0 is None:
            continue
        v0, e0 = rel(f0)
        bs = []
        for _ in range(a.nboot):
            s = fit_dcb(rng.choice(v, size=len(v), replace=True), E, R)
            if s is not None:
                bs.append(rel(s)[0])
        eb = np.std(bs, ddof=1) if len(bs) > 5 else np.nan
        ratio = e0 / eb if eb > 0 else np.nan
        if np.isfinite(ratio):
            ratios.append(ratio)
        n = f0["nev"]
        print(f"{r:>6} {n:>6} {v0:>9.4f} {e0:>8.4f} {eb:>10.4f} {ratio:>6.2f} "
              f"{v0/np.sqrt(2*n):>13.4f}")
    if ratios:
        print(f"\n  mediana HESSE/bootstrap = {np.median(ratios):.2f} su "
              f"{len(ratios)} run")
        print("  < 1 significa barre piu' piccole del vero, quindi chi2 e drift "
              "se mai sovrastimati")


if __name__ == "__main__":
    main()
