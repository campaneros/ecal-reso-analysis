"""
Response profiles against the hodoscope, with the fitted parabola and the window used.

One panel per energy: <A_tot> normalised to its maximum against the hodoscope
coordinate, the parabola fitted on it, its vertex, and the two vertical lines of the
selection window actually applied by resolution_hodo.py. It is the picture behind
the cut, and the place to look when a point comes out odd.

Usage:
  python3 plot/hodo_windows.py --base . --outdir plot/hodo_parab \\
      --window parabola [--resistances 340]
"""

import argparse, os, glob, re, csv
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import runsets
from hodoscope_calib import hodo_xy, FILES, parab, crystal_curvature
from resolution_hodo import (window_from_response, profile_peak, YSEARCH,
                             PLATEAU_TOL, PARAB_HALF, W_MIN, W_MAX)

A_TOT_MIN, CORE = 100., 0.10
COORD = (("x", "hodo x = (x1+x2)/2  [mm]", "pos_eta"),
         ("y", "hodo y2  [mm]", "pos_phi"))


def profile(v, w, lo, hi, nb=40, nmin=100):
    m = np.isfinite(v) & (v > lo) & (v < hi)
    if m.sum() < 1000:
        return None
    e = np.linspace(lo, hi, nb + 1)
    i = np.clip(np.digitize(v[m], e) - 1, 0, nb - 1)
    xs, ys = [], []
    for k in range(nb):
        q = i == k
        if q.sum() < nmin:
            continue
        xs.append(v[m][q].mean()); ys.append(w[m][q].mean())
    return (np.array(xs), np.array(ys)) if len(xs) > 5 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--outdir", default="plot/hodo_parab")
    ap.add_argument("--plotdir", default="plot")
    ap.add_argument("--resistances", nargs="+", type=int, default=[340, 400, 500])
    ap.add_argument("--window", choices=("plateau", "parabola"), default="parabola")
    ap.add_argument("--half", type=float, default=0.2)
    ap.add_argument("--tol", type=float, default=PLATEAU_TOL)
    ap.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    runsets.add_argument(ap)
    a = ap.parse_args()
    drop, only = runsets.resolve(a.runset, a.exclude_runs)
    os.makedirs(a.outdir, exist_ok=True)

    for R in a.resistances:
        cry = crystal_curvature(a.plotdir, R)
        d, pat = FILES[R]
        files = sorted(glob.glob(os.path.join(a.base, d, pat)),
                       key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)))
        n = len(files)
        for tag, xlab, coord in COORD:
            nc = 4; nr = int(np.ceil(n / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.4 * nr), squeeze=False)
            for k, f in enumerate(files):
                E = int(re.match(r"(\d+)", os.path.basename(f)).group(1))
                ax = axs[k // nc][k % nc]
                arr = uproot.open(f)["h4_reco"].arrays(
                    ["run", "A_tot", "hodo_x1_nclusters", "hodo_x1_pos",
                     "hodo_x2_nclusters", "hodo_x2_pos", "hodo_y2_nclusters",
                     "hodo_y2_pos"], library="ak")
                run = ak.to_numpy(arr["run"]); at = ak.to_numpy(arr["A_tot"]).astype(float)
                keep = at > A_TOT_MIN
                if drop:
                    keep &= ~np.isin(run, drop)
                if len(only):
                    keep &= np.isin(run, only)
                if keep.sum() < 2000:
                    ax.set_axis_off(); continue
                x, y = hodo_xy(arr)
                v = (x if tag == "x" else y)[keep]
                w = at[keep]
                w = w[np.isfinite(v)] if False else w
                core = np.abs(at[keep] / np.median(at[keep]) - 1) < CORE
                vv, ww = v[core], at[keep][core]
                g = np.isfinite(vv)
                if g.sum() < 2000:
                    ax.set_axis_off(); continue
                lo = YSEARCH[0] if tag == "y" else float(np.percentile(vv[g], 0.5))
                hi = YSEARCH[1] if tag == "y" else float(np.percentile(vv[g], 99.5))
                pr = profile(vv, ww, lo, hi)
                if pr is None:
                    ax.set_axis_off(); continue
                px, py = pr
                top = py.max()
                ax.plot(px, py / top, "o", ms=3, color="C0")
                win = None
                q = None
                if a.window == "parabola":
                    pk = profile_peak(vv, ww, lo if tag == "y" else None,
                                      hi if tag == "y" else None)
                    flo, fhi = ((max(pk - PARAB_HALF, lo), min(pk + PARAB_HALF, hi))
                                if pk is not None else (lo, hi))
                    q = parab(vv, ww, flo, fhi)
                    cc = cry.get((E, coord))
                    if (q is not None and cc is not None and q[1] < 0 and cc < 0
                            and flo <= q[0] <= fhi):
                        W = float(np.sqrt(cc / q[1]))
                        if W_MIN <= W <= W_MAX:
                            win = (q[0] - a.half * W, q[0] + a.half * W)
                else:
                    q = parab(vv, ww, lo, hi)
                    r = (window_from_response(vv, ww, a.tol, lo, hi) if tag == "y"
                         else window_from_response(vv, ww, a.tol))
                    if r:
                        win = (r[0], r[1])
                if q is not None:
                    xs = np.linspace(px.min(), px.max(), 200)
                    c2 = q[1] * top / 100.
                    ax.plot(xs, (top + c2 * (xs - q[0]) ** 2) / top, "r-", lw=1.6)
                    ax.axvline(q[0], color="r", lw=1, ls=":")
                if win:
                    for s in win:
                        ax.axvline(s, color="k", lw=1.4)
                    ax.set_title(f"{E} GeV   [{win[0]:+.1f}, {win[1]:+.1f}] mm", fontsize=9)
                else:
                    ax.set_title(f"{E} GeV   nessuna finestra", fontsize=9, color="C3")
                ax.set_xlabel(xlab, fontsize=8); ax.set_ylabel("$\\langle A_{tot}\\rangle$ / max", fontsize=8)
                ax.tick_params(labelsize=7); ax.grid(alpha=.3)
            for k in range(n, nr * nc):
                axs[k // nc][k % nc].set_axis_off()
            fig.suptitle(f"{R} $\\Omega$   $\\quad$   window: {a.window},  half {a.half} crystals"
                         + ("" if tag == "x" else f",  y search {YSEARCH}"), fontsize=12)
            fig.tight_layout()
            p = os.path.join(a.outdir, f"hodo_windows_{tag}_{R}ohm.png")
            fig.savefig(p, dpi=130); plt.close(fig)
            print("->", p)


if __name__ == "__main__":
    main()
