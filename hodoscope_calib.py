"""
Hodoscope calibration from the RESPONSE, so the position cut can be moved off the
ECAL centroid and onto the hodoscope.

Why not the centroid. The selection used everywhere else cuts on |pos_eta - 18| and
|pos_phi - 6|, and the centroid is built from the very amplitudes whose resolution is
being measured. Cutting on the hodoscope instead removes that circularity, but the
hodoscope reads millimetres with an arbitrary origin, so it needs an offset and a
scale first.

Which planes. There are two planes per view. In y the first one is much less
efficient (42 % of events with a single cluster against 65 % for the second), so only
y2 is used; in x both planes are healthy and their average is taken, which is also
the better position estimate. Only events with exactly one cluster in the planes used
enter the calibration.

How the two constants are obtained, without ever regressing against the centroid:

  offset  the crystal centre is where the response is maximum, so it is the vertex of
          the parabola fitted to <A_tot> against the hodoscope coordinate.
  scale   the same physical parabola, expressed in millimetres and in crystal units,
          has relative curvatures c_mm and c_crystal whose ratio is the square of the
          scale:  W = sqrt(c_crystal / c_mm)  millimetres per crystal. c_crystal comes
          from profili_pernorm.py. Measured this way W = 23.9 +- 1.3 mm in x and
          21.2 +- 2.6 mm in y, against the 24.2 +- 2.0 mm crystal pitch: the scale is
          recovered from the response alone.

Fit ranges. x is used in [-13, -2] mm and y2 in [0, 12] mm, where each plane behaves;
outside those the profiles are not parabolic. The y range is one-sided, so the y
vertex is less well determined than the x one and its energy-to-energy spread is
correspondingly larger.

Usage:
  python3 plot/hodoscope_calib.py --base . --outdir plot/hodo [--resistances 340]
"""

import argparse, os, glob, re, csv
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import runsets

A_TOT_MIN = 100.
CORE_FRAC = 0.10                 # |A_tot/peak - 1| < CORE_FRAC for the response profile
XRANGE, YRANGE = (-13., -2.), (0., 12.)
NBINS, NMIN_BIN, NMIN_FIT = 24, 150, 3000
FILES = {340: ("reco_340ohm", "*_merged.root"),
         400: ("reco_400ohm", "*_400_merged.root"),
         500: ("reco_500ohm", "*_500_merged.root")}


def hodo_xy(a):
    """x = mean of the two x planes, y = second plane only; nan where unusable."""
    n1 = ak.to_numpy(a["hodo_x1_nclusters"]); n2 = ak.to_numpy(a["hodo_x2_nclusters"])
    x1 = ak.to_numpy(ak.firsts(ak.pad_none(a["hodo_x1_pos"], 1)))
    x2 = ak.to_numpy(ak.firsts(ak.pad_none(a["hodo_x2_pos"], 1)))
    ok = (n1 == 1) & (n2 == 1) & np.isfinite(x1) & np.isfinite(x2)
    x = np.where(ok, (x1 + x2) / 2., np.nan)
    ny = ak.to_numpy(a["hodo_y2_nclusters"])
    y = ak.to_numpy(ak.firsts(ak.pad_none(a["hodo_y2_pos"], 1)))
    return x, np.where((ny == 1) & np.isfinite(y), y, np.nan)


def parab(v, w, lo, hi):
    """Vertex and relative curvature of <w> against v. Returns (vertex, c_rel, n)."""
    m = np.isfinite(v) & (v > lo) & (v < hi)
    if m.sum() < NMIN_FIT:
        return None
    e = np.linspace(lo, hi, NBINS + 1)
    i = np.clip(np.digitize(v[m], e) - 1, 0, NBINS - 1)
    xs, ys, ns = [], [], []
    for k in range(NBINS):
        q = i == k
        if q.sum() < NMIN_BIN:
            continue
        xs.append(v[m][q].mean()); ys.append(w[m][q].mean()); ns.append(q.sum())
    if len(xs) < 8:
        return None
    xs, ys, ns = np.array(xs), np.array(ys), np.array(ns, float)
    p = np.polyfit(xs, ys, 2, w=np.sqrt(ns))
    if p[0] >= 0:                       # no maximum: the fit did not find the crystal
        return None
    vx = -p[1] / (2 * p[0])
    return float(vx), float(100 * p[0] / np.polyval(p, vx)), int(ns.sum())


def crystal_curvature(plotdir, R):
    """Relative curvature in crystal units, from profili_pernorm.py."""
    out = {}
    f = os.path.join(plotdir, "profili", "profili_pernorm.csv")
    if not os.path.exists(f):
        return out
    for r in csv.DictReader(open(f)):
        if int(r["resistance"]) == R:
            out[(int(r["energy"]), r["coord"])] = float(r["rel_pct"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--outdir", default="plot/hodo")
    ap.add_argument("--plotdir", default="plot")
    ap.add_argument("--resistances", nargs="+", type=int, default=[340, 400, 500])
    ap.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    runsets.add_argument(ap)
    a = ap.parse_args()
    drop, only = runsets.resolve(a.runset, a.exclude_runs)
    os.makedirs(a.outdir, exist_ok=True)

    rows, summary = [], []
    for R in a.resistances:
        cry = crystal_curvature(a.plotdir, R)
        d, pat = FILES[R]
        for f in sorted(glob.glob(os.path.join(a.base, d, pat)),
                        key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1))):
            E = int(re.match(r"(\d+)", os.path.basename(f)).group(1))
            arr = uproot.open(f)["h4_reco"].arrays(
                ["run", "A_tot", "hodo_x1_nclusters", "hodo_x1_pos", "hodo_x2_nclusters",
                 "hodo_x2_pos", "hodo_y2_nclusters", "hodo_y2_pos"], library="ak")
            run = ak.to_numpy(arr["run"]); at = ak.to_numpy(arr["A_tot"])
            keep = at > A_TOT_MIN
            if drop:
                keep &= ~np.isin(run, drop)
            if len(only):
                keep &= np.isin(run, only)
            if keep.sum() < NMIN_FIT:
                continue
            x, y = hodo_xy(arr)
            pk = float(np.median(at[keep]))
            core = keep & (np.abs(at / pk - 1) < CORE_FRAC)
            row = dict(resistance=R, energy=E, nev=int(core.sum()), peak=pk)
            for tag, v, rng, coord in (("x", x, XRANGE, "pos_eta"),
                                       ("y", y, YRANGE, "pos_phi")):
                q = parab(v[core], at[core], *rng)
                cc = cry.get((E, coord))
                W = (np.sqrt(cc / q[1]) if (q and cc and q[1] < 0 and cc < 0) else np.nan)
                row.update({f"x0_{tag}": q[0] if q else np.nan,
                            f"c_{tag}": q[1] if q else np.nan,
                            f"n_{tag}": q[2] if q else 0, f"W_{tag}": W})
            rows.append(row)
            print(f"  {R} ohm {E:>4} GeV: vertice x {row['x0_x']:+6.2f} mm  W_x "
                  f"{row['W_x']:5.2f} mm | vertice y {row['x0_y']:+6.2f} mm  W_y "
                  f"{row['W_y']:5.2f} mm", flush=True)

        q = [r for r in rows if r["resistance"] == R]
        s = dict(resistance=R)
        for k in ("x0_x", "W_x", "x0_y", "W_y"):
            v = np.array([r[k] for r in q if np.isfinite(r[k])])
            s[k] = float(np.median(v)) if len(v) else np.nan
            s[k + "_mad"] = float(np.median(np.abs(v - np.median(v)))) if len(v) else np.nan
        summary.append(s)
        print(f"  --> {R} ohm: x0_x {s['x0_x']:+.2f} +- {s['x0_x_mad']:.2f} mm, "
              f"W_x {s['W_x']:.2f} +- {s['W_x_mad']:.2f} mm | x0_y {s['x0_y']:+.2f} "
              f"+- {s['x0_y_mad']:.2f} mm, W_y {s['W_y']:.2f} +- {s['W_y_mad']:.2f} mm")

    cols = "resistance,energy,nev,peak,x0_x,c_x,n_x,W_x,x0_y,c_y,n_y,W_y"
    p = os.path.join(a.outdir, "hodoscope_calib.csv")
    with open(p, "w") as fh:
        fh.write(cols + "\n")
        for r in rows:
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                              for c in cols.split(",")) + "\n")
    print("->", p)
    p = os.path.join(a.outdir, "hodoscope_calib_summary.csv")
    with open(p, "w") as fh:
        fh.write("resistance,x0_x,x0_x_mad,W_x,W_x_mad,x0_y,x0_y_mad,W_y,W_y_mad\n")
        for s in summary:
            fh.write(f"{s['resistance']},{s['x0_x']:.4f},{s['x0_x_mad']:.4f},"
                     f"{s['W_x']:.4f},{s['W_x_mad']:.4f},{s['x0_y']:.4f},"
                     f"{s['x0_y_mad']:.4f},{s['W_y']:.4f},{s['W_y_mad']:.4f}\n")
    print("->", p)

    if rows:
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
        for j, tag in enumerate("xy"):
            for R, mk in ((340, "o"), (400, "s"), (500, "^")):
                q = [r for r in rows if r["resistance"] == R and np.isfinite(r[f"x0_{tag}"])]
                if q:
                    axs[j].plot([r["energy"] for r in q], [r[f"x0_{tag}"] for r in q],
                                mk, ms=6, label=f"{R} $\\Omega$")
            for s in summary:
                if np.isfinite(s[f"x0_{tag}"]):
                    axs[j].axhline(s[f"x0_{tag}"], color="0.4", lw=1, ls="--")
            axs[j].set_xscale("log")
            axs[j].set_xlabel("$E_{nom}$ [GeV]")
            axs[j].set_ylabel(f"vertex of $\\langle A_{{tot}}\\rangle$ vs hodo {tag}  [mm]")
            axs[j].grid(alpha=.3); axs[j].legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(a.outdir, "hodoscope_calib.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print("->", p)


if __name__ == "__main__":
    main()
