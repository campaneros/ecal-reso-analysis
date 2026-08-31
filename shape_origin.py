#!/usr/bin/env python3
"""
Origin of the eta/phi shape distortion: beam, calibration, or geometry?
======================================================================
The observable is the vertex of the parabola fitted to <A_tot> vs the
reconstructed centroid. Three tests, each with a different signature:

TEST 1 -- does it follow the BEAM?
  The beam position b and width w are taken from the HODOSCOPE (pos_x, pos_y),
  which is independent of the calorimeter amplitudes. Conditioning on a noisy
  abscissa biases the vertex by
        x_vertex - x0 = -sigma_x^2 * dln f/dx |_x0 = sigma_x^2 (x0 - b) / w^2
  so d x_vertex / d b = -sigma_x^2 / w^2 : NEGATIVE slope, magnitude giving
  sigma_x / w. Regressing the vertex on -b/w^2 splits it into
        x_vertex = x0 (fixed)  +  sigma_x^2 * (-b/w^2)  (beam-driven)

TEST 2 -- is the fixed part x0 significant?
  If x0 is compatible with zero there is no detector asymmetry to explain.

TEST 3 -- if x0 != 0, is it GEOMETRY or the LATERAL crystals?
  A geometric displacement (matrix off-centre or tilted) moves the maximum of
  the SEED as well. A calibration asymmetry between the two lateral crystals
  moves the maximum of the SUM but leaves the seed where it is. So compare the
  vertex of <A_tot> with the vertex of <A_seed>.

Usage:  python3 shape_origin.py --base <dir with reco_*ohm/> --outdir plot
"""
import argparse, glob, os, re
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

MM_PER_XTAL = 22.0
SCALE = {340: 3500 / 150., 400: 1080 / 40., 500: 3340 / 100.}
COL = {340: "C0", 400: "C1", 500: "C2"}
HALF, NB, NMIN, FITHALF = 0.6, 96, 20, 0.3


def _parab(x, p0, p1, p2):
    return p1 + p2 * (x - p0) ** 2


def vertex(xv, y):
    H, e = np.histogram(xv, bins=NB, range=(-HALF, HALF))
    S, _ = np.histogram(xv, bins=NB, range=(-HALF, HALF), weights=y)
    S2, _ = np.histogram(xv, bins=NB, range=(-HALF, HALF), weights=y ** 2)
    c = 0.5 * (e[:-1] + e[1:])
    with np.errstate(invalid="ignore"):
        m = S / np.maximum(H, 1)
        v = S2 / np.maximum(H, 1) - m ** 2
        er = np.sqrt(np.maximum(v, 0) / np.maximum(H, 1))
    k = (H >= NMIN) & (np.abs(c) <= FITHALF) & (er > 0)
    if k.sum() < 6:
        return None
    mi = Minuit(LeastSquares(c[k], m[k], er[k], _parab),
                p0=0., p1=float(m[k].max()), p2=-100.)
    mi.migrad(); mi.hesse()
    return float(mi.values["p0"]), float(mi.errors["p0"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--outdir", default="plot")
    a = ap.parse_args()

    rows = []
    for R in (340, 400, 500):
        for f in sorted(glob.glob(os.path.join(a.base, f"reco_{R}ohm", "*_merged.root"))):
            E = int(re.match(r"^(\d+)_", os.path.basename(f)).group(1))
            print(f"  {R} ohm {E:4d} GeV", flush=True)
            arr = uproot.open(f)["h4_reco"].arrays(
                ["A", "sel_ieta", "sel_iphi", "A_tot", "pos_eta", "pos_phi",
                 "pos_x", "pos_y", "n_hit_x", "n_hit_y"])
            ie = ak.to_numpy(arr["sel_ieta"])[0]
            ip = ak.to_numpy(arr["sel_iphi"])[0]
            o = {(int(x), int(y)): k for k, (x, y) in enumerate(zip(ie, ip))}
            A = ak.to_numpy(arr["A"]); at = ak.to_numpy(arr["A_tot"])
            et = ak.to_numpy(arr["pos_eta"]) - 18
            ph = ak.to_numpy(arr["pos_phi"]) - 6
            px = ak.to_numpy(arr["pos_x"]).astype(float)
            py = ak.to_numpy(arr["pos_y"]).astype(float)
            nx = ak.to_numpy(arr["n_hit_x"]).astype(float)
            ny = ak.to_numpy(arr["n_hit_y"]).astype(float)
            nom = SCALE[R] * E
            c = at[(at > .5 * nom) & (at < 1.5 * nom)]
            if len(c) < 1000:
                continue
            pk = np.median(c); sg = 1.4826 * np.median(np.abs(c - pk))
            good = (at > pk - 10 * sg) & (at < pk + 10 * sg)
            hodo = good & (np.abs(px) < 200) & (np.abs(py) < 200) & (nx > 0) & (ny > 0)
            if hodo.sum() < 3000:
                continue
            for dirn, x, cut, hv in (("eta", et, ph, px), ("phi", ph, et, py)):
                m = good & (np.abs(cut) < 0.2)
                vt = vertex(x[m], at[m])
                vs = vertex(x[m], A[m, o[(18, 6)]])
                if not (vt and vs):
                    continue
                v = hv[hodo]
                b = np.median(v) / MM_PER_XTAL
                w = 1.4826 * np.median(np.abs(v - np.median(v))) / MM_PER_XTAL
                rows.append(dict(R=R, E=E, dir=dirn, b=b, w=w,
                                 vtot=vt[0], evtot=vt[1], vseed=vs[0], evseed=vs[1]))

    with open(os.path.join(a.outdir, "shape_origin.csv"), "w") as fh:
        fh.write("resistance,energy,direction,beam_centre_xtal,beam_width_xtal,"
                 "vertex_Atot,err_vertex_Atot,vertex_seed,err_vertex_seed\n")
        for r in rows:
            fh.write(f"{r['R']},{r['E']},{r['dir']},{r['b']:+.5f},{r['w']:.5f},"
                     f"{r['vtot']:+.5f},{r['evtot']:.5f},{r['vseed']:+.5f},{r['evseed']:.5f}\n")

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for ax, dirn in zip(axs, ("eta", "phi")):
        s = [r for r in rows if r["dir"] == dirn]
        v = np.array([r["vtot"] for r in s])
        md = np.median(v); mad = 1.4826 * np.median(np.abs(v - md))
        keep = np.abs(v - md) < 3 * mad
        b = np.array([r["b"] for r in s])[keep]
        w = np.array([r["w"] for r in s])[keep]
        y = v[keep]
        X = -b / w ** 2
        sl, ic = np.polyfit(X, y, 1)
        resid = y - (sl * X + ic)
        eic = resid.std() * np.sqrt(1 / keep.sum() + X.mean() ** 2 / ((X - X.mean()) ** 2).sum())
        for R in (340, 400, 500):
            k = np.array([r["R"] == R for r in s])[keep]
            if k.any():
                ax.errorbar(X[k], y[k], yerr=np.array([r["evtot"] for r in s])[keep][k],
                            fmt="o", ms=6, color=COL[R], label=f"{R} $\\Omega$")
        xx = np.linspace(X.min(), X.max(), 10)
        ax.plot(xx, sl * xx + ic, "r-", lw=1.6)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("$-b/w^2$  (beam, from the hodoscope)  [1/crystal]")
        ax.set_ylabel("vertex of $\\langle A_{tot}\\rangle$  [crystal units]")
        ax.set_title(f"{dirn}:  fixed part $x_0$ = {ic:+.4f} $\\pm$ {eic:.4f} "
                     f"({ic/eic:+.1f}$\\sigma$)\n"
                     f"slope = $\\sigma_x^2$ = {sl:+.5f}  ->  "
                     f"$\\sigma_x$ = {np.sqrt(abs(sl))*MM_PER_XTAL:.2f} mm", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        print(f"{dirn}: x0 = {ic:+.4f} +- {eic:.4f}  ({ic/eic:+.1f} sigma)")
        vt = np.array([r["vtot"] for r in s]); vs = np.array([r["vseed"] for r in s])
        d = vt - vs
        mdd = np.median(d); madd = 1.4826 * np.median(np.abs(d - mdd))
        print(f"      vertex(A_tot) - vertex(seed) = {mdd:+.4f} +- {madd/np.sqrt(len(d)):.4f}"
              f"  -> {'LATERAL crystals' if abs(mdd) > 3*madd/np.sqrt(len(d)) else 'nothing'}")
    fig.suptitle("Is the shape distortion driven by the beam?  Vertex vs the hodoscope beam term\n"
                 "intercept = fixed (detector) component, slope = beam-driven component",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "shape_origin.png"), dpi=150)
    print("scritto shape_origin.csv e shape_origin.png")


if __name__ == "__main__":
    main()
