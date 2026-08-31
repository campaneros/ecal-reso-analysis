#!/usr/bin/env python3
"""
Inter-calibration of the lateral crystals using the HODOSCOPE position
======================================================================
WHY
  With the ECAL centroid (pos_eta / pos_phi) the symmetry axis x_s and the
  calibration ratio k are degenerate: the profiles A_L(x), A_H(x) fall by a
  factor ~30 over the scanned range, so d ln A/dx ~ 6 per crystal and
        d ln k / d x_s  ~  -2 |d ln A/dx|  ~  -12 per crystal
  A 0.003-crystal error on x_s already gives 3% on k. Measured across the 29
  datasets, x_s and ln k are anticorrelated with r = -0.90 and exactly that
  slope, and the whole scatter of k equals MAD(x_s) x 11. So that fit measures
  ONE combination, not two numbers -- and it is circular, because the abscissa
  is built from the very amplitudes being calibrated.

WHAT CHANGES HERE
  pos_x / pos_y come from the hodoscope: a position measurement completely
  independent of the calorimeter amplitudes. The degeneracy between x_s and k
  is still there mathematically, BUT x_s is now a genuine geometric constant --
  the position, in hodoscope millimetres, of the crystal boundary -- which must
  be the same in every run of the same setup. So it can be fitted once globally
  and then held fixed, which is what breaks the degeneracy.

METHOD
  1. Events with a valid hodoscope cluster in both views (|pos| < 200 mm,
     n_hit > 0) and A_tot within peak +- nsigma.
  2. Strip cut on the other coordinate, |other - axis_other| < STRIP mm,
     iterated twice so the strip is centred on the fitted axis and not on the
     beam.
  3. Profiles <A_L>(x) and <A_H>(x) in bins of BINMM millimetres.
  4. Fit of x_s and k by minimising the spread of
        g(u) = ln A_L(x_s - u) - ln A_H(x_s + u)
     over u in [UMIN, UMAX] mm; k = exp(<g>).
  5. Second pass with x_s fixed to the global median: then k is the only free
     parameter and its scatter across datasets is a real measurement.

OUTCOME (read this before using the numbers)
  This route does NOT work either, and the reason is structural, not technical.
  The beam is always parked on the central crystal: only 0.05% - 1.2% of the
  events land closer to a lateral crystal than to the seed, and even then the
  lateral crystal carries ~3% of the energy. The two mirror profiles are
  therefore built from wildly unequal statistics, they are not mirror images of
  each other, and the axis scan escapes into regions with almost no data (the y
  pair returned k ~ 25, which is nonsense, not a calibration).
  With this dataset the inter-calibration of the lateral crystals is NOT
  measurable: not from the ECAL centroid (x_s and k are degenerate, and the
  abscissa is built from the amplitudes being calibrated) and not from the
  hodoscope (no illumination). It needs dedicated runs with the beam centred on
  each lateral crystal -- the standard test-beam inter-calibration scan.
  The fit here is kept, but hardened: full coverage in u is required and the
  axis is confined to +-3 mm around the beam, so failures show up as missing
  entries instead of absurd numbers.

Usage:
  python3 channels_vs_hodoscope.py --base <dir with reco_*ohm/> --outdir plot
"""
import argparse, glob, os, re
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINMM, NMIN = 0.5, 100
STRIP = 4.0
UMIN, UMAX, USTEP = 3.0, 9.0, 0.5
SCALE = {340: 3500 / 150., 400: 1080 / 40., 500: 3340 / 100.}
COL = {340: "C0", 400: "C1", 500: "C2"}
PAIRS = {"x": ((17, 6), (19, 6)), "y": ((18, 5), (18, 7))}
UGRID = np.arange(UMIN, UMAX + 1e-9, USTEP)


def profile(x, y, lo, hi):
    nb = max(int((hi - lo) / BINMM), 4)
    H, e = np.histogram(x, bins=nb, range=(lo, hi))
    S, _ = np.histogram(x, bins=nb, range=(lo, hi), weights=y)
    S2, _ = np.histogram(x, bins=nb, range=(lo, hi), weights=y ** 2)
    c = 0.5 * (e[:-1] + e[1:])
    ok = H >= NMIN
    with np.errstate(invalid="ignore", divide="ignore"):
        m = S / np.maximum(H, 1)
        v = S2 / np.maximum(H, 1) - m ** 2
        err = np.sqrt(np.maximum(v, 0) / np.maximum(H, 1))
    return c[ok], m[ok], err[ok]


def fit_axis_k(cL, mL, cH, mH, xs_grid, xs_fixed=None):
    grid = [xs_fixed] if xs_fixed is not None else xs_grid
    best = None
    for xs in grid:
        l = np.interp(xs - UGRID, cL, mL, left=np.nan, right=np.nan)
        h = np.interp(xs + UGRID, cH, mH, left=np.nan, right=np.nan)
        ok = np.isfinite(l) & np.isfinite(h) & (l > 0) & (h > 0)
        if ok.sum() < len(UGRID):        # copertura COMPLETA in u, altrimenti
            continue                     # il minimo puo' scappare dove non ci sono dati
        g = np.log(l[ok]) - np.log(h[ok])
        sd = g.std()
        n = ok.sum()
        if best is None or sd < best[0]:
            best = (sd, float(xs), float(np.exp(g.mean())), int(n),
                    float(g.std() / np.sqrt(n)))
    return best


def analyse(path, energy, resistance, nsigma, xs_fixed=None):
    t = uproot.open(path)["h4_reco"]
    arr = t.arrays(["A", "sel_ieta", "sel_iphi", "A_tot",
                    "pos_x", "pos_y", "n_hit_x", "n_hit_y"])
    ie = ak.to_numpy(arr["sel_ieta"])[0]
    ip = ak.to_numpy(arr["sel_iphi"])[0]
    o = {(int(e), int(p)): k for k, (e, p) in enumerate(zip(ie, ip))}
    A = ak.to_numpy(arr["A"])
    atot = ak.to_numpy(arr["A_tot"])
    px = ak.to_numpy(arr["pos_x"]).astype(float)
    py = ak.to_numpy(arr["pos_y"]).astype(float)
    nx = ak.to_numpy(arr["n_hit_x"]).astype(float)
    ny = ak.to_numpy(arr["n_hit_y"]).astype(float)

    hodo = (np.abs(px) < 200) & (np.abs(py) < 200) & (nx > 0) & (ny > 0)
    nom = SCALE[resistance] * energy
    core = atot[(atot > 0.5 * nom) & (atot < 1.5 * nom)]
    if len(core) < 500 or hodo.sum() < 3000:
        return None
    pk = np.median(core)
    sg = 1.4826 * np.median(np.abs(core - pk))
    good = hodo & (atot > pk - nsigma * sg) & (atot < pk + nsigma * sg)

    out = {}
    axis = {"x": np.median(px[good]), "y": np.median(py[good])}
    for _ in range(2):                       # due iterazioni sulla striscia
        for dirn, xv, ov in (("x", px, py), ("y", py, px)):
            L, H = PAIRS[dirn]
            other = "y" if dirn == "x" else "x"
            m = good & (np.abs(ov - axis[other]) < STRIP)
            if m.sum() < 2000:
                continue
            lo, hi = np.percentile(xv[m], [0.5, 99.5])
            cL, mL, _ = profile(xv[m], A[m, o[L]], lo, hi)
            cH, mH, _ = profile(xv[m], A[m, o[H]], lo, hi)
            if len(cL) < 8 or len(cH) < 8:
                continue
            gr = np.arange(np.median(xv[m]) - 3, np.median(xv[m]) + 3, 0.05)
            b = fit_axis_k(cL, mL, cH, mH, gr,
                           xs_fixed=(xs_fixed or {}).get(dirn))
            if b:
                axis[dirn] = b[1]
                out[dirn] = dict(resid=b[0], axis=b[1], k=b[2], npt=b[3], err_k=b[2] * b[4],
                                 prof=(cL, mL, cH, mH), nev=int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--outdir", default="plot")
    ap.add_argument("--nsigma", type=float, default=10.)
    a = ap.parse_args()

    files = []
    for R in (340, 400, 500):
        for f in sorted(glob.glob(os.path.join(a.base, f"reco_{R}ohm", "*_merged.root"))):
            m = re.match(r"^(\d+)_", os.path.basename(f))
            if m:
                files.append((f, int(m.group(1)), R))

    # ---------------- pass 1: axis free
    res1 = {}
    for f, E, R in files:
        print(f"  pass1  {R} ohm {E:4d} GeV", flush=True)
        r = analyse(f, E, R, a.nsigma)
        if r:
            res1[(R, E)] = r

    glob_axis = {}
    for dirn in ("x", "y"):
        v = [r[dirn]["axis"] for r in res1.values() if dirn in r]
        glob_axis[dirn] = float(np.median(v))
        print(f"asse globale {dirn}: mediana {np.median(v):+.3f} mm, "
              f"MAD {1.4826*np.median(np.abs(np.array(v)-np.median(v))):.3f} mm, N={len(v)}")

    # ---------------- pass 2: axis fixed to the global value
    res2 = {}
    for f, E, R in files:
        print(f"  pass2  {R} ohm {E:4d} GeV", flush=True)
        r = analyse(f, E, R, a.nsigma, xs_fixed=glob_axis)
        if r:
            res2[(R, E)] = r

    csv = os.path.join(a.outdir, "hodoscope_calibration.csv")
    with open(csv, "w") as fh:
        fh.write("resistance,energy,direction,axis_free_mm,k_free,"
                 "axis_fixed_mm,k_fixed,err_k_fixed,residual_rms,n_events\n")
        for (R, E) in sorted(res1):
            for dirn in ("x", "y"):
                if dirn not in res1[(R, E)] or dirn not in res2.get((R, E), {}):
                    continue
                p, q = res1[(R, E)][dirn], res2[(R, E)][dirn]
                fh.write(f"{R},{E},{dirn},{p['axis']:+.4f},{p['k']:.5f},"
                         f"{q['axis']:+.4f},{q['k']:.5f},{q['err_k']:.5f},"
                         f"{q['resid']:.5f},{q['nev']}\n")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, key, lab, ref in ((axs[0], "axis", "symmetry axis [hodoscope mm]", None),
                              (axs[1], "k", "calibration ratio $k$ (axis fixed)", 1.)):
        src = res1 if key == "axis" else res2
        for dirn, mk in (("x", "o"), ("y", "s")):
            for R in (340, 400, 500):
                pts = sorted([(E, src[(RR, E)][dirn][key], src[(RR, E)][dirn].get("err_k", 0))
                              for (RR, E) in src if RR == R and dirn in src[(RR, E)]])
                if pts:
                    ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                                yerr=[p[2] for p in pts] if key == "k" else None,
                                fmt=mk, ms=6, color=COL[R],
                                mfc="none" if dirn == "y" else COL[R],
                                label=f"{R} $\\Omega$, {dirn}")
        if ref is not None:
            ax.axhline(ref, color="k", lw=1.2)
        else:
            for dirn, ls in (("x", "--"), ("y", ":")):
                ax.axhline(glob_axis[dirn], color="k", lw=1, ls=ls)
        ax.set_xlabel("Beam energy [GeV]")
        ax.set_ylabel(lab)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Inter-calibration from the hodoscope position (independent of the amplitudes)\n"
                 "left: symmetry axis fitted per dataset -- right: k with the axis fixed to the "
                 "global median\nfilled = x pair (17,6)/(19,6), open = y pair (18,5)/(18,7)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "hodoscope_calibration.png"), dpi=150)
    plt.close(fig)
    print("scritto", csv)


if __name__ == "__main__":
    main()
