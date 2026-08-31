#!/usr/bin/env python3
"""
Symmetry point RUN BY RUN -- separating beam position from beam shape
=====================================================================
WHY PER RUN
  The measured curve is not the true response R(x) but
        A(x) = Int R(t) p(t|x) dt,      p(t|x) ~ f(t) g(x|t)
  so it depends on the beam f. Correcting for that by regressing on -b/w^2
  assumes a GAUSSIAN beam (only then is dln f/dx = -(x0-b)/w^2). The beam here
  is not Gaussian, and worse: the collimators change with energy (20 GeV
  -5/5/-5/5, 40 GeV -8/8/-8/8, 100 GeV -15/15/-8/8, ...), so between energies
  the beam SHAPE changes, not just its position.

  The collimators are however CONSTANT within each energy (only exception:
  500 ohm 60 GeV). So doing everything run by run, inside one energy the beam
  shape is fixed and only its position drifts. That separates the three cases
  without assuming anything about the beam shape:

    x_c varies within an energy, tracking the beam position -> beam position
    x_c constant within an energy but different between energies -> beam shape
    x_c constant everywhere -> detector

METHOD
  Same model-independent estimator as symmetry_point.py: the x_c minimising
      S(x_c) = < [ (A(x_c+u) - A(x_c-u)) / A(x_c) ]^2 >_u ,  equal weights in u.
  Coarser bins (0.025 crystals) because a single run has fewer events; runs with
  fewer than MINEV events in the strip are skipped. Beam position and width per
  run come from the HODOSCOPE.

Usage:  python3 symmetry_per_run.py --base <dir with reco_*ohm/> --outdir plot
"""
import argparse, glob, os, re
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM_PER_XTAL = 22.0
SCALE = {340: 3500 / 150., 400: 1080 / 40., 500: 3340 / 100.}
COL = {340: "C0", 400: "C1", 500: "C2"}
HALF, BINW, NMIN, MINEV = 0.6, 0.025, 20, 5000
UMIN, UMAX = 0.05, 0.20
XCGRID = np.arange(-0.20, 0.2001, 0.0025)
NBOOT = 150


def profile(x, y):
    nb = int(2 * HALF / BINW)
    H, e = np.histogram(x, bins=nb, range=(-HALF, HALF))
    S, _ = np.histogram(x, bins=nb, range=(-HALF, HALF), weights=y)
    S2, _ = np.histogram(x, bins=nb, range=(-HALF, HALF), weights=y ** 2)
    c = 0.5 * (e[:-1] + e[1:])
    ok = H >= NMIN
    with np.errstate(invalid="ignore", divide="ignore"):
        m = S / np.maximum(H, 1)
        v = S2 / np.maximum(H, 1) - m ** 2
        err = np.sqrt(np.maximum(v, 0) / np.maximum(H, 1))
    return c[ok], m[ok], err[ok]


def _scan(c, m):
    u = np.arange(UMIN, UMAX + 1e-9, BINW)
    xs = XCGRID[(XCGRID - UMAX >= c.min()) & (XCGRID + UMAX <= c.max())]
    if len(xs) < 5 or len(u) < 3:
        return None
    AP = np.interp((xs[:, None] + u[None, :]).ravel(), c, m).reshape(len(xs), len(u))
    AM = np.interp((xs[:, None] - u[None, :]).ravel(), c, m).reshape(len(xs), len(u))
    A0 = np.interp(xs, c, m)
    S = np.mean(((AP - AM) / A0[:, None]) ** 2, axis=1)
    return float(xs[int(S.argmin())])


def symmetry(c, m, err, rng):
    b = _scan(c, m)
    if b is None:
        return None
    boot = [v for v in (_scan(c, m + err * rng.standard_normal(len(m)))
                        for _ in range(NBOOT)) if v is not None]
    return b, (float(np.std(boot)) if len(boot) > 20 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--outdir", default="plot")
    a = ap.parse_args()
    rng = np.random.default_rng(7)

    rows = []
    for R in (340, 400, 500):
        for f in sorted(glob.glob(os.path.join(a.base, f"reco_{R}ohm", "*_merged.root"))):
            E = int(re.match(r"^(\d+)_", os.path.basename(f)).group(1))
            arr = uproot.open(f)["h4_reco"].arrays(
                ["run", "A_tot", "pos_eta", "pos_phi", "pos_x", "pos_y",
                 "n_hit_x", "n_hit_y"])
            run = ak.to_numpy(arr["run"]); at = ak.to_numpy(arr["A_tot"])
            et = ak.to_numpy(arr["pos_eta"]) - 18
            ph = ak.to_numpy(arr["pos_phi"]) - 6
            px = ak.to_numpy(arr["pos_x"]).astype(float)
            py = ak.to_numpy(arr["pos_y"]).astype(float)
            nx = ak.to_numpy(arr["n_hit_x"]).astype(float)
            ny = ak.to_numpy(arr["n_hit_y"]).astype(float)
            nom = SCALE[R] * E
            cc = at[(at > .5 * nom) & (at < 1.5 * nom)]
            if len(cc) < 1000:
                continue
            pk = np.median(cc); sg = 1.4826 * np.median(np.abs(cc - pk))
            good = (at > pk - 10 * sg) & (at < pk + 10 * sg)
            hodo = (np.abs(px) < 200) & (np.abs(py) < 200) & (nx > 0) & (ny > 0)
            for r in np.unique(run):
                sel = good & (run == r)
                for dirn, x, cut, hv in (("eta", et, ph, px), ("phi", ph, et, py)):
                    m = sel & (np.abs(cut) < 0.2)
                    if m.sum() < MINEV:
                        continue
                    c_, m_, e_ = profile(x[m], at[m])
                    if len(c_) < 12:
                        continue
                    s = symmetry(c_, m_, e_, rng)
                    if s is None:
                        continue
                    hm = sel & hodo
                    if hm.sum() < 500:
                        continue
                    v = hv[hm]
                    b = np.median(v) / MM_PER_XTAL
                    w = 1.4826 * np.median(np.abs(v - np.median(v))) / MM_PER_XTAL
                    rows.append(dict(R=R, E=E, run=int(r), dir=dirn, nev=int(m.sum()),
                                     xc=s[0], exc=s[1], b=b, w=w))
            print(f"  {R} ohm {E:4d} GeV -> {sum(1 for x in rows if x['R']==R and x['E']==E)} punti",
                  flush=True)

    with open(os.path.join(a.outdir, "symmetry_per_run.csv"), "w") as fh:
        fh.write("resistance,energy,run,direction,n_events,x_c,err_x_c,"
                 "beam_centre_xtal,beam_width_xtal\n")
        for r in rows:
            fh.write(f"{r['R']},{r['E']},{r['run']},{r['dir']},{r['nev']},"
                     f"{r['xc']:+.5f},{r['exc']:.5f},{r['b']:+.5f},{r['w']:.5f}\n")

    # ---- analisi: dentro l'energia (forma fascio fissa) vs tra energie
    print()
    for dirn in ("eta", "phi"):
        print(f"=== {dirn}")
        withins, slopes, means = [], [], []
        for R in (340, 400, 500):
            for E in sorted({r["E"] for r in rows if r["R"] == R}):
                s = [r for r in rows if r["R"] == R and r["E"] == E and r["dir"] == dirn]
                if len(s) < 4:
                    continue
                xc = np.array([r["xc"] for r in s]); b = np.array([r["b"] for r in s])
                sl = np.polyfit(b, xc, 1)[0] if b.std() > 1e-4 else np.nan
                cr = np.corrcoef(b, xc)[0, 1] if b.std() > 1e-4 else np.nan
                withins.append(xc.std()); slopes.append(sl); means.append(xc.mean())
                print(f"   {R} ohm {E:4d} GeV  {len(s):2d} run | x_c medio {xc.mean():+.4f} "
                      f"RMS dentro l'energia {xc.std():.4f} | fascio si sposta di "
                      f"{b.max()-b.min():.4f} xtal, corr(b,x_c) {cr:+.2f}, pendenza {sl:+.3f}")
        if withins:
            print(f"   --> RMS TIPICA DENTRO un'energia : {np.median(withins):.4f}")
            print(f"   --> RMS DEI MEDI TRA le energie  : {np.std(means):.4f}")
            sl = np.array(slopes); sl = sl[np.isfinite(sl)]
            print(f"   --> pendenza x_c vs posizione fascio, mediana su tutte le energie: "
                  f"{np.median(sl):+.3f} (MAD {1.4826*np.median(np.abs(sl-np.median(sl))):.3f})")

    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, dirn in zip(axs, ("eta", "phi")):
        for R in (340, 400, 500):
            for E in sorted({r["E"] for r in rows if r["R"] == R}):
                s = [r for r in rows if r["R"] == R and r["E"] == E and r["dir"] == dirn]
                if len(s) < 4:
                    continue
                b = np.array([r["b"] for r in s]) - np.mean([r["b"] for r in s])
                xc = np.array([r["xc"] for r in s]) - np.mean([r["xc"] for r in s])
                ax.errorbar(b, xc, yerr=[r["exc"] for r in s], fmt="o", ms=5,
                            color=COL[R], alpha=.7,
                            label=f"{R} $\\Omega$" if E == sorted({r["E"] for r in rows
                                                                   if r["R"] == R})[0] else None)
        ax.axhline(0, color="k", lw=1); ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("beam position - mean of that energy  [crystal units]")
        ax.set_title(f"{dirn}: within each energy the collimators are fixed,\n"
                     "so only the beam POSITION changes", fontsize=10)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    axs[0].set_ylabel("$x_c$ - mean of that energy  [crystal units]")
    fig.suptitle("Symmetry point run by run, at fixed beam shape", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "symmetry_per_run.png"), dpi=150)
    print("\nscritto symmetry_per_run.csv e symmetry_per_run.png")


if __name__ == "__main__":
    main()
