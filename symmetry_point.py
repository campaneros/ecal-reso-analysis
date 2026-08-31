#!/usr/bin/env python3
"""
Model-independent symmetry point of the response
================================================
WHY NOT THE PARABOLA
  The response is NOT parabolic: the parabola fit gives chi2/ndf between 3 and 6,
  and its curvature p2 changes by ~18% just by moving the fit range from +-0.3 to
  +-0.4. Two consequences for the vertex p0 used before:
    (a) it depends on the chosen range;
    (b) the chi2 weights are 1/(RMS/sqrt(N))^2, so bins where the beam is dense
        dominate the fit. A symmetric model forced onto an asymmetric curve then
        centres itself where the beam is -- which puts the occupancy back in,
        with the OPPOSITE sign to the conditioning bias. The measured slope is
        therefore a mixture and the extracted sigma_x is not trustworthy.

WHAT THIS DOES INSTEAD
  No shape model at all. The question is only "is there a point about which the
  curve is mirror-symmetric?", so we look for the x_c that minimises

        S(x_c) = < [ (A(x_c+u) - A(x_c-u)) / A(x_c) ]^2 >_u        u in [UMIN, umax]

  with EQUAL weights in u, so the occupancy cannot come back in through the
  weighting. sqrt(S) at the minimum is the residual relative asymmetry: if it is
  large, no symmetry centre exists and the question itself has to be reframed.

  Everything is repeated for several umax, because the stability of x_c against
  the window is the check that was missing with the parabola.

  Uncertainty on x_c: bootstrap, resampling every profile point within its own
  statistical error.

OUTPUTS
  symmetry_point.csv           x_c, its error and the residual asymmetry, for
                               A_tot and for the seed, both directions, each umax
  symmetry_stability.png       x_c vs umax -- the missing check
  symmetry_beam.png            decomposition into fixed + beam-driven, using x_c
  symmetry_seed_vs_total.png   geometry vs lateral crystals, using x_c

Usage:  python3 symmetry_point.py --base <dir with reco_*ohm/> --outdir plot
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
HALF, BINW, NMIN = 0.6, 0.0125, 20
UMIN = 0.05
UMAXES = (0.15, 0.20, 0.25, 0.30)
XCGRID = np.arange(-0.20, 0.2001, 0.002)
NBOOT = 200


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


def _scan(c, m, umax):
    """Ritorna (x_c, asimmetria residua) minimizzando S(x_c). Vettorizzato."""
    u = np.arange(UMIN, umax + 1e-9, BINW)
    xs = XCGRID[(XCGRID - umax >= c.min()) & (XCGRID + umax <= c.max())]
    if len(xs) < 5 or len(u) < 4:
        return None
    AP = np.interp((xs[:, None] + u[None, :]).ravel(), c, m).reshape(len(xs), len(u))
    AM = np.interp((xs[:, None] - u[None, :]).ravel(), c, m).reshape(len(xs), len(u))
    A0 = np.interp(xs, c, m)
    S = np.mean(((AP - AM) / A0[:, None]) ** 2, axis=1)
    k = int(S.argmin())
    return float(xs[k]), float(np.sqrt(S[k]))


def symmetry_point(c, m, err, umax, rng):
    base = _scan(c, m, umax)
    if base is None:
        return None
    boot = []
    for _ in range(NBOOT):
        r = _scan(c, m + err * rng.standard_normal(len(m)), umax)
        if r:
            boot.append(r[0])
    e = float(np.std(boot)) if len(boot) > 20 else np.nan
    return base[0], e, base[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--outdir", default="plot")
    a = ap.parse_args()
    rng = np.random.default_rng(12345)

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
            cc = at[(at > .5 * nom) & (at < 1.5 * nom)]
            if len(cc) < 1000:
                continue
            pk = np.median(cc); sg = 1.4826 * np.median(np.abs(cc - pk))
            good = (at > pk - 10 * sg) & (at < pk + 10 * sg)
            hodo = good & (np.abs(px) < 200) & (np.abs(py) < 200) & (nx > 0) & (ny > 0)
            if hodo.sum() < 3000:
                continue
            for dirn, x, cut, hv in (("eta", et, ph, px), ("phi", ph, et, py)):
                m = good & (np.abs(cut) < 0.2)
                v = hv[hodo]
                b = np.median(v) / MM_PER_XTAL
                w = 1.4826 * np.median(np.abs(v - np.median(v))) / MM_PER_XTAL
                for obs, yv in (("A_tot", at), ("seed", A[:, o[(18, 6)]])):
                    c_, m_, e_ = profile(x[m], yv[m])
                    if len(c_) < 20:
                        continue
                    for umax in UMAXES:
                        r = symmetry_point(c_, m_, e_, umax, rng)
                        if r:
                            rows.append(dict(R=R, E=E, dir=dirn, obs=obs, umax=umax,
                                             xc=r[0], exc=r[1], asym=r[2], b=b, w=w))

    with open(os.path.join(a.outdir, "symmetry_point.csv"), "w") as fh:
        fh.write("resistance,energy,direction,observable,umax,x_c,err_x_c,"
                 "residual_asymmetry,beam_centre_xtal,beam_width_xtal\n")
        for r in rows:
            fh.write(f"{r['R']},{r['E']},{r['dir']},{r['obs']},{r['umax']},"
                     f"{r['xc']:+.5f},{r['exc']:.5f},{r['asym']:.5f},"
                     f"{r['b']:+.5f},{r['w']:.5f}\n")

    # ---------------- stabilita' rispetto alla finestra
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for ax, dirn in zip(axs, ("eta", "phi")):
        for R in (340, 400, 500):
            for E in sorted({r["E"] for r in rows if r["R"] == R}):
                s = sorted([r for r in rows if r["R"] == R and r["E"] == E
                            and r["dir"] == dirn and r["obs"] == "A_tot"],
                           key=lambda r: r["umax"])
                if len(s) >= 2:
                    ax.plot([r["umax"] for r in s], [r["xc"] for r in s], "-o", ms=3,
                            lw=.8, color=COL[R], alpha=.6)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("half-window $u_{max}$  [crystal units]")
        ax.set_title(f"{dirn}: symmetry point vs window\n"
                     "one line per dataset -- flat = a symmetry centre exists", fontsize=10)
        ax.grid(alpha=.3)
    axs[0].set_ylabel("$x_c$  [crystal units]")
    fig.suptitle("Stability of the model-independent symmetry point (the check the parabola never had)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "symmetry_stability.png"), dpi=150)
    plt.close(fig)

    # ---------------- decomposizione fascio / fisso, a umax = 0.25
    UREF = 0.25
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    summary = {}
    for ax, dirn in zip(axs, ("eta", "phi")):
        s = [r for r in rows if r["dir"] == dirn and r["obs"] == "A_tot" and r["umax"] == UREF]
        if len(s) < 6:
            continue
        y = np.array([r["xc"] for r in s]); ey = np.array([r["exc"] for r in s])
        b = np.array([r["b"] for r in s]); w = np.array([r["w"] for r in s])
        md = np.median(y); mad = 1.4826 * np.median(np.abs(y - md))
        k = np.abs(y - md) < 3 * mad
        X = (-b / w ** 2)[k]
        sl, ic = np.polyfit(X, y[k], 1)
        res = y[k] - (sl * X + ic)
        eic = res.std() * np.sqrt(1 / k.sum() + X.mean() ** 2 / ((X - X.mean()) ** 2).sum())
        summary[dirn] = (ic, eic, sl, res.std(), y[k].std(), k.sum())
        for R in (340, 400, 500):
            kk = np.array([r["R"] == R for r in s])[k]
            if kk.any():
                ax.errorbar(X[kk], y[k][kk], yerr=ey[k][kk], fmt="o", ms=6,
                            color=COL[R], label=f"{R} $\\Omega$")
        xx = np.linspace(X.min(), X.max(), 10)
        ax.plot(xx, sl * xx + ic, "r-", lw=1.6)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("$-b/w^2$  (beam, from the hodoscope)  [1/crystal]")
        ax.set_ylabel("symmetry point $x_c$  [crystal units]")
        ax.set_title(f"{dirn}:  fixed part = {ic:+.4f} $\\pm$ {eic:.4f} ({ic/eic:+.1f}$\\sigma$)\n"
                     f"slope = {sl:+.5f}   ($u_{{max}}$ = {UREF})", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Fixed vs beam-driven, using the model-independent symmetry point", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "symmetry_beam.png"), dpi=150)
    plt.close(fig)

    # ---------------- seed contro somma
    fig, ax = plt.subplots(figsize=(9, 6))
    print()
    for dirn, mk in (("eta", "o"), ("phi", "s")):
        d = []
        for R in (340, 400, 500):
            for E in sorted({r["E"] for r in rows if r["R"] == R}):
                t = [r for r in rows if r["R"] == R and r["E"] == E and r["dir"] == dirn
                     and r["umax"] == UREF]
                a_ = [r for r in t if r["obs"] == "A_tot"]
                s_ = [r for r in t if r["obs"] == "seed"]
                if a_ and s_:
                    d.append((E, a_[0]["xc"] - s_[0]["xc"], R,
                              np.hypot(a_[0]["exc"], s_[0]["exc"])))
        if not d:
            continue
        for R in (340, 400, 500):
            p = [x for x in d if x[2] == R]
            if p:
                ax.errorbar([x[0] for x in p], [x[1] for x in p], yerr=[x[3] for x in p],
                            fmt=mk, ms=6, color=COL[R],
                            mfc="none" if dirn == "phi" else COL[R],
                            label=f"{R} $\\Omega$, {dirn}")
        v = np.array([x[1] for x in d])
        md = np.median(v); mad = 1.4826 * np.median(np.abs(v - md))
        print(f"{dirn}: x_c(A_tot) - x_c(seed) = {md:+.4f} +- {mad/np.sqrt(len(v)):.4f}"
              f"   (N={len(v)})")
        if dirn in summary:
            ic, eic, sl, rr, r0, n = summary[dirn]
            print(f"      parte fissa {ic:+.4f} +- {eic:.4f} ({ic/eic:+.1f} sigma), "
                  f"pendenza {sl:+.5f}, residuo RMS {rr:.4f} su {r0:.4f}, N={n}")
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xlabel("Beam energy [GeV]")
    ax.set_ylabel("$x_c(A_{tot}) - x_c(\\mathrm{seed})$  [crystal units]")
    ax.set_title("Geometry moves both; a lateral-crystal asymmetry moves only the sum\n"
                 f"model-independent symmetry point, $u_{{max}}$ = {UREF}", fontsize=11)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "symmetry_seed_vs_total.png"), dpi=150)
    print("\nscritto symmetry_point.csv e le tre figure")


if __name__ == "__main__":
    main()
