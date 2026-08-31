#!/usr/bin/env python3
"""
Per-channel response vs centroid, and left/right mirror test
============================================================
WHAT THE DATA CONTAINS
  Only 9 channels are read out: a 3x3 matrix at ieta 17-19, iphi 5-7, in a fixed
  order that is identical for every event. sel_S9 and sel_S25 are all-ones masks,
  so S9 = S25 = A_tot = sum of the 9 channels: there is NO 5x5 in these files.
  Only sel_S1 is a real mask (the seed channel).

WHAT THIS SCRIPT DOES
  1. Event selection: A_tot inside peak +- nsigma, where peak and sigma are the
     median and the MAD-based sigma of A_tot in [0.5, 1.5] x nominal. No position
     cut other than the one on the *other* coordinate (|other| < 0.2), which is
     what makes the profile one-dimensional.
  2. Per-channel profile: for each of the 9 crystals, <A_i> in bins of the
     reconstructed centroid, with the error on the mean (RMS/sqrt(N)), bins of
     0.025 crystal units over +-0.6, at least 20 events per bin.
  3. MIRROR TEST. Call L the low-index crystal and H the high-index one along the
     scanned direction. If the crystal response and the inter-calibration were
     symmetric, the shower seen by L when it sits at -d must equal the shower seen
     by H when it sits at +d:
            ratio(d) = <A_L>(x = -d) / <A_H>(x = +d)   ->   1
     Any departure from 1 is a genuine left/right asymmetry.
     eta: L = (17,6), H = (19,6).   phi: L = (18,5), H = (18,7).
  4. The ratio is evaluated at d = 0.2, 0.3, 0.4 crystal units, averaging the
     profile bins within +-0.025 of the target and propagating the errors.

WHY REPRODUCIBILITY IS THE DISCRIMINANT
  The mirror test is not immune to a mis-calibration by itself, because the x axis
  (the centroid) is built from the same amplitudes being compared: a wrong constant
  moves both the numerator and the abscissa. What separates a detector effect from
  a beam effect is that a calibration constant is a property of the crystal and
  CANNOT change between runs, energies or preamp resistances, while the beam can.

Usage:
  python3 channels_vs_centroid.py --base <dir with reco_*ohm/> --outdir plot [--grids]
"""
import argparse, glob, os, re
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETA0, PHI0 = 18., 6.
HALF, BIN, NMIN = 0.6, 0.025, 20
SEL = 0.2
DVALS = (0.2, 0.3, 0.4)
DWIN = 0.025
SCALE = {340: 3500 / 150., 400: 1080 / 40., 500: 3340 / 100.}
COL = {340: "C0", 400: "C1", 500: "C2"}
PAIRS = {"eta": ((17, 6), (19, 6)), "phi": ((18, 5), (18, 7))}


def profile(x, y):
    nb = int(2 * HALF / BIN)
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


UFIT = np.arange(0.10, 0.36, 0.02)


def fit_axis_and_calib(cL, mL, cH, mH):
    """Separa geometria e calibrazione.

    Se la coppia di cristalli e' simmetrica attorno a un asse x_s e ha un rapporto
    di calibrazione k, allora per ogni u vale
        ln A_L(x_s - u) - ln A_H(x_s + u) = ln k     (costante in u)
    Si cerca l'x_s che rende quella differenza piu' piatta possibile; la costante
    residua e' k, e la RMS residua dice quanto il modello (un asse + una costante)
    descrive i dati. Fatto su u in [0.10, 0.35], dove i profili sono ben misurati:
    oltre 0.4 le code sono povere e il rapporto specchiato diventa inaffidabile.

    NOTA: il rapporto specchiato a x fisso e' ipersensibile al punto di riferimento
    (d ln(ratio)/dx ~ 10 per cristallo), quindi NON va usato da solo.
    """
    best = None
    for xs in np.arange(-0.25, 0.2501, 0.002):
        l = np.interp(xs - UFIT, cL, mL, left=np.nan, right=np.nan)
        h = np.interp(xs + UFIT, cH, mH, left=np.nan, right=np.nan)
        ok = np.isfinite(l) & np.isfinite(h) & (l > 0) & (h > 0)
        if ok.sum() < 8:
            continue
        g = np.log(l[ok]) - np.log(h[ok])
        if best is None or g.std() < best[0]:
            best = (g.std(), xs, float(np.exp(g.mean())), int(ok.sum()))
    return best


def value_at(c, m, err, target):
    """Media pesata dei bin entro +-DWIN dal punto voluto."""
    k = np.abs(c - target) <= DWIN
    if k.sum() == 0 or not np.all(err[k] > 0):
        return np.nan, np.nan
    w = 1 / err[k] ** 2
    return (m[k] * w).sum() / w.sum(), 1 / np.sqrt(w.sum())


def analyse(path, energy, resistance, outdir, nsigma, grids):
    t = uproot.open(path)["h4_reco"]
    arr = t.arrays(["A", "sel_ieta", "sel_iphi", "A_tot", "pos_eta", "pos_phi"])
    ie = ak.to_numpy(arr["sel_ieta"])[0]
    ip = ak.to_numpy(arr["sel_iphi"])[0]
    order = {(int(e), int(p)): k for k, (e, p) in enumerate(zip(ie, ip))}
    A = ak.to_numpy(arr["A"])
    atot = ak.to_numpy(arr["A_tot"])
    eta = ak.to_numpy(arr["pos_eta"]) - ETA0
    phi = ak.to_numpy(arr["pos_phi"]) - PHI0

    nom = SCALE[resistance] * energy
    core = atot[(atot > 0.5 * nom) & (atot < 1.5 * nom)]
    if len(core) < 500:
        return []
    pk = np.median(core)
    sg = 1.4826 * np.median(np.abs(core - pk))
    good = (atot > pk - nsigma * sg) & (atot < pk + nsigma * sg)

    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, (dirn, xv, cutv, xl) in zip(axs, (
            ("eta", eta, phi, "pos_eta - 18"), ("phi", phi, eta, "pos_phi - 6"))):
        (lo, hi) = PAIRS[dirn]
        m = good & (np.abs(cutv) < SEL)
        cL, mL, eL = profile(xv[m], A[m, order[lo]])
        cH, mH, eH = profile(xv[m], A[m, order[hi]])
        ax.errorbar(cL, mL, yerr=eL, fmt="o", ms=3, color="C0",
                    label=f"crystal {lo} at $+x$")
        ax.errorbar(-cH, mH, yerr=eH, fmt="s", ms=3, color="C3",
                    label=f"crystal {hi} mirrored, at $-x$")
        for d in DVALS:
            vL, sL = value_at(cL, mL, eL, -d)
            vH, sH = value_at(cH, mH, eH, +d)
            r = vL / vH if vH else np.nan
            er = (r * np.hypot(sL / vL, sH / vH)) if (vH and vL) else np.nan
            rows.append(dict(resistance=resistance, energy=energy, direction=dirn,
                             d=d, low=vL, err_low=sL, high=vH, err_high=sH,
                             ratio=r, err_ratio=er))
        txt = "\n".join(f"|x|={r['d']}:  ratio = {r['ratio']:.3f} $\\pm$ {r['err_ratio']:.3f}"
                        for r in rows[-len(DVALS):])
        ax.text(.03, .97, txt, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="w", ec="0.6", alpha=.9))
        b = fit_axis_and_calib(cL, mL, cH, mH)
        if b:
            rows.append(dict(resistance=resistance, energy=energy, direction=dirn, d=-1,
                             low=b[1], err_low=0., high=b[2], err_high=b[0],
                             ratio=b[2], err_ratio=b[0]))
            ax.text(.03, .03, f"symmetry axis $x_s$ = {b[1]:+.4f}\n"
                              f"calibration $k$ = {b[2]:.4f}\n"
                              f"residual RMS = {b[0]:.4f}", transform=ax.transAxes,
                    va="bottom", fontsize=9, bbox=dict(fc="#eef", ec="0.6", alpha=.9))
        ax.set_xlabel(f"{xl}  (mirrored for the high-index crystal)  [crystal units]")
        ax.set_ylabel("$\\langle A_i \\rangle$ [ADC]")
        ax.set_title(f"Mirror test in {dirn}", fontsize=11)
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.3); ax.set_xlim(-HALF, HALF)
    fig.suptitle(f"Left/right symmetry of the 3x3 -- {resistance} $\\Omega$, {energy} GeV   "
                 f"(A_tot within peak $\\pm$ {nsigma:.0f}$\\sigma$, |other coord| < {SEL})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"channels_mirror_{energy}GeV_{resistance}ohm.png"), dpi=130)
    plt.close(fig)

    if grids:
        etas, phis = sorted({int(x) for x in ie}), sorted({int(x) for x in ip})
        for dirn, xv, cutv, xl, ol in (("eta", eta, phi, "pos_eta - 18", "pos_phi - 6"),
                                       ("phi", phi, eta, "pos_phi - 6", "pos_eta - 18")):
            m = good & (np.abs(cutv) < SEL)
            f2, ax2 = plt.subplots(len(phis), len(etas), figsize=(4.4 * len(etas), 3.2 * len(phis)),
                                   squeeze=False, sharex=True)
            for r_, pp in enumerate(reversed(phis)):
                for c_, ee in enumerate(etas):
                    ax = ax2[r_][c_]
                    cx, cy, ce = profile(xv[m], A[m, order[(ee, pp)]])
                    ax.errorbar(cx, cy, yerr=ce, fmt="o", ms=2.4, lw=.7, color="C0")
                    for v in (-SEL, SEL):
                        ax.axvline(v, color="k", lw=.8)
                    seed = (ee, pp) == (18, 6)
                    ax.set_title(f"crystal (ieta={ee}, iphi={pp})" + ("   [SEED]" if seed else ""),
                                 fontsize=9, fontweight="bold" if seed else "normal")
                    ax.grid(alpha=.3); ax.set_xlim(-HALF, HALF)
                    if r_ == len(phis) - 1:
                        ax.set_xlabel(f"{xl}  [crystal units]")
                    if c_ == 0:
                        ax.set_ylabel("$\\langle A_i \\rangle$ [ADC]")
            f2.suptitle(f"Per-channel response vs {xl} -- {resistance} $\\Omega$, {energy} GeV\n"
                        f"CUTS: |{ol}| < {SEL} && A_tot within peak $\\pm$ {nsigma:.0f}$\\sigma$"
                        f"   (panels laid out as the 3x3 matrix)", fontsize=12)
            f2.tight_layout()
            f2.savefig(os.path.join(outdir,
                       f"channels_vs_{dirn}_{energy}GeV_{resistance}ohm.png"), dpi=120)
            plt.close(f2)
    return rows


def summary_axis(rows, outdir):
    """x_s e k per ogni dataset: entrambi devono essere costanti se sono
    proprieta' del rivelatore."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, key, lab, ref in ((axs[0], "low", "symmetry axis $x_s$  [crystal units]", 0.),
                              (axs[1], "high", "calibration ratio $k$", 1.)):
        for dirn, mk in (("eta", "o"), ("phi", "s")):
            for R in (340, 400, 500):
                sub = [r for r in rows if r["d"] == -1 and r["direction"] == dirn
                       and r["resistance"] == R]
                sub.sort(key=lambda r: r["energy"])
                if sub:
                    ax.plot([r["energy"] for r in sub], [r[key] for r in sub], mk,
                            ms=6, color=COL[R], mfc="none" if dirn == "phi" else COL[R],
                            label=f"{R} $\\Omega$, {dirn}")
        ax.axhline(ref, color="k", lw=1.2)
        ax.set_xlabel("Beam energy [GeV]")
        ax.set_ylabel(lab)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Symmetry axis and calibration ratio, fitted separately over |u| in [0.10, 0.35]\n"
                 "filled = eta pair (17,6)/(19,6), open = phi pair (18,5)/(18,7)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mirror_axis_and_calibration.png"), dpi=150)
    plt.close(fig)


def summary(rows, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, dirn in zip(axs, ("eta", "phi")):
        for R in (340, 400, 500):
            for d, mk in zip(DVALS, ("o", "s", "^")):
                sub = [r for r in rows if r["resistance"] == R and r["direction"] == dirn
                       and r["d"] == d and np.isfinite(r["ratio"])]
                if not sub:
                    continue
                sub.sort(key=lambda r: r["energy"])
                ax.errorbar([r["energy"] for r in sub], [r["ratio"] for r in sub],
                            yerr=[r["err_ratio"] for r in sub], fmt=mk + "-", ms=5, lw=1,
                            color=COL[R], alpha=0.45 + 0.55 * (d - 0.2) / 0.2,
                            label=f"{R} $\\Omega$, |x|={d}")
        ax.axhline(1, color="k", lw=1.2)
        ax.set_xlabel("Beam energy [GeV]")
        ax.set_title(f"Mirror ratio in {dirn}\n"
                     f"{PAIRS[dirn][0]} at $-d$  /  {PAIRS[dirn][1]} at $+d$", fontsize=11)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=3)
    axs[0].set_ylabel("ratio low-index / high-index  (1 = symmetric)")
    fig.suptitle("Left/right asymmetry of the 3x3, all energies and resistances\n"
                 "a detector effect must be the same everywhere; a beam effect need not be",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mirror_asymmetry_summary.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--outdir", default="plot")
    ap.add_argument("--nsigma", type=float, default=10.)
    ap.add_argument("--grids", action="store_true",
                    help="also write the 3x3 per-channel grids (2 more figures per energy)")
    a = ap.parse_args()

    allrows = []
    for R in (340, 400, 500):
        out = os.path.join(a.outdir, str(R))
        os.makedirs(out, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(a.base, f"reco_{R}ohm", "*_merged.root"))):
            m = re.match(r"^(\d+)_", os.path.basename(f))
            if not m:
                continue
            E = int(m.group(1))
            print(f"  {R} ohm  {E:4d} GeV", flush=True)
            allrows += analyse(f, E, R, out, a.nsigma, a.grids)

    csv = os.path.join(a.outdir, "mirror_asymmetry.csv")
    with open(csv, "w") as fh:
        fh.write("resistance,energy,direction,d,A_low,err_low,A_high,err_high,ratio,err_ratio\n")
        for r in allrows:
            fh.write(f"{r['resistance']},{r['energy']},{r['direction']},{r['d']},"
                     f"{r['low']:.4f},{r['err_low']:.4f},{r['high']:.4f},{r['err_high']:.4f},"
                     f"{r['ratio']:.5f},{r['err_ratio']:.5f}\n")
    summary(allrows, a.outdir)
    summary_axis(allrows, a.outdir)
    with open(os.path.join(a.outdir, "mirror_axis_calib.csv"), "w") as fh:
        fh.write("resistance,energy,direction,symmetry_axis,calibration_k,residual_rms\n")
        for r in allrows:
            if r["d"] == -1:
                fh.write(f"{r['resistance']},{r['energy']},{r['direction']},"
                         f"{r['low']:+.5f},{r['high']:.5f},{r['err_high']:.5f}\n")
    print("scritto", csv, "e mirror_asymmetry_summary.png")


if __name__ == "__main__":
    main()
