#!/usr/bin/env python3
"""
Is the RESPONSE really asymmetric, or is it the abscissa?
=========================================================
THE QUESTION
  The profile of A_tot against pos_eta is not symmetric. Two candidate causes:
    - the response A(true position) really is asymmetric  -> inter-calibration
      of a lateral crystal, or geometry;
    - the response is symmetric and the distortion lives in the ABSCISSA, because
      pos_eta is the ECAL centroid: it carries the conditioning bias (which goes
      as sigma_x^2 and therefore depends on the beam) and the S-shape of the
      centroid estimator (which is itself distorted by a mis-calibration).

THE TEST
  Measure A_tot against the HODOSCOPE x instead. It is independent of the
  amplitudes and linear in the true position, so neither the conditioning (which
  is reduced ~20x, see below) nor the S-shape apply. Then simply ask whether the
  curve is mirror-symmetric about SOME point -- no need to know where the crystal
  centre is in hodoscope millimetres, and no mapping between the two coordinate
  systems is required (that mapping is NOT affine, which is why comparing the two
  profiles directly would be wrong).

    A_tot vs hodoscope symmetric      -> the response is symmetric
                                         -> the eta distortion is the abscissa
    A_tot vs hodoscope asymmetric     -> the response really is asymmetric
                                         -> calibration or geometry

WHY THE HODOSCOPE X AND NOT THE Y
  Both y planes are degraded (broken fibres): only 44% / 38% of events have a
  single cluster, the two planes correlate at 0.77 and their difference has an
  RMS of 3.6 mm. The x planes are healthy: 85% / 78% single-cluster, correlation
  0.972, difference IQR 0.500 mm -> sigma ~ 0.37 mm, and extrapolating 1 m to the
  crystals gives an impact-point error <= 0.4 mm, against ~1.8 mm for the ECAL
  centroid: sigma^2 smaller by ~20. Requiring only x (not y) keeps 86% of the
  events instead of 39%.

  The other coordinate is cut with the ECAL pos_phi: as a SELECTION that is fine,
  it is only as an ABSCISSA that the centroid is problematic.

ESTIMATOR
  Model independent, no parabola. x_c minimises
      S(x_c) = < [ (A(x_c+u) - A(x_c-u)) / A(x_c) ]^2 >_u ,  equal weights in u
  and the residual sqrt(S_min) is compared with what pure statistics would give,
      S_stat = < 2 err(u)^2 / A^2 >_u
  so the output is an EXCESS asymmetry over the statistical floor.

Usage:  python3 response_symmetry_hodo.py --base <dir with reco_*ohm/> --outdir plot
"""
import argparse, glob, os, re
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALE = {340: 3500 / 150., 400: 1080 / 40., 500: 3340 / 100.}
COL = {340: "C0", 400: "C1", 500: "C2"}
BIN_MM, NMIN = 0.4, 50
UMIN_MM, UMAXES_MM = 1.0, (4.8, 6.5, 8.0)
NBOOT = 150


def profile(x, y, lo, hi, binw):
    nb = max(int((hi - lo) / binw), 6)
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


def scan(c, m, umax, step, umin):
    u = np.arange(umin, umax + 1e-9, step)
    xs = np.arange(c.min() + umax, c.max() - umax + 1e-9, step / 2)
    if len(xs) < 5 or len(u) < 3:
        return None
    AP = np.interp((xs[:, None] + u[None, :]).ravel(), c, m).reshape(len(xs), len(u))
    AM = np.interp((xs[:, None] - u[None, :]).ravel(), c, m).reshape(len(xs), len(u))
    A0 = np.interp(xs, c, m)
    S = np.mean(((AP - AM) / A0[:, None]) ** 2, axis=1)
    k = int(S.argmin())
    return float(xs[k]), float(S[k])


def stat_floor(c, m, err, xc, umax, step, umin):
    u = np.arange(umin, umax + 1e-9, step)
    ep = np.interp(xc + u, c, err); em = np.interp(xc - u, c, err)
    a0 = np.interp(xc, c, m)
    return float(np.mean((ep ** 2 + em ** 2) / a0 ** 2))


def analyse(c, m, err, umax, step, rng, umin):
    b = scan(c, m, umax, step, umin)
    if b is None:
        return None
    xc, S = b
    floor = stat_floor(c, m, err, xc, umax, step, umin)
    boot = [r[0] for r in (scan(c, m + err * rng.standard_normal(len(m)), umax, step, umin)
                           for _ in range(NBOOT)) if r]
    exc = max(S - floor, 0.)
    return dict(xc=xc, exc_xc=float(np.std(boot)) if len(boot) > 20 else np.nan,
                asym=np.sqrt(S), floor=np.sqrt(floor), excess=np.sqrt(exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--outdir", default="plot")
    ap.add_argument("--replot", action="store_true",
                    help="rifa solo la figura leggendo il CSV")
    ap.add_argument("--profiles", action="store_true",
                    help="disegna i profili A_tot vs x odoscopica, uno per energia")
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    if a.profiles:
        draw_profiles(a.base, a.outdir)
        return
    if a.replot:
        _replot(a.outdir)
        print("figura rifatta")
        return

    rows = []
    for R in (340, 400, 500):
        for f in sorted(glob.glob(os.path.join(a.base, f"reco_{R}ohm", "*_merged.root"))):
            E = int(re.match(r"^(\d+)_", os.path.basename(f)).group(1))
            arr = uproot.open(f)["h4_reco"].arrays(
                ["A_tot", "pos_eta", "pos_phi", "pos_x"])
            at = ak.to_numpy(arr["A_tot"])
            et = ak.to_numpy(arr["pos_eta"]) - 18
            ph = ak.to_numpy(arr["pos_phi"]) - 6
            px = ak.to_numpy(arr["pos_x"]).astype(float)
            nom = SCALE[R] * E
            cc = at[(at > .5 * nom) & (at < 1.5 * nom)]
            if len(cc) < 1000:
                continue
            pk = np.median(cc); sg = 1.4826 * np.median(np.abs(cc - pk))
            # SOLO la x dell'odoscopio, nessun requisito sulla y (piani rotti)
            sel = (at > pk - 10 * sg) & (at < pk + 10 * sg) & (np.abs(ph) < 0.2)
            hod = sel & (np.abs(px) < 200)
            if hod.sum() < 8000:
                print(f"  {R} ohm {E:4d} GeV: solo {hod.sum()} eventi con x odoscopica, salto")
                continue
            lo, hi = np.percentile(px[hod], [1, 99])
            ch, mh, eh = profile(px[hod], at[hod], lo, hi, BIN_MM)
            ce, me, ee = profile(et[hod], at[hod], -0.5, 0.5, 0.0125)   # stessi eventi
            out = {}
            for umax in UMAXES_MM:
                r = analyse(ch, mh, eh, umax, BIN_MM, rng, UMIN_MM)
                if r:
                    out[("hodo", umax)] = r
                re_ = analyse(ce, me, ee, umax / 22.0, 0.0125, rng, UMIN_MM / 22.0)  # stesso u in cristalli
                if re_:
                    out[("ecal", umax)] = re_
            for (abs_, umax), r in out.items():
                rows.append(dict(R=R, E=E, absc=abs_, umax=umax, nev=int(hod.sum()), **r))
            k = ("hodo", 4.0); j = ("ecal", 4.0)
            if k in out and j in out:
                print(f"  {R} ohm {E:4d} GeV  n={hod.sum():6d} | asimmetria residua: "
                      f"ODOSCOPIO {100*out[k]['asym']:.3f}% (floor {100*out[k]['floor']:.3f}%, "
                      f"eccesso {100*out[k]['excess']:.3f}%)  |  ECAL "
                      f"{100*out[j]['asym']:.3f}% (floor {100*out[j]['floor']:.3f}%, "
                      f"eccesso {100*out[j]['excess']:.3f}%)", flush=True)

    with open(os.path.join(a.outdir, "response_symmetry_hodo.csv"), "w") as fh:
        fh.write("resistance,energy,abscissa,umax_mm,n_events,x_c,err_x_c,"
                 "residual_asym,stat_floor,excess_asym\n")
        for r in rows:
            fh.write(f"{r['R']},{r['E']},{r['absc']},{r['umax']},{r['nev']},"
                     f"{r['xc']:+.5f},{r['exc_xc']:.5f},{r['asym']:.6f},"
                     f"{r['floor']:.6f},{r['excess']:.6f}\n")

    make_plot(rows, a.outdir)


def make_plot(rows, outdir, umax=4.8):
    """Due righe. Sopra: asimmetria misurata e pavimento statistico, ASSE LINEARE.
    Sotto: il rapporto misurata/pavimento, che e' la lettura diretta -- sotto 1
    significa 'simmetrica entro la statistica'."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for j, (absc, lab) in enumerate((("hodo", "hodoscope x  (independent of the amplitudes)"),
                                     ("ecal", "ECAL centroid pos_eta  (same events)"))):
        ax, ax2 = axs[0][j], axs[1][j]
        for R in (340, 400, 500):
            s = sorted([r for r in rows if r["absc"] == absc and r["R"] == R
                        and r["umax"] == umax], key=lambda r: r["E"])
            if not s:
                continue
            E = [r["E"] for r in s]
            meas = [100 * r["asym"] for r in s]
            floor = [100 * r["floor"] for r in s]
            ax.plot(E, meas, "o-", ms=6, color=COL[R], label=f"{R} $\\Omega$  measured")
            ax.plot(E, floor, "s--", ms=5, mfc="none", color=COL[R], alpha=.75,
                    label=f"{R} $\\Omega$  statistical floor")
            ax2.plot(E, [m / f if f > 0 else np.nan for m, f in zip(meas, floor)],
                     "o-", ms=6, color=COL[R], label=f"{R} $\\Omega$")
        ax.set_ylabel("residual asymmetry  [%]")
        ax.set_title(f"abscissa: {lab}", fontsize=11)
        ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(0, None)
        ax2.axhline(1, color="k", lw=1.6)
        ax2.fill_between([0, 300], 0, 1, color="green", alpha=.08)
        ax2.set_xlim(0, 300)
        rr = [r["asym"] / r["floor"] for r in rows
              if r["absc"] == absc and r["umax"] == umax and r["floor"] > 0]
        ax2.set_ylim(0, max(rr) * 1.1 if rr else 2.2)
        ax2.set_xlabel("Beam energy [GeV]")
        ax2.set_ylabel("measured / statistical floor")
        ax2.set_title("below 1 = symmetric within the available statistics", fontsize=10)
        ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    fig.suptitle("Is the response itself asymmetric?   (half-window "
                 f"$u_{{max}}$ = {umax:.0f} mm, model-independent estimator)\n"
                 "left: abscissa independent of the amplitudes -- right: the ECAL centroid, "
                 "same events", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "response_symmetry_hodo.png"), dpi=150)
    plt.close(fig)


def draw_profiles(base, outdir, umax=8.0):
    """Un pannello per energia: <A_tot> contro la x odoscopica, con sovrapposta la
    sua immagine SPECULARE attorno al punto di simmetria x_c. Se le due curve si
    sovrappongono la risposta e' simmetrica; lo SPOSTAMENTO di x_c rispetto al
    centro del fascio e' un'altra cosa e si legge dalla riga verticale."""
    rng = np.random.default_rng(3)
    for R in (340, 400, 500):
        files = sorted(glob.glob(os.path.join(base, f"reco_{R}ohm", "*_merged.root")))
        info = []
        for f in files:
            E = int(re.match(r"^(\d+)_", os.path.basename(f)).group(1))
            arr = uproot.open(f)["h4_reco"].arrays(["A_tot", "pos_phi", "pos_x"])
            at = ak.to_numpy(arr["A_tot"])
            ph = ak.to_numpy(arr["pos_phi"]) - 6
            px = ak.to_numpy(arr["pos_x"]).astype(float)
            nom = SCALE[R] * E
            cc = at[(at > .5 * nom) & (at < 1.5 * nom)]
            if len(cc) < 1000:
                continue
            pk = np.median(cc); sg = 1.4826 * np.median(np.abs(cc - pk))
            hod = (at > pk - 10 * sg) & (at < pk + 10 * sg) & (np.abs(ph) < 0.2) \
                  & (np.abs(px) < 200)
            if hod.sum() < 8000:
                continue
            lo, hi = np.percentile(px[hod], [1, 99])
            c, m, e = profile(px[hod], at[hod], lo, hi, BIN_MM)
            r = uu = None
            for cand in (umax, 6.0, 4.0):          # fallback se i dati non arrivano
                r = analyse(c, m, e, cand, BIN_MM, rng, UMIN_MM)
                if r:
                    uu = cand
                    break
            if r:
                info.append((E, c, m, e, r, int(hod.sum()), uu))
        if not info:
            continue
        ncol = 4; nrow = int(np.ceil(len(info) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)
        for ax, (E, c, m, e, r, n, uu) in zip(axes.ravel(), info):
            xc = r["xc"]
            a0 = np.interp(xc, c, m)
            ax.errorbar(c, 100 * (m / a0 - 1), yerr=100 * e / a0, fmt="o", ms=3.2, lw=.8,
                        color="C0", label="measured")
            mir = np.interp(2 * xc - c, c, m, left=np.nan, right=np.nan)
            ax.plot(c, 100 * (mir / a0 - 1), "s", ms=3.2, mfc="none", color="C3",
                    label="mirrored about $x_c$")
            ax.axvline(xc, color="k", lw=1.4)
            ax.axvline(np.median(c), color="grey", lw=1, ls=":")
            ax.axvspan(xc - uu, xc + uu, color="orange", alpha=.10)
            ax.set_title(f"{E} GeV   $x_c$={xc:+.2f} mm   $u_{{max}}$={uu:.0f} mm\n"
                         f"asym {100*r['asym']:.3f}%  floor {100*r['floor']:.3f}%  "
                         f"excess {100*r['excess']:.3f}%", fontsize=8.5)
            ax.set_xlabel("hodoscope x [mm]", fontsize=8)
            ax.set_ylabel("$\\langle A_{tot}\\rangle / A(x_c) - 1$  [%]", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(alpha=.3)
            ax.legend(fontsize=6.5)
        for ax in axes.ravel()[len(info):]:
            ax.set_axis_off()
        fig.suptitle(f"$A_{{tot}}$ vs hodoscope x -- {R} $\\Omega$   "
                     f"(black line = symmetry point $x_c$, dotted = centre of the sampled range)\n"
                     f"if the red squares sit on the blue circles the response is symmetric; "
                     f"a shift of $x_c$ is a DISPLACEMENT, not an asymmetry.\n"
                     f"the shaded band is the window actually used for the number quoted",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"profiles_Atot_vs_hodo_{R}ohm.png"), dpi=130)
        plt.close(fig)
        print(f"  scritto profiles_Atot_vs_hodo_{R}ohm.png ({len(info)} energie)")


def _replot(outdir):
    import csv as _csv
    rows = []
    with open(os.path.join(outdir, "response_symmetry_hodo.csv")) as fh:
        for r in _csv.DictReader(fh):
            rows.append(dict(R=int(r["resistance"]), E=int(r["energy"]),
                             absc=r["abscissa"], umax=float(r["umax_mm"]),
                             asym=float(r["residual_asym"]), floor=float(r["stat_floor"]),
                             excess=float(r["excess_asym"])))
    make_plot(rows, outdir)


def _tail(a):
    print("\nscritto response_symmetry_hodo.csv e .png")


if __name__ == "__main__":
    main()
