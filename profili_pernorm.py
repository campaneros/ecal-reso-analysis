"""
<A_tot> vs centroid profiles, redone by normalising every event to the peak of its
own run.

Why. After the re-merge many files contain several runs, and at 340 ohm the runs
split into two response populations separated by a few percent (the size can be read
off the per-run peaks written by drift_dcb_all.py). If the runs are centred at
different points of the crystal, the mean <A_tot>(eta) is no longer a response curve
but the composition of the runs: where the higher run carries more weight the mean
goes up, and a ramp appears instead of the parabola -- the curvature can come out
with the wrong sign and with a chi2/ndf orders of magnitude too large.

Remedy: A_tot -> A_tot / peak_DCB(run), then rescaled to the average peak so that
the axis stays in ADC. The runs collapse onto each other and the parabola comes
back. The CSV produced also reports, next to every fit, the curvature one would get
without normalising, so the effect is quantified point by point rather than in
words.

Fit: quadratic  a + b x + c x^2  over |x| <= 0.3, LINEAR least squares.
NOT the form p1 + p2 (x - p0)^2: that one has degenerate vertex and curvature -- a
shift of the vertex is compensated by a change of curvature -- and on this data it
runs away, returning vertices several crystals away from the fit window. The
curvature is read as c/a, in % per crystal^2.

A_tot window: peak +- NSIG sigma per run, with peak and sigma taken from the
uniformita_pos.py cache (plot/uniformita/_cache/<R>_<E>.json). If the cache is
missing the run is skipped.

Usage:
  python3 plot/profili_pernorm.py --base . --outdir plot/profili \
      --resistances 340 --energies 20 30 40 60 --exclude-runs 20592 ...
  python3 plot/profili_pernorm.py --only-collect --outdir plot/profili
"""

import argparse, os, glob, re, json
import runsets
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

ETA0, PHI0, SEL = 18., 6., 0.2
HALF, NB, FIT_HALF, NMIN = 0.6, 96, 0.3, 20
A_TOT_MIN, NSIG = 100., 10.

FILES = {340: ("reco_340ohm", "*_merged.root"),
         400: ("reco_400ohm", "*_400_merged.root"),
         500: ("reco_500ohm", "*_500_merged.root")}

COORD = (("pos_eta", "pos\\_eta - 18"), ("pos_phi", "pos\\_phi - 6"))


def profile(x, y):
    H, e = np.histogram(x, bins=NB, range=(-HALF, HALF))
    S, _ = np.histogram(x, bins=NB, range=(-HALF, HALF), weights=y)
    S2, _ = np.histogram(x, bins=NB, range=(-HALF, HALF), weights=y ** 2)
    c = 0.5 * (e[:-1] + e[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = S / np.maximum(H, 1)
        var = S2 / np.maximum(H, 1) - m ** 2
        er = np.sqrt(np.maximum(var, 0) / np.maximum(H, 1))
    ok = (H >= NMIN) & (er > 0)
    return c[ok], m[ok], er[ok]


def quad_fit(c, m, er, half=FIT_HALF):
    """a + b x + c x^2, minimi quadrati lineari pesati."""
    k = np.abs(c) <= half
    if k.sum() < 8:
        return None
    X = np.column_stack([np.ones(k.sum()), c[k], c[k] ** 2]) / er[k][:, None]
    y = m[k] / er[k]
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    cov = np.linalg.inv(X.T @ X)
    res = y - X @ co
    a, cc = co[0], co[2]
    return dict(co=[float(v) for v in co],
                err=[float(v) for v in np.sqrt(np.diag(cov))],
                chi2=float((res ** 2).sum()), ndf=int(k.sum()) - 3,
                rel=float(100 * cc / a),
                err_rel=float(100 * np.sqrt(cov[2, 2]) / abs(a)))


def load_cache(cachedir, R, E):
    f = os.path.join(cachedir, f"{R}_{E}.json")
    if not os.path.exists(f):
        return None
    c = json.load(open(f))
    out = {}
    for p in c["per_run"]:
        if "raw" in p and p.get("peak", 0) > 0:
            out[int(p["run"])] = (float(p["peak"]), float(p["peak"]) * float(p["raw"][0]) / 100.)
    return out or None


def analyse(path, E, R, drop, cachedir, outdir, only=()):
    per = load_cache(cachedir, R, E)
    if per is None:
        print(f"      manca la cache {cachedir}/{R}_{E}.json, salto")
        return None
    t = uproot.open(path)["h4_reco"]
    arr = t.arrays(["run", "A_tot", "pos_eta", "pos_phi"], library="np")
    run = arr["run"]
    at = arr["A_tot"].astype(float)
    u = arr["pos_eta"] - ETA0
    v = arr["pos_phi"] - PHI0

    keep = (at > A_TOT_MIN) & np.isin(run, list(per))
    if drop:
        keep &= ~np.isin(run, drop)
    if len(only):
        keep &= np.isin(run, only)
    runs = sorted(int(r) for r in np.unique(run[keep]))
    if not runs:
        return None

    pk = np.ones(len(at))
    sg = np.ones(len(at))
    for r in runs:
        m = run == r
        pk[m], sg[m] = per[r]
    inwin = keep & (np.abs(at - pk) < NSIG * sg)
    nev = np.array([int((inwin & (run == r)).sum()) for r in runs], float)
    ref = float((nev * [per[r][0] for r in runs]).sum() / nev.sum())
    y = at / pk * ref

    rows, fits = [], {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    cmap = cm.viridis(np.linspace(0.05, 0.9, max(len(runs), 1)))
    for j, (name, lab) in enumerate(COORD):
        x = u if j == 0 else v
        oth = v if j == 0 else u
        base = inwin & (np.abs(oth) <= SEL)
        ax = axes[j]
        for i, r in enumerate(runs):
            b = base & (run == r)
            if b.sum() < 200:
                continue
            cc, mm, ee = profile(x[b], y[b])
            ax.plot(cc, mm, "-", lw=0.9, alpha=.75, color=cmap[i], label=f"run {r}")
        c, m, er = profile(x[base], y[base])
        ax.errorbar(c, m, yerr=er, fmt="o", ms=3.2, lw=.9, color="C0",
                    label="all runs", zorder=4)
        f = quad_fit(c, m, er)
        graw = quad_fit(*profile(x[base], at[base]))
        if f:
            xs = np.linspace(-FIT_HALF, FIT_HALF, 200)
            ax.plot(xs, f["co"][0] + f["co"][1] * xs + f["co"][2] * xs ** 2,
                    "r-", lw=2.2, zorder=5)
            txt = (f"$c$ = {f['co'][2]:.1f} $\\pm$ {f['err'][2]:.1f} ADC/cr$^2$   "
                   f"$c/a$ = {f['rel']:.2f} $\\pm$ {f['err_rel']:.2f} %/cr$^2$   "
                   f"$\\chi^2$/ndf {f['chi2']:.1f} / {f['ndf']}")
            if graw and len(runs) > 1:
                txt += (f"\nsenza normalizzazione:   $c/a$ = {graw['rel']:.2f} %/cr$^2$   "
                        f"$\\chi^2$/ndf {graw['chi2']:.1f} / {graw['ndf']}")
            ax.text(.5, .03, txt, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, bbox=dict(fc="white", ec="0.5"))
            fits[name] = f
            rows.append([R, E, name, len(runs), int(base.sum())]
                        + f["co"] + f["err"] + [f["rel"], f["err_rel"], f["chi2"], f["ndf"]]
                        + ([graw["rel"], graw["chi2"], graw["ndf"]] if graw else [np.nan]*3))
        for s in (-SEL, SEL):
            ax.axvline(s, color="k", lw=1)
        ax.set_xlabel(f"${lab}$  [crystal units]")
        ax.set_ylabel("$\\langle A_{tot} / \\mathrm{peak}(A_{tot})_{run} \\rangle "
                      "\\times$ ref  [ADC]")
        ax.set_title(("$|\\mathrm{pos\\_phi}-6| \\leq 0.2$" if j == 0
                      else "$|\\mathrm{pos\\_eta}-18| \\leq 0.2$")
                     + f",   $|A_{{tot}} - \\mu_{{run}}| < {NSIG:.0f}\\sigma_{{run}}$",
                     fontsize=10)
        if len(runs) <= 8:
            ax.legend(fontsize=8, ncol=1 + len(runs) // 5)
        ax.grid(alpha=.3)
    fig.suptitle(f"{R} $\\Omega$,  {E} GeV   $\\quad$   $A_{{tot}} > {A_TOT_MIN:.0f}$ ADC,  "
                 f"$A_{{tot}} \\to A_{{tot}} / \\mathrm{{peak}}(A_{{tot}})_{{run}}$   "
                 f"$\\quad$   {len(runs)} runs,  $N$ = {int(inwin.sum())}", fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, f"profilo_{E}GeV_{R}ohm.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    return dict(rows=rows, png=p, runs=runs, nev=int(inwin.sum()), ref=ref, fits=fits)


COLS = ("resistance,energy,coord,nrun,nev,a,b,c,err_a,err_b,err_c,"
        "rel_pct,err_rel_pct,chi2,ndf,rel_pct_raw,chi2_raw,ndf_raw")


def collect_plots(outdir, resistances):
    """All the energies of one resistance on the same plot, normalised to a, plus the
    relative curvature as a function of energy."""
    rows = []
    for l in open(os.path.join(outdir, "profili_pernorm.csv")).read().splitlines()[1:]:
        p = l.split(",")
        rows.append(dict(zip(COLS.split(","),
                             [int(p[0]), int(p[1]), p[2], int(p[3]), int(p[4])]
                             + [float(v) for v in p[5:15]] + [float(v) for v in p[15:]])))
    figc, axc = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for j, (name, lab) in enumerate(COORD):
        for R, mk in ((340, "o-"), (400, "s-"), (500, "^-")):
            rr = sorted([r for r in rows if r["resistance"] == R and r["coord"] == name],
                        key=lambda r: r["energy"])
            if not rr:
                continue
            axc[j].errorbar([r["energy"] for r in rr], [-r["rel_pct"] for r in rr],
                            yerr=[r["err_rel_pct"] for r in rr], fmt=mk, ms=6, capsize=3,
                            label=f"{R} $\\Omega$")
        axc[j].set_xscale("log")
        axc[j].set_xlabel("$E_{nom}$ [GeV]")
        axc[j].set_ylabel("$-c/a$  [% / crystal$^2$]")
        axc[j].set_title(f"${lab}$", fontsize=11)
        axc[j].grid(alpha=.3, which="both"); axc[j].legend(fontsize=9)
    figc.suptitle(f"$A_{{tot}} > {A_TOT_MIN:.0f}$ ADC,  "
                  f"$|A_{{tot}} - \\mu_{{run}}| < {NSIG:.0f}\\sigma_{{run}}$,  "
                  f"fit $a + bx + cx^2$,  $|x| \\leq {FIT_HALF}$", fontsize=11)
    figc.tight_layout()
    p = os.path.join(outdir, "curvatura_vs_energia.png")
    figc.savefig(p, dpi=150); plt.close(figc)
    print("->", p)

    for R in resistances:
        rr = [r for r in rows if r["resistance"] == R]
        if not rr:
            continue
        Es = sorted({r["energy"] for r in rr})
        norm = mcolors.Normalize(vmin=min(Es), vmax=max(Es))
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
        for j, (name, lab) in enumerate(COORD):
            for E in Es:
                q = [r for r in rr if r["coord"] == name and r["energy"] == E]
                if not q:
                    continue
                q = q[0]
                xs = np.linspace(-FIT_HALF, FIT_HALF, 200)
                yy = (q["a"] + q["b"] * xs + q["c"] * xs ** 2) / q["a"]
                axes[j].plot(xs, yy, "-", lw=1.6, color=cm.viridis(norm(E)))
            axes[j].axvline(-SEL, color="k", lw=1); axes[j].axvline(SEL, color="k", lw=1)
            axes[j].set_xlabel(f"${lab}$  [crystal units]")
            axes[j].set_ylabel("$(a + bx + cx^2) / a$")
            axes[j].grid(alpha=.3)
            axes[j].set_title(("$|\\mathrm{pos\\_phi}-6| \\leq 0.2$" if j == 0
                               else "$|\\mathrm{pos\\_eta}-18| \\leq 0.2$"), fontsize=10)
        sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis); sm.set_array([])
        fig.colorbar(sm, ax=axes, label="$E_{nom}$ [GeV]")
        fig.suptitle(f"{R} $\\Omega$   $\\quad$   $A_{{tot}} > {A_TOT_MIN:.0f}$ ADC,  "
                     f"$A_{{tot}} \\to A_{{tot}} / \\mathrm{{peak}}(A_{{tot}})_{{run}}$",
                     fontsize=11)
        p = os.path.join(outdir, f"profili_all_energies_{R}ohm.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("->", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--outdir", default="plot/profili")
    ap.add_argument("--cache", default="plot/uniformita/_cache")
    ap.add_argument("--resistances", nargs="+", type=int, default=[340, 400, 500])
    ap.add_argument("--energies", nargs="*", type=int, default=None)
    ap.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    runsets.add_argument(ap)
    ap.add_argument("--only-collect", action="store_true")
    a = ap.parse_args()
    drop, only = runsets.resolve(a.runset, a.exclude_runs)
    os.makedirs(a.outdir, exist_ok=True)
    rowdir = os.path.join(a.outdir, "_rows"); os.makedirs(rowdir, exist_ok=True)

    if not a.only_collect:
        for R in a.resistances:
            d, pat = FILES[R]
            files = sorted(glob.glob(os.path.join(a.base, d, pat)),
                           key=lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1)))
            for f in files:
                E = int(re.match(r"(\d+)", os.path.basename(f)).group(1))
                if a.energies is not None and E not in a.energies:
                    continue
                print(f"  {R} ohm {E:4d} GeV", flush=True)
                res = analyse(f, E, R, drop, a.cache, a.outdir, only)
                if res is None:
                    continue
                json.dump(res["rows"], open(os.path.join(rowdir, f"{R}_{E}.json"), "w"),
                          default=float)
                for nm in ("pos_eta", "pos_phi"):
                    if nm in res["fits"]:
                        q = res["fits"][nm]
                        print(f"      {nm}: c/a = {q['rel']:+.2f} +- {q['err_rel']:.2f} %/cr^2"
                              f"   chi2/ndf {q['chi2']:.1f}/{q['ndf']}", flush=True)

    rows = []
    for f in sorted(glob.glob(os.path.join(rowdir, "*.json"))):
        rows += json.load(open(f))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    csv = os.path.join(a.outdir, "profili_pernorm.csv")
    with open(csv, "w") as fh:
        fh.write(COLS + "\n")
        for r in rows:
            fh.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v)
                              for v in r) + "\n")
    print("->", csv, f"({len(rows)} righe)")
    if rows:
        collect_plots(a.outdir, a.resistances)


if __name__ == "__main__":
    main()
