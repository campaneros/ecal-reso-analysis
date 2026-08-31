# Energy resolution analysis — ECAL test beam H4, 2026

Scripts to measure the energy resolution `σ/E = N/E ⊕ S/√E ⊕ C` of an ECAL module
exposed to an electron beam in the H4 line at the SPS, with particular attention to
the **non-uniformity of the response across the impact point**.

This repository contains **code only**. Reconstructed data, intermediate CSV files
and the plots produced are kept out (see `.gitignore`).

---

## Expected inputs

The scripts expect to run from a working directory containing:

```
reco_340ohm/<E>_merged.root          13 energies
reco_400ohm/<E>_400_merged.root       9 energies
reco_500ohm/<E>_500_merged.root       7 energies
bes/rereco_<R>_withBES.csv           beam energy spread per (resistance, energy)
```

The merged files hold the `h4_reco` tree. The variables used are `run`, `spill`,
`evt`, `A_tot` (sum of the amplitudes over the 3×3 matrix, in ADC counts),
`pos_eta` and `pos_phi` (hodoscope centroid, in crystal units).

The three directories correspond to three values of the CATIA preamplifier feedback
resistance — 340, 400 and 500 Ω — that is, to three different gains.

Most scripts take `--base` for the data directory and `--outdir`/`--plotdir` for the
outputs, so the repository can live anywhere relative to the data.

## Dependencies

Python 3 with `numpy`, `uproot`, `iminuit`, `matplotlib`. The macros under `root/`
require ROOT, and they are the only part that does: everything else reads the
`.root` files through `uproot` and needs no ROOT installation.

```bash
pip3 install --user numpy uproot iminuit matplotlib
```

---

## Main chain

Order matters: each step reads the outputs of the previous one.

| # | script | what it does |
|---|---|---|
| 0 | `runsets.py` | Not a script: the definition of which runs belong to which readout condition, imported by everything that reads runs. See **Run conditions** below. |
| 1 | `drift_dcb_all.py` | Base engine. Double-Crystal-Ball fit of `A_tot` **run by run**, with the iterative mean ± 3 RMS window and Freedman-Diaconis binning. Writes peak and σ per run, the drift systematic, the 2D maps and the profiles. |
| 2 | `uniformita_pos.py` | Non-uniformity correction. Estimates the quadratic response surface in (η, φ) and compares four treatments: none, event-by-event correction, quadrature subtraction, and reweighting to flat illumination. |
| 3 | `uniformita_maps.py` | Three estimates of the same surface — per run, per energy, averaged per resistance — which yield the **centroid systematic**. Three stages: `--stage moments`, `--stage apply`, `--stage collect`. |
| 4 | `resolution_final_uniforme.py` | Final curve. Subtracts BES, synchrotron and the position term in quadrature, then fits `N ⊕ S ⊕ C` per resistance and simultaneously with S and C in common. |
| 5 | `fit_root_all.py` + `root/fit_resolution.C` | Repeats the same fits in ROOT and stores TGraphErrors, TF1, TCanvas and a summary TTree in a `.root` file. |

Every step takes `--runset {standard,filter50,all}`, which defaults to `standard`.

```bash
# 1
python3 drift_dcb_all.py --base <data> --outdir plot

# 2  (in blocks of a few energies: each call writes its own cache)
python3 uniformita_pos.py --base <data> --outdir plot/uniformita \
    --resistances 340 --energies 20 30 40 --exclude-runs 20592
python3 uniformita_pos.py --only-collect --resistances 340 400 500

# 3
python3 uniformita_maps.py --stage moments --resistances 340 400 500
python3 uniformita_maps.py --stage apply   --resistances 340 400 500
python3 uniformita_maps.py --stage collect

# 4
python3 resolution_final_uniforme.py --central run --syst map --suffix _unif

# 5
bash root/run_fit_root.sh
```

### The 50 MHz condition

The same chain, with `--runset filter50` and separate output directories:

```bash
python3 uniformita_pos.py  --runset filter50 --exclude-runs 20592 \
    --outdir plot/uniformita_50mhz --resistances 340 --energies 60 80 150 250
python3 uniformita_pos.py  --only-collect --outdir plot/uniformita_50mhz --resistances 340

python3 uniformita_maps.py --stage moments --runset filter50 --exclude-runs 20592 \
    --outdir plot/uniformita_maps_50mhz --resistances 340 --energies 60 80 150 250
python3 uniformita_maps.py --stage apply   --runset filter50 --exclude-runs 20592 \
    --outdir plot/uniformita_maps_50mhz --resistances 340 --energies 60 80 150 250
python3 uniformita_maps.py --stage collect --outdir plot/uniformita_maps_50mhz --resistances 340

python3 resolution_final_uniforme.py --central run --syst map --suffix _50mhz \
    --unif  plot/uniformita_50mhz/uniformita_pos.csv \
    --maps  plot/uniformita_maps_50mhz/uniformita_maps.csv \
    --cache plot/uniformita_50mhz/_cache --exclude 340:275
```

Run 20592 has to be excluded by hand even here: it is nominally a 150 GeV run but its
median A_tot matches 80 GeV, so it is not at the energy its label says.

With four energies and three free parameters the N/S/C fit has one degree of freedom
and S comes out unconstrained. Fixing S at the standard value is the only way to read
N and C off these points.

**Watch out for `--only-collect`**: it rewrites the summary CSV with only the
resistances passed in `--resistances`. Run on a single resistance, it drops the
others from the CSV (the cache is untouched, so re-running it on all three is
enough to recover).

---

## Supporting studies

These do not feed the chain, but they produce the results that justify it.

| script | what it shows |
|---|---|
| `quattro_metodi.py` | Comparison of four ways of extracting σ/E: global fit, mean of the per-run σ, rescaling to a reference peak, and separated populations. This is what justifies fitting run by run: a global fit on an energy with two response populations fits two bumps with a single function. |
| `diagnosi_60GeV.py` | `A_tot` distribution run by run: the plot where the two populations are visible. |
| `profili_pernorm.py` | `⟨A_tot⟩` vs centroid profiles, normalising every event to the peak of its own run, fitted with `a + bx + cx²`. Without that normalisation the profile of an energy with several runs is the composition of the runs rather than a response curve. The CSV also reports the curvature one would get without normalising. |
| `linearity.py` | Linearity of the peak against the true beam energy. |
| `sistematica_risoluzione.py` | Drift systematic on σ, scaled until χ²/ndf = 1. |
| `fit_fixedSC.py` + `root/fit_fixedSC.C` | Fits with S and C frozen at the values of a reference resistance, to check whether the three share the same stochastic and constant terms. |

`resolution_final.py` is the previous version of the final curve, superseded by
`resolution_final_uniforme.py`. It is kept because `fit_fixedSC.py` reads the CSV it
produces.

---

## Conventions

**True energies, not nominal ones.** A beam set to 250 GeV actually carries 240.76.
Above 100 GeV the difference reaches 5 %, and using the nominal values fabricates a
spurious non-linearity of a few percent. Every script uses the "Final Energy" column
of the beam log, encoded in the `VERA` dictionary.

**Selection.** `|pos_eta − 18| ≤ 0.2`, `|pos_phi − 6| ≤ 0.2`, `A_tot > 100` ADC. The
noise threshold is the same for all energies and resistances.

**Run conditions.** Not every run in the merged files belongs to the same readout
configuration, and mixing them turns a single response peak into two. `runsets.py`
holds the three lists, and every script that reads runs takes `--runset`:

| set | runs | why |
|---|---|---|
| `FILTER_50MHZ` | 20587-20604 | 50 MHz filter on the readout. Only at 340 ohm, at 60, 80, 150 and 250 GeV; median A_tot 2.6-3.5 % above the standard runs at the same energy. Analysed separately with `--runset filter50`. |
| `HIGH_275` | 20652-20659 | High-response population at 275 GeV, +3.9 %. Same size of shift as the 50 MHz runs but outside that range; cause not established. |
| `OUTLIERS` | 20491 | 120 GeV, 1.5 % above the other run at the same energy, no configuration difference on record. |

`--runset standard` (the default) drops all three; `--runset filter50` keeps only the
50 MHz runs; `--runset all` filters nothing beyond `--exclude-runs`, which stays
available for one-off exclusions.

After these exclusions eight of the twelve energies at 340 ohm are left with a single
run. A single run means no run-to-run drift systematic can be estimated, so the error
bar on those points is purely statistical and the chi2 of the N/S/C fit is not a
measure of goodness of fit. `sistematica_risoluzione.py --fallback` exists to assign
those points the systematic measured where two or more runs are available.

**Parabola fits.** Always in the form `a + bx + cx²`, never `p₁ + p₂(x − p₀)²`: the
latter has degenerate vertex and curvature — a shift of the vertex is compensated by
a change in curvature — and on this data the fit runs away. The former is a linear
system and has a single solution.

**Plot titles.** Cuts in mathematical form plus identifiers (resistance, energy, run,
N) only. Axis labels in English.
