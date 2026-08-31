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

## Runs used

The standard set, after `--runset standard` drops the 50 MHz runs, the 275 GeV high
population and the excluded outliers (see **Run conditions** below).

### 340 ohm — 13 energies, 39 runs

| E [GeV] | runs |
|---|---|
| 20 | 20895 20896 20897 20898 20899 |
| 30 | 20541 |
| 40 | 20530 |
| 60 | 20528 |
| 80 | 20526 |
| 100 | 20521 |
| 120 | 20474 |
| 150 | 20535 |
| 175 | 20539 |
| 200 | 20427 20428 20429 20434 |
| 225 | 20513 20514 20515 20517 20518 20615 20616 20617 |
| 250 | 20481 20482 20560 20561 20562 20563 20564 20565 20566 20585 |
| 275 | 20636 20637 20638 20639 |

Eight of these twelve energies (275 GeV is excluded from the final curve) are left
with a **single run**, which is why the drift systematic cannot be estimated there.

### 400 ohm — 9 energies, 54 runs

| E [GeV] | runs |
|---|---|
| 20 | 20753 |
| 40 | 20841 20842 20843 |
| 60 | 20847 20848 20849 |
| 80 | 20909 20911 20912 20913 20914 20915 20917 20918 20919 20920 |
| 100 | 20769 20770 20771 20772 |
| 150 | 20780 20781 20782 20786 20787 20788 20789 20799 20800 20801 |
| 200 | 20700 20701 20702 |
| 225 | 20676 20677 20678 20679 20680 20681 |
| 250 | 20683 20684 20686 20687 20688 20689 20690 20691 20692 20693 20694 20695 20696 20699 |

### 500 ohm — 7 energies, 30 runs

| E [GeV] | runs |
|---|---|
| 30 | 21045 21046 21047 |
| 40 | 21090 21091 21092 21093 21094 21095 21096 21097 21098 21099 |
| 50 | 21033 21034 21035 21036 21037 |
| 60 | 21081 21082 21116 |
| 80 | 21119 |
| 100 | 21056 21057 21058 |
| 150 | 20938 20950 20951 20953 20954 |

500 ohm 50 GeV is dropped from the final curve because no BES is available for it.

Two runs deserve a note, because they are kept in the standard set but their
configuration is not fully documented: **20615-20617** (225 GeV) are marked
`340 ohm 35 MHz LPF on` in the run sheet, a low-pass filter that does not shift the
response but does change the shaping; **20636-20639** (275 GeV) have an empty CATIA
resistance field.

---

## Systematics

The nominal point is the weighted mean over runs of sigma/mu from the per-run
double-CB fit, each run corrected event by event for the response non-uniformity with
the parabola of that run. Three contributions are then subtracted in quadrature,
because they are not calorimeter resolution:

```
(sigma/E)^2 = (sigma/mu)^2 - BES^2 - synchrotron^2 - POS_eff^2 - drift^2
```

POS_eff only in the chain that cuts on the centroid; drift wherever it is defined,
that is on the points with more than one run.

| term | what it is | how it is obtained |
|---|---|---|
| BES | beam energy spread: the SPS does not deliver monochromatic electrons | `dp/p [%] = sqrt(C3^2 + C8^2)/(27*sqrt(3))` from the collimator half-openings in mm, eq. (2) of CERN-SL-Note-97-81. Read from `bes/rereco_<R>_withBES.csv` |
| synchrotron | radiation loss in the beam-line magnets | `1.92e-7 * E^2.5` in percent, on the true beam energy. Negligible below 100 GeV, 0.17 % at 250 |
| POS_eff | the response is not flat across the selection window, so part of the width comes from where the shower landed | defined as what the correction actually removes, `POS_eff^2 = (sigma_raw/mu)^2 - (sigma_corr/mu)^2`. See below |

**Why POS_eff and not std(f)/mean(f).** The obvious estimate of the position term is
the spread of the response factor over the events. It **overestimates**: 0.166 %
against 0.132 % at 340 ohm, 0.184 against 0.089 at 500. Quadrature subtraction is
exact for the total RMS but not for the sigma of the *core* of a double-CB, and the
position term is a parabola over a nearly flat beam, so it is not Gaussian and widens
the tails more than the core. Verified both ways: on the truncated RMS the quadrature
identity holds exactly (340 ohm 40 GeV: 1.0540 % to 1.0411 % measured against 1.0414
expected), on the fitted sigma it does not.

The error bar is the statistical term alone, since drift and POS_eff are
subtracted rather than carried:

| term | how it is obtained |
|---|---|
| statistical | weighted variance of the per-run sigma/mu, `SE^2 = sum w (x - xbar)^2 / (sum w * (n_eff - 1))` with `n_eff = (sum w)^2 / sum w^2`. This already contains both the noise of the individual fits and the run-to-run spread. Where a point has a single run it is undefined and the fit error is used |
| drift | subtracted as well, and **not** an error bar: run-to-run systematic on sigma, computed **energy by energy on the per-run sigma values of the selection in use**: the extra error which, added in quadrature, makes the fit of those sigmas to a constant give chi2/ndf = 1 (the PDG scale-factor method). With a single run there is no spread and it is zero by construction. It must not be imported from another selection: cutting on the hodoscope keeps a different set of runs, so the drift differs -- at 340 ohm 250 GeV it is 0.022 with the centroid cut and 0.000 with the hodoscope one |
| centroid | how much the answer depends on **how** the response surface is estimated: the difference between correcting with the parabola of each run and correcting with the parabola of the energy, `uniformita_maps.py`. Median 0.0006 percentage points, and exactly zero on the six points with a single run, where the two maps coincide by construction |

Typical sizes, as medians over the points of each resistance:

| | statistical | drift | centroid |
|---|---|---|---|
| 340 ohm | 0.008 | 0.008 | 0.0004 |
| 400 ohm | 0.009 | 0.020 | 0.0007 |
| 500 ohm | 0.011 | 0.018 | 0.0042 |

**A caveat on the chi2.** After the run exclusions, eight of the twelve energies at
340 ohm have a single run. There the drift systematic is zero by construction and the
error bar is purely statistical, so the chi2 of the N/S/C fit is not a measure of
goodness of fit: it is dominated by a point-to-point scatter that nothing is left to
estimate. `sistematica_risoluzione.py --fallback` exists to assign those points the
systematic measured where two or more runs are available; whether to apply it is an
open choice.

Two systematics that were **measured and found negligible**, and are therefore not
carried: the granularity of the response map (grids of 12, 40 and 150 bins per side
give a median spread of 0.001-0.003 percentage points, under 1 % of POS_eff) and the
choice of a flat-illumination reweighting, which does not remove the position term at
all and is kept only as a cross-check.

---

## Cutting on the hodoscope instead of the centroid

The standard selection cuts on the ECAL centroid, which is built from the same
amplitudes whose width is being measured. `resolution_hodo.py` repeats the analysis
with the position cut taken from the hodoscope, which is independent of the
calorimeter.

**The window needs no calibration.** It is the contiguous range around the maximum
of the response profile where `<A_tot>` stays within `--tol` (default 0.5 %) of its
plateau. x is the average of the two x planes over [-15, 0] mm, y is the second
plane only over [0, 8] mm: the first y plane fires a single cluster in 42 % of events
against 65 % for the second, below zero the y profile is jagged with no parabola, and
above 8 mm some energies show a rise of up to 2 % that is not the crystal response.
Only events with exactly one cluster in the planes used are kept, which costs a large
fraction of the statistics; the fraction surviving is written to the output CSV point
by point.

**What differs between the two chains.** The position correction belongs only to the
chain that cuts on the centroid: with the hodoscope the position does not enter the
selection, so there is nothing to correct and POS_eff is not subtracted. The drift is
computed and subtracted in both, each on its own selection.

| | centroid cut | hodoscope cut |
|---|---|---|
| non-uniformity correction | applied, per-run 2D paraboloid | not applied |
| POS_eff subtracted | yes | no |
| drift subtracted | yes, computed on this selection | yes, computed on this selection |
| statistical error | sigma of the sigma, weights 1/sigma^2 | same |

Because of this the constant term of the two chains is not the same quantity: the
centroid one has POS_eff removed, the hodoscope one still contains it. They must not
be compared directly.

```bash
python3 resolution_hodo.py --base <data> --outdir plot/hodo --besdir plot/bes \
    --plotdir plot --resistances 340 400 500 --exclude 340:275
```

Outputs: `resolution_hodo.csv` with both selections point by point, including the
window limits, the response drop inside the window and the number of events kept;
`resolution_hodo_<R>ohm.png` with the two cuts overlaid; and
`resolution_terms_cen.png` / `resolution_terms_hodo.png` with the three resistances
and the size of every subtracted term in the panel below.

`hodoscope_calib.py` measures the hodoscope offset and scale from the response
(vertex of the parabola, and crystal width from the ratio of the curvatures in
millimetres and in crystal units). It is not needed by the plateau cut and is kept as
a separate check of the geometry.
