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
| drift | subtracted as well, and **not** an error bar: run-to-run systematic on sigma, computed **energy by energy on the per-run sigma values of the selection in use**: the extra error which, added in quadrature, makes the fit of those sigmas to a constant give chi2/ndf = 1 (the PDG scale-factor method). It must not be imported from another selection: cutting on the hodoscope keeps a different set of runs, so the drift differs. See **Reading a drift of zero** below |
| centroid | how much the answer depends on **how** the response surface is estimated: the difference between correcting with the parabola of each run and correcting with the parabola of the energy, `uniformita_maps.py`. Median 0.0006 percentage points, and exactly zero on the six points with a single run, where the two maps coincide by construction |

Typical sizes, as medians over the points of each resistance:

| | statistical | drift | centroid |
|---|---|---|---|
| 340 ohm | 0.008 | 0.008 | 0.0004 |
| 400 ohm | 0.009 | 0.020 | 0.0007 |
| 500 ohm | 0.011 | 0.018 | 0.0042 |

**A caveat on the chi2.** After the run exclusions, eight of the eleven energies at
340 ohm that survive the hodoscope selection have a single run. There the drift systematic is zero by construction and the
error bar is purely statistical, so the chi2 of the N/S/C fit is not a measure of
goodness of fit: it is dominated by a point-to-point scatter that nothing is left to
estimate. `sistematica_risoluzione.py --fallback` exists to assign those points the
systematic measured where two or more runs are available; whether to apply it is an
open choice.

**Are the per-run error bars right?** The whole drift depends on them, so they were
checked against a bootstrap — `dcb_error_check.py`. The bar on each per-run sigma/mu is
the HESSE error of the double-CB fit propagated to the ratio,
`e = (sigma/mu) * sqrt((d_sigma/sigma)^2 + (d_peak/peak)^2)`. It comes out three to
four times larger than `sigma/sqrt(2N)`, but that reference does not apply here: it is
the error on the RMS of a Gaussian from an unbinned ML fit, whereas sigma is the width
of the *core* of a seven-parameter double CB fitted by binned least squares and
correlated with the four tail parameters. Resampling the events of a run and refitting
gives, as the median of HESSE / bootstrap:

| point | runs | median ratio |
|---|---|---|
| 340 ohm 225 GeV | 7 | 1.00 |
| 400 ohm 80 GeV | 10 | 0.72 |
| 500 ohm 60 GeV | 3 | 0.83 |

The bars are right on average and, where they are not, they are **smaller** than the
truth by 20-30 %. The direction matters: too-small errors make the chi2 too large and
therefore the drift too large, so the points where the drift comes out zero would come
out zero with the correct errors as well.

**Reading a drift of zero.** A drift term equal to zero means one of two different
things, and the CSV keeps them apart through the `nrun_*` and `*_chi2` columns:

* `nrun = 1` — the point has a single run, there is no run-to-run dispersion to
  measure and the drift is undefined, not small. Eight of the eleven energies at
  340 ohm are in this situation after the run exclusions;
* `nrun > 1` and `chi2/ndf <= 1` — the per-run sigmas are already compatible with a
  constant within their own errors, so the extra error needed to reach chi2/ndf = 1 is
  exactly zero. This is what the PDG prescription gives, not a failure of it.

The two cases are drawn differently in the lower panel of the resolution plot: a cross
on the axis for a single run, an open square for a drift that came out zero.

Where chi2/ndf is above 1 the drift is there and is usually the largest term at that
point: 340 ohm 20 GeV chi2/ndf 1.85 gives 0.045, 400 ohm 40 GeV chi2/ndf 6.8 gives
0.052, 500 ohm 30 GeV chi2/ndf 6.6 gives 0.045.

The drift comes out **smaller with the hodoscope cut than with the centroid one**, and
this is a consequence of the definition rather than a discrepancy: the hodoscope cut
keeps roughly half the events, so the error on each per-run sigma grows by about
sqrt(2), the chi2 against a constant falls by about a factor of two, and the excess
scatter that the drift is meant to describe shrinks with it. The drift measures what
is left over and above the statistical error, and that error is larger here.

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

**Which planes.** x is the average of the two x planes. y is taken from **y1**
(`--yplane`, default y1). y2 fires a single cluster more often, 65 % of events against
42 %, but that is efficiency and not quality: profiled against A_tot, y1 gives a clean
parabola over its whole range while y2 is jagged below zero and usable only above it.
Cutting on y1 needs no range restriction, keeps the window centred on the crystal
instead of one-sided, and raises the fraction of events kept from about 35 % to about
60 %. Only events with exactly one cluster in the planes used are kept; the surviving
fraction is written to the output CSV point by point.

**No range is imposed a priori** on either coordinate: the profile covers the whole
range the data span, from the 0.5th to the 99.5th percentile.

**Two ways of setting the window**, through `--window`:

`plateau` (default) needs no fit: the window is the contiguous range around the
maximum of the response profile where `<A_tot>` stays within `--tol`, 0.5 % by
default, of its plateau value. Windows at different energies are then comparable by
construction, because the response falls by the same amount inside each.

`parabola` takes the window as vertex +- `--half` * W, with the vertex of the response
parabola as the centre and W the crystal width from the ratio of the curvatures in
millimetres and in crystal units.

The fit range is not chosen, it is **scanned**. A single fixed range cannot work: the
response is parabolic only near the crystal centre, so a fit over everything the data
span puts the lever arm on the tails and the vertex follows them; but a fixed window
around the maximum cannot be right at every energy either, because the beam moves --
the x vertex runs from -3 mm at 20 GeV to -10 mm at 175 GeV -- and at the top energies
part of the crystal falls outside the hodoscope acceptance altogether. So the profile
is fitted over `[peak - h, peak + h]` for every h in `SCAN_HALVES` = 5, 6, 7, 8, 9,
10 mm, **independently in x and in y**, and the answer is accepted only if it does not
depend on h:

| check | threshold | what it catches |
|---|---|---|
| fits with a maximum inside their own range | at least 4 of 6 | profiles with no maximum in the acceptance |
| excursion of the vertex over the scan | <= 1.5 mm | a vertex dragged by the tails; the window is only +- 0.2 W ~ 5 mm, so 1.5 mm is already a third of it |
| relative spread of W over the scan | <= 15 % | a curvature that is not the crystal's |
| median W | 12 to 40 mm | a runaway fit; W comes out 21-28 mm everywhere it is accepted, against the 24.2 mm crystal pitch |
| window inside the range the data span | -- | a cut that would fall outside the hodoscope |

The vertex and W returned are the medians over the scan, which is more stable than any
single fit. Where the checks fail **there is no parabola to be found** at that energy
in that view: the point is dropped from the parabola chain with the reason printed,
rather than fitted anyway. That is the hodoscope acceptance limit, and it is drawn as
such -- red title and the reason underneath -- by `hodo_windows.py`.

With `--runset standard` and `--yplane y1` the scan fails on 5 of the 58 (resistance,
energy, view) combinations, and they are exactly the physically suspect ones:

| point | view | why |
|---|---|---|
| 340 ohm 250 GeV | x | vertex moves by 2.8 mm across the fit ranges |
| 340 ohm 250 GeV | y | only 3 of 6 fit ranges give a maximum |
| 340 ohm 275 GeV | y | the window falls outside the hodoscope acceptance |
| 400 ohm 20 GeV | x | W = 9 mm, not a crystal |
| 400 ohm 20 GeV | y | no curvature in crystal units available |

Those energies do not get a parabola window. They are **not thrown away**: the window
falls back to the plateau definition, which needs no fit, the `window` column of the
CSV records it (`x+y-plateau`, `y-plateau`) and the point is drawn with a grey ring
and the legend entry *no parabola: plateau window*. Dropping them would have made them
vanish from the plot without saying why.

**When no run has enough events.** The per-run fit needs 300 events. At **340 ohm
250 GeV** the hodoscope cut keeps 846 events out of 50691, 2 %, spread over eight runs
— about a hundred each — because the y window sits at `[+10.8, +14.5] mm`, against the
edge of the acceptance: the beam is not on the part of the crystal the hodoscope sees.
Rather than lose the point, the fit is then done **once on all the runs pooled**. The
point comes out with the error of that single fit, the drift is undefined (there is no
longer a run-to-run dispersion to measure), `nrun_* = 0` and `*_pooled = 1` record it,
and it is drawn with a violet square and the legend entry *one pooled fit*. It is of
course a point to treat with suspicion, which is why 250 GeV is in `--nofit-energies`
by default.

**Points excluded from the fit but kept in the plot.** `--nofit-energies`, 250 and
275 GeV by default, are drawn as open markers and left out of the N/S/C fit. Every
resolution figure is produced twice, once that way and once with all the points in the
fit (`_allpoints`), for both chains, so that the weight of those two energies on the
parameters can be read off directly:

| chain | fit | 340 ohm | 400 ohm | 500 ohm |
|---|---|---|---|---|
| hodoscope | without 250, 275 | N 281 ± 8, S 2.88, chi2 56.4/8 | N 283 ± 3, S 0.00, chi2 50.4/5 | N 232 ± 19, S 3.25, chi2 18.1/5 |
| hodoscope | all points | N 289 ± 8, S 2.62, chi2 93.7/10 | N 283 ± 3, S 0.01, chi2 51.7/6 | unchanged |
| centroid | without 250, 275 | N 289 ± 6, S 3.19, chi2 79.6/8 | N 311 ± 3, S 0.01, chi2 47.0/5 | N 269 ± 8, S 2.25, chi2 125.1/5 |
| centroid | all points | N 303 ± 5, S 2.75, chi2 196.2/10 | N 312 ± 3, S 0.00, chi2 47.6/6 | unchanged |

Adding the two energies moves N by 8 MeV at 340 ohm with the hodoscope cut and by
14 MeV with the centroid one, and roughly doubles the chi2 in both. 500 ohm has
neither energy, so nothing changes there.

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
# the two window definitions, into two directories
python3 resolution_hodo.py --base <data> --outdir plot/hodo_parab --besdir plot/bes \
    --plotdir plot --resistances 340 400 500 --window parabola
python3 resolution_hodo.py --base <data> --outdir plot/hodo_plateau --besdir plot/bes \
    --plotdir plot --resistances 340 400 500 --window plateau

# the picture behind each cut
python3 hodo_windows.py --base <data> --outdir plot/hodo_parab --plotdir plot \
    --resistances 340 400 500 --window parabola
python3 hodo_windows.py --base <data> --outdir plot/hodo_plateau --plotdir plot \
    --resistances 340 400 500 --window plateau

# the calibration on its own, with the list of points that have no parabola
python3 hodoscope_calib.py --base <data> --outdir plot/hodo_scan --plotdir plot \
    --resistances 340 400 500
```

Outputs of `resolution_hodo.py`:

| file | what |
|---|---|
| `resolution_hodo.csv` | both selections point by point: which window was used, its limits, the response drop inside it, events kept, and for each chain sigma, statistical error, drift, the observed chi2/ndf of the per-run sigmas, and the corrected value |
| `sigma_per_run.csv` | the individual per-run sigma/mu with its error, for both chains — the input of the drift |
| `resolution_terms_<chain>.png` | the three resistances, with `--nofit-energies` drawn but left out of the fit, and the size of every subtracted term in the panel below |
| `resolution_terms_<chain>_allpoints.png` | the same with **every** point inside the fit, so the weight of those energies on N, S and C is visible. Both variants are produced for **both** chains, centroid and hodoscope |
| `drift_check_<chain>_<R>ohm.png` | one panel per energy: the per-run sigma/mu with their errors, the weighted mean, the drift band, and the observed chi2/ndf written in the title — green where it is already <= 1 and the drift is therefore zero, red where an extra error is needed, grey where there is a single run |

Outputs of `hodo_windows.py`, per resistance and per view:

| file | what it shows |
|---|---|
| `hodo_windows_<view>_<R>ohm.png` | one panel per energy, axes zoomed on the parabola: profile with errors, the accepted parabola over the range it was scanned on, the vertex, the window |
| `hodo_windows_<view>_<R>ohm_full.png` | the same panels on a **common scale**, x from -15 to +15 mm and y from 0.90 up, so that panels can be compared with each other and the tails and the edges of the acceptance are visible |

`hodoscope_calib.py` holds the shared machinery -- the response profile, the parabola
scan and its acceptance checks -- and run on its own writes the offset and the scale
per energy, `hodoscope_calib.csv`, plus a plot of the vertex and of W against energy
with the rejected points drawn as open red markers.
