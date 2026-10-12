# Time resolution between two crystals, ECAL test beam H4 2025

Three scripts to measure the time resolution of two neighbouring crystals when the beam is aimed between them (runs with `phi 5/6` in the good-run list). They need Python 3 with PyROOT and numpy.

```
skim files  ->  two_crystal_tree.py  ->  per_run_dcb.py
                                     ->  th2_per_run.py
```

## 1. Merge the runs into a tree

`two_crystal_tree.py` reads the per-run reco files `<run>_<energy>_reco.root` (tree `h4_reco`) and merges the two-crystal runs of one eta into a flat tree called `two_crystals`, with no cut.

```
python3 two_crystal_tree.py --skim-dir <dir with the reco files> --eta 18 \
    --out two_crystal_tree_eta18.root --good-runs good_run_list_2025.csv
```

Crystal 1 is phi 5 and crystal 2 is phi 6. For each of them the tree has `A_*`, `t_*` (in samples), `lg_nsample_*` and `fit_status_*`, plus `A_3x3`, `A_tot`, `pos_eta`, `pos_phi`, the first hodoscope cluster of every plane (`hodo_x1`, `hodo_x2`, `hodo_y1`, `hodo_y2`) with their number of clusters, `run`, `spill`, `evt` and `energy`.

`good_run_list_2025.csv` is the run list used so far (columns `Run, Energy (GeV), ..., eta, phi, ...`). Only the rows with `phi` equal to `5/6` and the chosen `eta` are used. Runs taken after 19946 are not in it.

## 2. Sigma and mean against A_eff

`per_run_dcb.py` bins the events in A_eff/sigma_n, fits a double Crystal Ball to t2 - t1 in every bin, and plots sigma/sqrt(2) and the mean of t2 - t1 against A_eff, run by run. All the fits are done with ROOT.

```
python3 per_run_dcb.py --tree two_crystal_tree_eta18.root --outdir out \
    --larger second --max-energy 175 --energy-fix 19398:200
```

With `--merge` all the runs are fitted together and the curve sigma = N/A_eff (+) C is fitted from `--min-x` (150) on:

```
python3 per_run_dcb.py --tree two_crystal_tree_eta18.root --outdir out_merged --merge \
    --larger second --window 0.1 --max-energy 175 --energy-fix 19398:200 --fine-bins 100
```

This gives `N = 12.32 +- 0.09 ns` and `C = 23.3 +- 0.3 ps` for eta 18 with the runs up to 175 GeV.

Output in `--outdir`: `points.csv`, `points.root` (graphs and canvases), `projections.root` (the histogram of every bin with its fit), `sigma.png`, `mean.png` and, with `--merge`, `sigma_fit.png` and `sigma_fit.root`.

## 3. Time difference against A_eff, one TH2 per run

```
python3 th2_per_run.py --tree two_crystal_tree_eta18.root --outdir out_th2 \
    --larger second --max-energy 175 --energy-fix 19398:200
```

One image with a TH2 of t2 - t1 against A_eff for every run, and one for all the runs together; the histograms are also saved in `th2_per_run.root`.

## Definitions

- A_eff/sigma_n = sqrt(2 / ((sigma_n/A_1)^2 + (sigma_n/A_2)^2)), with sigma_n = 2.5 ADC (`--sigma-noise`).
- t2 - t1 = 6.25 ns * (t_2 - t_1), the time is in samples of 6.25 ns.
- sigma/sqrt(2) is the resolution of one crystal. The A_eff bins are made of at least `--min-per-bin` events (600).

## Cuts

By default the events must have `A_tot > 100`, both crystals in high gain (`lg_nsample == 0`), `fit_status == 0` on both and positive amplitudes. This is `--cut` and can be replaced. On top of it:

| option | cut |
|---|---|
| `--extra-cut 'A_1<3500 && A_2<3500'` | any ROOT expression |
| `--larger second` / `first` | `A_2 > A_1` / `A_1 > A_2` |
| `--window 0.1` | `abs(pos_phi - 5.5) < 0.1` and `abs(pos_eta - 18) < 0.1` (`--phi-centre`, `--eta-centre`) |
| `--max-energy 175` | drop the runs above 175 GeV |
| `--exclude-runs 19397 19398` | drop single runs |

Run 19398 is listed at 150 GeV, but its amplitudes are those of 200 GeV: use `--energy-fix 19398:200`.

Every script has `--help` with all the options.
