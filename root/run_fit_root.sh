#!/bin/bash
# Repeats in ROOT the fits already done in Python and stores TGraphErrors, TF1 and
# TCanvas objects in .root files that can be versioned on GitHub. Run from the
# repository root.
#
#   plot/root/resolution_fits.root    four fits: {nopos, pos} x {indep, common}
#                                     nopos  = without the centroid systematic
#                                     pos    = with POS_eff subtracted
#                                     indep  = N, S, C per resistance
#                                     common = S and C common, N per resistance
#   plot/root/resolution_fixedSC.root S and C frozen at the 340 ohm values
#
# The printed numbers must match those in the CSV files produced by the Python
# scripts (plot/root/fit_summary.csv and plot/root/fit_fixedSC_summary.csv). If they
# do not, the culprit is almost always the minimiser seed: compare N, S and C before
# blaming the data.
set -e
cd "$(dirname "$0")/../.."
if ! command -v root >/dev/null 2>&1; then
  echo "ROOT is not in PATH. With a standard installation:  source /path/to/thisroot.sh"
  exit 1
fi
python3 plot/fit_root_all.py --plotdir plot --outdir plot/root
python3 plot/fit_fixedSC.py  --plotdir plot --outdir plot/root
root -l -b -q plot/root/fit_resolution.C
root -l -b -q plot/root/fit_fixedSC.C
echo
echo "done: plot/root/resolution_fits.root, plot/root/resolution_fixedSC.root"
