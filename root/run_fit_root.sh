#!/bin/bash
# Rifa' in ROOT i fit gia' fatti in Python e salva TGraphErrors, TF1 e TCanvas in
# file .root versionabili su GitHub. Da lanciare dalla radice del repo.
#
#   plot/root/resolution_fits.root    quattro fit: {nopos, pos} x {indep, common}
#                                     nopos = senza la sistematica sul centroide
#                                     pos   = con POS_eff sottratto
#                                     indep = N, S, C per resistenza
#                                     common = S e C comuni, N per resistenza
#   plot/root/resolution_fixedSC.root S e C congelati ai valori di 340 ohm
#
# I numeri stampati devono coincidere con quelli dei CSV prodotti dagli script
# Python (plot/root/fit_summary.csv e plot/root/fit_fixedSC_summary.csv). Se non
# coincidono il colpevole e' quasi sempre il seme del minimizzatore: confrontare
# N, S, C prima di dare la colpa ai dati.
set -e
cd "$(dirname "$0")/../.."
if ! command -v root >/dev/null 2>&1; then
  echo "ROOT non e' nel PATH. Con una installazione standard:  source /path/to/thisroot.sh"
  exit 1
fi
python3 plot/fit_root_all.py --plotdir plot --outdir plot/root
python3 plot/fit_fixedSC.py  --plotdir plot --outdir plot/root
root -l -b -q plot/root/fit_resolution.C
root -l -b -q plot/root/fit_fixedSC.C
echo
echo "fatto: plot/root/resolution_fits.root, plot/root/resolution_fixedSC.root"
