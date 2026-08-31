#!/bin/bash
# Lancia reweight_flat_2d.py su tutti i dataset merged.
cd "$(dirname "$0")/.."
OUT=plot/reweight
CSV=reweight_summary_all.csv
mkdir -p "$OUT"
: > "$OUT/$CSV"
for E in 20 30 40 60 80 100 120 150 175 200 225 250 275; do
  f="reco_340ohm/${E}_merged.root"; [ -f "$f" ] || continue
  python3 plot/reweight_flat_2d.py --file "$f" --energy $E --resistance 340 --outdir "$OUT" --csv "$CSV"
done
for E in 20 40 60 80 100 150 200 225 250; do
  f="reco_400ohm/${E}_400_merged.root"; [ -f "$f" ] || continue
  python3 plot/reweight_flat_2d.py --file "$f" --energy $E --resistance 400 --outdir "$OUT" --csv "$CSV"
done
for E in 30 40 50 60 80 100 150; do
  f="reco_500ohm/${E}_500_merged.root"; [ -f "$f" ] || continue
  python3 plot/reweight_flat_2d.py --file "$f" --energy $E --resistance 500 --outdir "$OUT" --csv "$CSV"
done
echo "FATTO"
