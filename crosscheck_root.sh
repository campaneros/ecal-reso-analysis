#!/bin/sh
# Lancia crosscheck_root.C su tutti i merged. Da eseguire nella cartella che
# contiene dcb.cxx (cioe' la root di energy-reso-fitter), con ROOT nel path.
#
#   ./plot/crosscheck_root.sh .
#
base=${1:-.}
dcb=$(cd "$(dirname "$0")" && pwd)/..
out=$(cd "$(dirname "$0")" && pwd)
rm -f "$out/root_fit.csv"
for R in 340 400 500; do
  for f in "$base"/reco_${R}ohm/*_merged.root; do
    [ -e "$f" ] || continue
    E=$(basename "$f" | cut -d_ -f1)
    echo "=== $R ohm  $E GeV"
    root -l -b -q "$out/crosscheck_root.C(\"$f\",$E,$R,\"$out/root_fit.csv\",\"$dcb\")"
  done
done
echo
echo "fatto -> $out/root_fit.csv"
echo "ora:  python3 $out/confronta_root_python.py --plotdir $out"
