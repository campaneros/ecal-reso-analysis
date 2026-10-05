#!/usr/bin/env python3
"""
Skim of the per-run reco files for the time analysis: keeps only the branches the
time stages read, so that the files fit through a web area and a slow link.

  python3 skim_time.py --source <Reco dir> --outdir <dir> [--runs 19564 19565 ...]

Every *.root file under --source (recursively) is skimmed to <outdir>/<same name>.
A file whose tree lacks one of the wanted branches keeps the ones it has and says
which are missing. Run it on lxplus with ROOT, e.g.
  source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
"""

import argparse
import glob
import os
import re

import ROOT

TREE = "h4_reco"
BRANCHES = ["run", "spill", "evt", "energy", "A_tot", "A", "A_err", "t", "t_err",
            "sel_ieta", "sel_iphi", "pos_eta", "pos_phi", "bsl_rms",
            "mcp_A", "mcp_A_err", "mcp_t", "mcp_t_err", "mcp_fit_status",
            "fit_status", "gs", "clk_period", "clk_phase",
            "hodo_x1_nclusters", "hodo_y1_nclusters", "hodo_x1_pos", "hodo_y1_pos",
            "hodo_x2_nclusters", "hodo_y2_nclusters", "hodo_x2_pos", "hodo_y2_pos"]


def run_of(path):
    match = re.search(r"(\d{5})", os.path.basename(path))
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--runs", nargs="*", type=int, default=[],
                        help="keep only these runs (default: every file)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.source, "**", "*.root"), recursive=True))
    print(len(paths), "files under", args.source, flush=True)
    for path in paths:
        if args.runs and run_of(path) not in args.runs:
            continue
        target = os.path.join(args.outdir, os.path.basename(path))
        if os.path.exists(target) and not args.overwrite:
            print("exists, skipped:", target)
            continue
        frame = ROOT.RDataFrame(TREE, path)
        present = set(str(name) for name in frame.GetColumnNames())
        keep = [name for name in BRANCHES if name in present]
        missing = [name for name in BRANCHES if name not in present]
        frame.Snapshot(TREE, target, keep)
        print(f"{os.path.basename(path)}: {len(keep)} branches, {os.path.getsize(target) / 1e6:.1f} MB"
              + (f", missing {missing}" if missing else ""), flush=True)


if __name__ == "__main__":
    main()
