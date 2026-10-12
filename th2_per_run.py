import argparse
import math
import os

import numpy as np
import ROOT

from per_run_dcb import add_selection_arguments, load_events


def main():
    parser = argparse.ArgumentParser(
        description="One TH2 of t2 - t1 against A_eff for every run, in a single image, plus all runs together.",
        epilog="example:\n  python3 th2_per_run.py --tree two_crystal_tree_eta18.root --outdir out "
               "--larger second --max-energy 175 --energy-fix 19398:200",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_arguments(parser)
    parser.add_argument("--x-max", type=float, default=1500., help="upper edge of the A_eff axis")
    parser.add_argument("--delta-t-range", nargs=2, type=float, default=(2., 4.), help="range of t2 - t1 [ns]")
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(ROOT.kViridis)
    os.makedirs(args.outdir, exist_ok=True)
    a_eff, delta_t, run, energy = load_events(args)

    groups = []
    for number in sorted(np.unique(run), key=lambda value: (np.median(energy[run == value]), value)):
        groups.append((f"run {int(number)} ({int(np.median(energy[run == number]))} GeV)", run == number))
    groups.append(("all runs", np.ones(len(a_eff), dtype=bool)))

    columns = 4
    rows = math.ceil(len(groups) / columns)
    canvas = ROOT.TCanvas("canvas", "", 1800, 380 * rows)
    canvas.Divide(columns, rows, 0.002, 0.002)
    output = ROOT.TFile(os.path.join(args.outdir, "th2_per_run.root"), "RECREATE")
    histograms = []
    for position, (title, mask) in enumerate(groups, 1):
        histogram = ROOT.TH2D(f"h{position}", f"{title}: {int(mask.sum())} events;A_{{eff}}/#sigma_{{n}};t_{{2}}-t_{{1}} [ns]",
                              150, 0., args.x_max, 200, args.delta_t_range[0], args.delta_t_range[1])
        histogram.FillN(int(mask.sum()), np.ascontiguousarray(a_eff[mask], dtype=float),
                        np.ascontiguousarray(delta_t[mask], dtype=float), np.ones(int(mask.sum())))
        canvas.cd(position)
        ROOT.gPad.SetLogz()
        ROOT.gPad.SetLeftMargin(0.14)
        ROOT.gPad.SetRightMargin(0.12)
        histogram.Draw("colz")
        histogram.Write()
        histograms.append(histogram)
    canvas.SaveAs(os.path.join(args.outdir, "th2_per_run.png"))
    output.Close()


if __name__ == "__main__":
    main()
