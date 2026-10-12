import argparse
import csv
import math
import os

import numpy as np
import ROOT

DEFAULT_CUT = ("A_tot>100 && lg_nsample_1==0 && lg_nsample_2==0 && fit_status_1==0 "
               "&& fit_status_2==0 && A_1>0 && A_2>0")
MARKERS = [20, 21, 22, 23, 33, 34, 29, 47]
DELTA_T_RANGE = 3.0
DELTA_T_BINS = 600
MIN_FIT_ENTRIES = 50
MIN_PEAK_COUNTS = 30
MAX_SIGMA = 1.0
FIT_RANGE_SIGMAS = 6.

ROOT.gInterpreter.Declare(r'''
double double_crystal_ball(double *x, double *par) {
  const double alpha_low = par[0], alpha_high = par[1], n_low = par[2], n_high = par[3];
  const double mean = par[4], sigma = par[5], norm = par[6];
  const double t = (x[0] - mean) / sigma;
  if (t < -alpha_low) {
    const double tail = (n_low / alpha_low) - alpha_low - t;
    return norm * std::exp(-0.5 * alpha_low * alpha_low) * std::pow(std::max(alpha_low / n_low * tail, 1e-12), -n_low);
  }
  if (t > alpha_high) {
    const double tail = (n_high / alpha_high) - alpha_high + t;
    return norm * std::exp(-0.5 * alpha_high * alpha_high) * std::pow(std::max(alpha_high / n_high * tail, 1e-12), -n_high);
  }
  return norm * std::exp(-0.5 * t * t);
}
''')


def add_selection_arguments(parser):
    parser.add_argument("--tree", required=True, help="tree made by two_crystal_tree.py")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cut", default=DEFAULT_CUT, help="ROOT selection (default: %(default)s)")
    parser.add_argument("--extra-cut", help="ROOT expression added to --cut, e.g. 'A_1<3500 && A_2<3500'")
    parser.add_argument("--larger", choices=("first", "second"), help="keep only A_1 > A_2 or A_2 > A_1")
    parser.add_argument("--window", type=float, help="keep events with |pos_phi - phi-centre| and |pos_eta - eta-centre| below this")
    parser.add_argument("--eta-centre", type=float, default=18.)
    parser.add_argument("--phi-centre", type=float, default=5.5)
    parser.add_argument("--max-energy", type=int, default=1000, help="drop the runs above this energy [GeV]")
    parser.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    parser.add_argument("--energy-fix", nargs="*", default=[], help="run:energy pairs overriding the energy of a run, e.g. 19398:200")
    parser.add_argument("--sigma-noise", type=float, default=2.5, help="noise of one crystal [ADC]")


def build_cut(args):
    cut = args.cut
    if args.extra_cut:
        cut += f" && ({args.extra_cut})"
    if args.larger:
        cut += " && A_2>A_1" if args.larger == "second" else " && A_1>A_2"
    if args.window:
        cut += (f" && abs(pos_phi-{args.phi_centre})<{args.window}"
                f" && abs(pos_eta-{args.eta_centre})<{args.window}")
    for run in args.exclude_runs:
        cut += f" && run!={run}"
    return cut


def load_events(args):
    noise = args.sigma_noise
    frame = (ROOT.RDataFrame("two_crystals", args.tree).Filter(build_cut(args))
             .Define("a_eff", f"sqrt(2./(pow({noise}/A_1,2)+pow({noise}/A_2,2)))")
             .Define("delta_t", "6.25*(t_2-t_1)"))
    data = frame.AsNumpy(["a_eff", "delta_t", "run", "energy"])
    fixes = {int(item.split(":")[0]): int(item.split(":")[1]) for item in args.energy_fix}
    energy = np.array([fixes.get(int(run), int(value)) for run, value in zip(data["run"], data["energy"])])
    keep = energy <= args.max_energy
    return data["a_eff"][keep], data["delta_t"][keep], data["run"][keep], energy[keep]


def adaptive_edges(values, fine_bins, min_events):
    low, high = np.percentile(values, [0.1, 99.9])
    grid = np.linspace(low, high, fine_bins + 1)
    grid[0], grid[-1] = min(float(values.min()), grid[0]), max(float(values.max()), grid[-1])
    counts, _ = np.histogram(values, bins=grid)
    edges, filled = [float(np.nextafter(grid[-1], np.inf))], 0
    for index in range(fine_bins - 1, -1, -1):
        filled += counts[index]
        if filled >= min_events and index > 0:
            edges.append(float(grid[index]))
            filled = 0
    edges.append(float(grid[0]))
    edges = np.array(edges[::-1])
    if len(edges) > 2 and np.histogram(values, bins=edges)[0][0] < min_events:
        edges = np.delete(edges, 1)
    return edges


def peak_seed(histogram):
    histogram = histogram.Clone("seed")
    while histogram.GetMaximum() < MIN_PEAK_COUNTS and histogram.GetNbinsX() % 2 == 0:
        histogram.Rebin(2)
    contents = np.array([histogram.GetBinContent(index) for index in range(1, histogram.GetNbinsX() + 1)])
    top = int(contents.argmax())
    half = 0.5 * contents[top]
    low, high = top, top
    while low > 0 and contents[low - 1] > half:
        low -= 1
    while high < len(contents) - 1 and contents[high + 1] > half:
        high += 1
    width = (high - low + 1) * histogram.GetBinWidth(1)
    return histogram.GetBinCenter(top + 1), max(width / 2.355, histogram.GetBinWidth(1))


def fit_double_crystal_ball(histogram):
    if histogram.GetEntries() < MIN_FIT_ENTRIES:
        return None, None
    mean, sigma = peak_seed(histogram)
    low = max(mean - FIT_RANGE_SIGMAS * sigma, -DELTA_T_RANGE)
    high = min(mean + FIT_RANGE_SIGMAS * sigma, DELTA_T_RANGE)
    function = ROOT.TF1(f"dcb_{histogram.GetName()}", ROOT.double_crystal_ball, low, high, 7)
    function.SetParNames("alpha_low", "alpha_high", "n_low", "n_high", "mean", "sigma", "norm")
    function.SetParameters(2., 2., 2., 2., mean, sigma, histogram.GetMaximum())
    limits = ((0.1, 10), (0.1, 10), (1, 10), (1, 10), (low, high), (0.1 * sigma, high - low))
    for index, (lower, upper) in enumerate(limits):
        function.SetParLimits(index, lower, upper)
    result = histogram.Fit(function, "QRS0")
    if not result.Get() or result.Status() != 0 or function.GetParameter(5) > MAX_SIGMA:
        return None, None
    histogram.GetListOfFunctions().Add(function)
    return dict(mean=function.GetParameter(4), sigma=function.GetParameter(5),
                error=function.GetParError(5)), function


def fit_bins(a_eff, delta_t, min_per_bin, fine_bins, label, projections):
    constant = float(np.median(delta_t))
    shifted = delta_t - constant
    inside = np.abs(shifted) < DELTA_T_RANGE
    a_eff, shifted = a_eff[inside], shifted[inside]
    edges = adaptive_edges(a_eff, fine_bins, min_per_bin)
    bin_of_event = np.clip(np.digitize(a_eff, edges) - 1, 0, len(edges) - 2)
    points = []
    for index in range(len(edges) - 1):
        in_bin = bin_of_event == index
        histogram = ROOT.TH1D(f"{label.split()[0]}_bin{index}",
                              f"{label}: A_eff in [{edges[index]:.4g}, {edges[index + 1]:.4g}];t_2-t_1 - constant [ns];events",
                              DELTA_T_BINS, -DELTA_T_RANGE, DELTA_T_RANGE)
        histogram.FillN(int(in_bin.sum()), np.ascontiguousarray(shifted[in_bin], dtype=float), np.ones(int(in_bin.sum())))
        fit, function = fit_double_crystal_ball(histogram)
        if fit is None:
            continue
        projections.cd()
        histogram.Write()
        points.append(dict(a_eff=float(a_eff[in_bin].mean()), n_events=int(histogram.GetEntries()),
                           mean_ns=fit["mean"] + constant, sigma_ns=fit["sigma"] / math.sqrt(2),
                           error_ns=fit["error"] / math.sqrt(2)))
    return points


def make_graph(points, column, name, scale=1.):
    errors = [point["error_ns"] * scale if column == "sigma_ns" else 0. for point in points]
    graph = ROOT.TGraphErrors(len(points), np.array([point["a_eff"] for point in points]),
                              np.array([point[column] * scale for point in points]),
                              np.zeros(len(points)), np.array(errors))
    graph.SetName(name)
    return graph


def draw(series, column, path, title, y_title, y_range, x_min, output):
    ROOT.gStyle.SetPalette(ROOT.kRainbow)
    colors = ROOT.TColor.GetPalette()
    energies = [energy for _, energy, _ in series]
    low, high = min(energies), max(energies)
    canvas = ROOT.TCanvas(f"canvas_{column}", "", 1250, 750)
    canvas.SetRightMargin(0.2)
    canvas.SetGrid()
    canvas.DrawFrame(x_min, y_range[0], 1500, y_range[1], f"{title};A_{{eff}}/#sigma_{{n}};{y_title}")
    legend = ROOT.TLegend(0.81, 0.12, 0.995, 0.9)
    legend.SetBorderSize(0)
    graphs = []
    for index, (label, energy, points) in enumerate(series):
        points = [point for point in points if point["a_eff"] >= x_min]
        if not points:
            continue
        graph = make_graph(points, column, f"{column}_{label.split()[0]}", 1000. if column == "sigma_ns" else 1.)
        color = int(colors[int(round((len(colors) - 1) * (energy - low) / max(high - low, 1)))])
        graph.SetMarkerStyle(MARKERS[index % len(MARKERS)])
        graph.SetMarkerColor(color)
        graph.SetLineColor(color)
        graph.Draw("PL same")
        legend.AddEntry(graph, label, "lp")
        graphs.append(graph)
        output.cd()
        graph.Write()
    legend.Draw()
    canvas.SaveAs(path)
    output.cd()
    canvas.Write()


def fit_resolution(points, x_min, path, title):
    points = [point for point in points if point["a_eff"] >= x_min]
    graph = make_graph(points, "sigma_ns", "sigma_graph")
    curve = ROOT.TF1("curve", "sqrt(pow([0]/x,2)+pow([1],2))", x_min, 1.02 * max(point["a_eff"] for point in points))
    curve.SetParNames("N", "C")
    curve.SetParameters(10., 0.03)
    curve.SetParLimits(0, 0., 100.)
    curve.SetParLimits(1, 0., 1.)
    result = graph.Fit(curve, "QRS0")
    noise, constant = curve.GetParameter(0), curve.GetParameter(1)
    noise_error, constant_error = curve.GetParError(0), curve.GetParError(1)
    canvas = ROOT.TCanvas("fit_canvas", "", 1000, 700)
    canvas.SetGrid()
    graph.SetTitle(f"{title};A_{{eff}}/#sigma_{{n}};#sigma_{{DCB}}/#sqrt{{2}} [ns]")
    graph.SetMarkerStyle(20)
    graph.Draw("AP")
    curve.SetLineColor(ROOT.kRed)
    curve.Draw("same")
    box = ROOT.TPaveText(0.55, 0.62, 0.89, 0.88, "NDC")
    box.AddText(f"N = {noise:.2f} #pm {noise_error:.2f} ns")
    box.AddText(f"C = {1000 * constant:.1f} #pm {1000 * constant_error:.1f} ps")
    box.AddText(f"#chi^{{2}}/ndf = {result.Chi2():.1f}/{result.Ndf()}")
    box.Draw()
    canvas.SaveAs(path)
    output = ROOT.TFile(path.replace(".png", ".root"), "RECREATE")
    graph.Write()
    curve.Write("fit")
    canvas.Write()
    output.Close()
    print(f"N = {noise:.2f} +- {noise_error:.2f} ns, C = {1000 * constant:.1f} +- {1000 * constant_error:.1f} ps, "
          f"chi2/ndf = {result.Chi2():.1f}/{result.Ndf()}")


def main():
    parser = argparse.ArgumentParser(
        description="Sigma and mean of t2 - t1 against A_eff, double Crystal Ball fit in every A_eff bin, run by run "
                    "or with all runs together (--merge, which also fits sigma = N/A_eff (+) C).",
        epilog="examples:\n"
               "  python3 per_run_dcb.py --tree two_crystal_tree_eta18.root --outdir out --larger second --max-energy 175\n"
               "  python3 per_run_dcb.py --tree two_crystal_tree_eta18.root --outdir out --merge --window 0.1 "
               "--exclude-runs 19397 19398 --extra-cut 'A_2<3500'",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_arguments(parser)
    parser.add_argument("--min-per-bin", type=int, default=600)
    parser.add_argument("--fine-bins", type=int, default=30, help="bins of the grid that is merged until --min-per-bin is reached")
    parser.add_argument("--min-x", type=float, default=150., help="lowest A_eff drawn and fitted")
    parser.add_argument("--sigma-range", nargs=2, type=float, default=(15, 70), help="y range of the sigma plot [ps]")
    parser.add_argument("--mean-range", nargs=2, type=float, default=(2.95, 3.25), help="y range of the mean plot [ns]")
    parser.add_argument("--merge", action="store_true", help="fit all runs together instead of one run at a time")
    args = parser.parse_args()

    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    os.makedirs(args.outdir, exist_ok=True)
    a_eff, delta_t, run, energy = load_events(args)

    projections = ROOT.TFile(os.path.join(args.outdir, "projections.root"), "RECREATE")
    series = []
    if args.merge:
        series.append(("all runs", int(energy.max()),
                       fit_bins(a_eff, delta_t, args.min_per_bin, args.fine_bins, "all_runs", projections)))
    else:
        for number in sorted(np.unique(run), key=lambda value: (np.median(energy[run == value]), value)):
            mask = run == number
            if mask.sum() < args.min_per_bin:
                continue
            run_energy = int(np.median(energy[mask]))
            points = fit_bins(a_eff[mask], delta_t[mask], args.min_per_bin, args.fine_bins, f"run{int(number)}", projections)
            series.append((f"{int(number)} ({run_energy} GeV)", run_energy, points))
    projections.Close()

    with open(os.path.join(args.outdir, "points.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "energy", "a_eff", "n_events", "mean_ns", "sigma_over_sqrt2_ns", "error_ns"])
        for label, run_energy, points in series:
            for point in points:
                writer.writerow([label, run_energy, point["a_eff"], point["n_events"], point["mean_ns"],
                                 point["sigma_ns"], point["error_ns"]])

    output = ROOT.TFile(os.path.join(args.outdir, "points.root"), "RECREATE")
    draw(series, "sigma_ns", os.path.join(args.outdir, "sigma.png"), "sigma of t2 - t1",
         "#sigma_{DCB}/#sqrt{2} [ps]", args.sigma_range, args.min_x, output)
    draw(series, "mean_ns", os.path.join(args.outdir, "mean.png"), "mean of t2 - t1",
         "mean (t_{2}-t_{1}) [ns]", args.mean_range, args.min_x, output)
    output.Close()
    if args.merge:
        fit_resolution(series[0][2], args.min_x, os.path.join(args.outdir, "sigma_fit.png"),
                       f"all runs together, DCB, >={args.min_per_bin} events/bin")


if __name__ == "__main__":
    main()
