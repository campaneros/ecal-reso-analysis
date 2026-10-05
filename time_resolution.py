#!/usr/bin/env python3
"""
Time resolution, after the selection shared with the energy resolution (selection.py).

Modes
  crystals  Delta T between the two crystals the beam was aimed between (runs with
            "5/6" in the eta or phi column of the good-run list): the beam centre X, Y
            has a .5, the two crystals are its two neighbours. x = A_eff / sigma_n =
            sqrt(2 / ((sigma_n/A_1)^2 + (sigma_n/A_2)^2)), y = sigma(Delta T)/sqrt(2).
  mcp-mcp   Delta T between the two MCPs, both with mcp_A > --mcp-min;
            x = 2 / (1/mcp_A[0] + 1/mcp_A[1]), y = sigma(Delta T)/sqrt(2).
  ecal-mcp  ECAL time of the target crystal minus the MCP time (mean of the two MCPs,
            both above --mcp-min), folded on the clock period as in var_bins.C; the
            fold offset is the circular mean of each run instead of a fixed constant.
            Two sets of outputs: x = A_ecal / sigma_n and x = MCP A_eff. y = sigma.

Chain, for every mode
  1. per energy: A_tot cut, run selection, hodoscope window (parabola study) when it
     can be built, and the centroid cut |pos_eta - X| < 0.2 && |pos_phi - Y| < 0.2;
  2. per run: the median (circular mean for ecal-mcp) of Delta T is subtracted;
  3. all runs and energies together: TH2 of Delta T against x with variable x bins of
     --per-bin events each (as var_bins.C)                       -> <mode>_th2.root
  4. Y projection of every x bin, gaussian fit in +- 2 sigma     -> <mode>_projections.root
  5. sigma against x, fitted with sqrt((N/x)^2 + C^2)            -> <mode>_sigma.root/.png/.csv

Usage
  python3 time_resolution.py --base <dir of per-run reco files> --outdir out_time \\
      --mode crystals mcp-mcp ecal-mcp --good-runs bookkeeping2025/good_run_list_2025.csv
"""

import argparse
import csv
import glob
import math
import os
import re
from types import SimpleNamespace

import numpy as np
import ROOT

import common
import selection
from common import runsets

RESISTANCE_2025 = 500
NS_PER_DIGITIZER_UNIT = 0.2       # mcp_t, clk_phase, clk_period are in units of 0.2 ns
# MCP runs of the October 2025 test beam (README_2025.md of the h4docs bookkeeping)
MCP_RUNS = (19582, 19583, 19579, 19580, 19581, 19578, 19576, 19577, 19574, 19575, 19572,
            19573, 19614, 19571, 19565, 19566, 19564, 19567, 19568, 19569, 19632, 19633,
            19626, 19587)
MCP_CRYSTAL = (18, 6)
Y_RANGE_NS, Y_BIN_NS = 3.0, 0.005
SLICE_FIT_SIGMAS, SLICE_FIT_ROUNDS, SLICE_MIN_ENTRIES = 2.0, 3, 50

ROOT.gInterpreter.Declare(r'''
// value of the crystal (eta, phi) in a per-crystal vector of the reco, NaN if absent
template <typename Values, typename Index>
double pipeline_crystal_value(const Values &values, const Index &ieta, const Index &iphi,
                              int eta, int phi) {
  for (size_t index = 0; index < values.size() && index < ieta.size(); ++index)
    if ((int)ieta[index] == eta && (int)iphi[index] == phi) return values[index];
  return std::nan("");
}
''')


# ------------------------------------------------------------------ run sets
def run_files(base):
    """{run: (energy, path)} of the per-run reco files <run>_<E>_reco.root."""
    files = {}
    for path in glob.glob(os.path.join(base, "*.root")):
        match = re.match(r"(\d+)_(\d+)", os.path.basename(path))
        if match:
            files[int(match.group(1))] = (int(match.group(2)), path)
    return files


def two_crystal_runs(good_runs_csv):
    """{run: ((eta_1, phi_1), (eta_2, phi_2))} for the rows of the good-run list whose
    eta or phi column reads "a/b": the beam was aimed between those two crystals."""
    runs = {}
    with open(good_runs_csv) as handle:
        for row in csv.DictReader(handle):
            eta_text, phi_text = row["eta"].strip(), row["phi"].strip()
            if "/" not in eta_text + phi_text:
                continue
            etas = [int(value) for value in eta_text.split("/")]
            phis = [int(value) for value in phi_text.split("/")]
            if len(etas) == 1:
                etas = etas * 2
            if len(phis) == 1:
                phis = phis * 2
            runs[int(row["Run"])] = ((etas[0], phis[0]), (etas[1], phis[1]))
    return runs


def points_of(mode, files, good_runs_csv):
    """One point per (crystal pair, energy): dict(energy, paths, crystals, centre)."""
    if mode == "crystals":
        pairs = two_crystal_runs(good_runs_csv)
    else:
        pairs = {run: (MCP_CRYSTAL, MCP_CRYSTAL) for run in MCP_RUNS}
    grouped = {}
    for run, crystals in pairs.items():
        if run not in files:
            continue
        energy, path = files[run]
        grouped.setdefault((crystals, energy), []).append(path)
    points = []
    for (crystals, energy), paths in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        centre = (0.5 * (crystals[0][0] + crystals[1][0]), 0.5 * (crystals[0][1] + crystals[1][1]))
        points.append(dict(energy=energy, paths=sorted(paths), crystals=crystals, centre=centre))
    return points


# ------------------------------------------------------------------ Delta T and x
def time_columns(mode, crystals):
    """Expressions read from the tree for this mode."""
    (eta_1, phi_1), (eta_2, phi_2) = crystals
    columns = {"clk": "(double)clk_period"}
    if mode == "crystals":
        columns.update({
            "amp_1": f"pipeline_crystal_value(A, sel_ieta, sel_iphi, {eta_1}, {phi_1})",
            "amp_2": f"pipeline_crystal_value(A, sel_ieta, sel_iphi, {eta_2}, {phi_2})",
            "time_1": f"pipeline_crystal_value(t, sel_ieta, sel_iphi, {eta_1}, {phi_1})",
            "time_2": f"pipeline_crystal_value(t, sel_ieta, sel_iphi, {eta_2}, {phi_2})"})
        return columns
    columns.update({"mcp_amp_0": "(double)mcp_A[0]", "mcp_amp_1": "(double)mcp_A[1]",
                    "mcp_time_0": "(double)mcp_t[0]", "mcp_time_1": "(double)mcp_t[1]"})
    if mode == "ecal-mcp":
        columns.update({
            "phase": "(double)clk_phase",
            "amp_1": f"pipeline_crystal_value(A, sel_ieta, sel_iphi, {eta_1}, {phi_1})",
            "time_1": f"pipeline_crystal_value(t, sel_ieta, sel_iphi, {eta_1}, {phi_1})"})
    return columns


def delta_and_x(mode, events, args):
    """(Delta T in ns before the per-run correction, {x name: x}, validity mask)."""
    if mode == "crystals":
        amp_1, amp_2 = events["amp_1"], events["amp_2"]
        delta = NS_PER_DIGITIZER_UNIT * events["clk"] * (events["time_1"] - events["time_2"])
        with np.errstate(divide="ignore", invalid="ignore"):
            x_value = np.sqrt(2. / ((args.sigma_noise / amp_1) ** 2 + (args.sigma_noise / amp_2) ** 2))
        valid = (amp_1 > 0) & (amp_2 > 0) & np.isfinite(delta)
        return delta, {"aeff_over_sigman": x_value}, valid

    mcp_0, mcp_1 = events["mcp_amp_0"], events["mcp_amp_1"]
    mcp_ok = (mcp_0 > args.mcp_min) & (mcp_1 > args.mcp_min)
    with np.errstate(divide="ignore", invalid="ignore"):
        mcp_aeff = 2. / (1. / mcp_0 + 1. / mcp_1)
    if mode == "mcp-mcp":
        delta = NS_PER_DIGITIZER_UNIT * (events["mcp_time_0"] - events["mcp_time_1"])
        return delta, {"mcp_aeff": mcp_aeff}, mcp_ok & np.isfinite(delta)

    # ecal-mcp: raw difference in digitizer units, folded later run by run
    mcp_time = 0.5 * (events["mcp_time_0"] + events["mcp_time_1"])
    raw = events["time_1"] * events["clk"] + events["phase"] - mcp_time
    with np.errstate(divide="ignore", invalid="ignore"):
        x_ecal = events["amp_1"] / args.sigma_noise
    valid = mcp_ok & (events["amp_1"] > 0) & np.isfinite(raw)
    return raw, {"aecal_over_sigman": x_ecal, "mcp_aeff": mcp_aeff}, valid


def correct_per_run(mode, delta, run, clk, selected):
    """Subtract the per-run centre of Delta T. Returns (corrected in ns, rows)."""
    corrected = np.full(len(delta), np.nan)
    rows = []
    for this_run in np.unique(run[selected]):
        in_run = selected & (run == this_run)
        if mode == "ecal-mcp":
            period = clk[in_run]
            angle = 2 * np.pi * delta[in_run] / period
            offset_angle = math.atan2(np.sin(angle).mean(), np.cos(angle).mean())
            shifted = delta[in_run] - offset_angle * period / (2 * np.pi)
            folded = np.mod(shifted + 0.5 * period, period) - 0.5 * period
            corrected[in_run] = NS_PER_DIGITIZER_UNIT * folded
            offset = NS_PER_DIGITIZER_UNIT * offset_angle * np.median(period) / (2 * np.pi)
        else:
            offset = float(np.median(delta[in_run]))
            corrected[in_run] = delta[in_run] - offset
        rows.append(dict(run=int(this_run), n_events=int(in_run.sum()), offset_ns=float(offset),
                         rms_ns=float(np.std(corrected[in_run]))))
    return corrected, rows


# ------------------------------------------------------------------ histograms and fits
def variable_edges(x_values, per_bin):
    """Edges with per_bin events each (var_bins.C): a last bin with fewer than half of
    per_bin events is merged into the previous one."""
    ordered = np.sort(x_values)
    count = len(ordered)
    edges = [float(ordered[0])]
    for index in range(per_bin, count, per_bin):
        edge = 0.5 * (ordered[index - 1] + ordered[index])
        if edge > edges[-1]:
            edges.append(float(edge))
    if count % per_bin and (count % per_bin) < per_bin / 2 and len(edges) > 1:
        edges.pop()
    edges.append(float(np.nextafter(ordered[-1], np.inf)))
    return np.array(edges)


def fill_th2(name, title, x_values, y_values, edges):
    n_y = int(round(2 * Y_RANGE_NS / Y_BIN_NS))
    histogram = ROOT.TH2D(name, title, len(edges) - 1, edges, n_y, -Y_RANGE_NS, Y_RANGE_NS)
    x_values = np.ascontiguousarray(x_values, float)
    y_values = np.ascontiguousarray(y_values, float)
    histogram.FillN(len(x_values), x_values, y_values, np.ones(len(x_values)))
    return histogram


def fit_slice(projection):
    """Gaussian in +- SLICE_FIT_SIGMAS sigma, re-centred SLICE_FIT_ROUNDS times."""
    if projection.GetEntries() < SLICE_MIN_ENTRIES:
        return None
    mean, sigma = projection.GetMean(), projection.GetStdDev()
    function = ROOT.TF1(common.unique_name("slice_gaus"), "gaus", -Y_RANGE_NS, Y_RANGE_NS)
    result = None
    for _ in range(SLICE_FIT_ROUNDS):
        low, high = mean - SLICE_FIT_SIGMAS * sigma, mean + SLICE_FIT_SIGMAS * sigma
        function.SetRange(low, high)
        function.SetParameters(projection.GetMaximum(), mean, sigma)
        result = projection.Fit(function, "QRS0")
        if result.Status() != 0 or function.GetParameter(2) <= 0:
            return None
        mean, sigma = function.GetParameter(1), abs(function.GetParameter(2))
    projection.GetListOfFunctions().Add(function)
    return dict(mean=mean, sigma=sigma, err_sigma=function.GetParError(2),
                chi2=function.GetChisquare(), ndf=function.GetNDF())


def resolution_curve(name, low, high):
    """sigma = N/x (+) C, N in ns (x dimensionless or ADC), C in ns."""
    function = ROOT.TF1(name, "sqrt(pow([0]/x, 2) + pow([1], 2))", low, high)
    function.SetParNames("N", "C")
    function.SetParameters(15., 0.03)
    return function


# ------------------------------------------------------------------ one mode
def analyse_mode(mode, files, args):
    outdir = os.path.join(args.outdir, mode)
    os.makedirs(outdir, exist_ok=True)
    dropped, kept_only = runsets.resolve(args.runset, args.exclude_runs)
    selection_rows, run_rows = [], []
    pooled = {}                       # x name -> (x list, y list)
    for point in points_of(mode, files, args.good_runs):
        energy, crystals, centre = point["energy"], point["crystals"], point["centre"]
        matrix_eta, matrix_phi = crystals[1]
        print(f"[{mode} {energy:>4} GeV] crystals {crystals} centre {centre}, "
              f"{len(point['paths'])} runs", flush=True)
        events = common.read_events(point["paths"], matrix_eta, matrix_phi, "a3x3",
                                    extra=time_columns(mode, crystals))
        point_args = SimpleNamespace(**vars(args))
        point_args.outdir = outdir
        window_row, cuts = selection.select_point(events, RESISTANCE_2025, energy, dropped,
                                                  kept_only, point_args)
        if cuts is not None:
            position = cuts["nominal"]
            hodoscope_used = 1
        elif args.hodoscope == "required":
            print("    point dropped: no hodoscope window and --hodoscope required")
            continue
        else:
            position = ((events["A_tot"] > common.A_TOT_MIN)
                        & common.runset_mask(events["run"], dropped, kept_only))
            hodoscope_used = 0
            print("    hodoscope window not available: base cut only before the centroid cut")
        centroid = selection.centroid_mask(events, centre[0], centre[1], args.centroid_half)
        delta, x_values, valid = delta_and_x(mode, events, args)
        selected = position & centroid & valid
        corrected, rows = correct_per_run(mode, delta, events["run"], events["clk"], selected)
        for row in rows:
            row.update(energy=energy)
        run_rows += rows
        selection_rows.append(dict(energy=energy, crystals=f"{crystals[0]}/{crystals[1]}",
                                   centre_eta=centre[0], centre_phi=centre[1],
                                   n_runs=len(point["paths"]), n_events=len(delta),
                                   n_base=window_row["n_base"], hodoscope=hodoscope_used,
                                   n_position=int(position.sum()),
                                   n_centroid=int((position & centroid).sum()),
                                   n_selected=int(selected.sum()),
                                   window_reason=window_row.get("reason", "")))
        print(f"    {window_row['n_base']} base, {int(position.sum())} position, "
              f"{int((position & centroid).sum())} centroid, {int(selected.sum())} selected", flush=True)
        for x_name, x_array in x_values.items():
            store = pooled.setdefault(x_name, ([], []))
            store[0].append(x_array[selected])
            store[1].append(corrected[selected])

    common.write_csv(os.path.join(outdir, f"{mode}_selection.csv"), selection_rows,
                     ("energy", "crystals", "centre_eta", "centre_phi", "n_runs", "n_events",
                      "n_base", "hodoscope", "n_position", "n_centroid", "n_selected",
                      "window_reason"))
    common.write_csv(os.path.join(outdir, f"{mode}_per_run.csv"), run_rows,
                     ("run", "energy", "n_events", "offset_ns", "rms_ns"))
    for x_name, (x_parts, y_parts) in pooled.items():
        if x_parts:
            resolution_vs_x(mode, x_name, np.concatenate(x_parts), np.concatenate(y_parts),
                            outdir, args)


X_TITLE = {"aeff_over_sigman": "#sqrt{2/((#sigma_{n}/A_{1})^{2}+(#sigma_{n}/A_{2})^{2})}",
           "aecal_over_sigman": "A_{ECAL}/#sigma_{n}",
           "mcp_aeff": "2/(1/A_{MCP0}+1/A_{MCP1}) [ADC]"}
Y_TITLE = {"crystals": "#sigma_{#DeltaT}/#sqrt{2} [ns]", "mcp-mcp": "#sigma_{#DeltaT}/#sqrt{2} [ns]",
           "ecal-mcp": "#sigma_{t_{ECAL}-t_{MCP}} [ns]"}


def resolution_vs_x(mode, x_name, x_values, y_values, outdir, args):
    tag = f"{mode}_{x_name}"
    finite = np.isfinite(x_values) & np.isfinite(y_values) & (np.abs(y_values) < Y_RANGE_NS)
    x_values, y_values = x_values[finite], y_values[finite]
    if len(x_values) < args.per_bin:
        print(f"    {tag}: only {len(x_values)} events, nothing to bin")
        return
    edges = variable_edges(x_values, args.per_bin)
    print(f"    {tag}: {len(x_values)} events -> {len(edges) - 1} x bins", flush=True)
    th2 = fill_th2(f"h2_{tag}", f";{X_TITLE[x_name]};#DeltaT [ns]", x_values, y_values, edges)
    th2_file = ROOT.TFile(os.path.join(outdir, f"{tag}_th2.root"), "RECREATE")
    th2.Write()
    th2_file.Close()

    scale = 1. / math.sqrt(2.) if mode in ("crystals", "mcp-mcp") else 1.
    projection_file = ROOT.TFile(os.path.join(outdir, f"{tag}_projections.root"), "RECREATE")
    rows = []
    for index in range(1, th2.GetNbinsX() + 1):
        projection = th2.ProjectionY(f"py_{tag}_{index}", index, index)
        low, high = th2.GetXaxis().GetBinLowEdge(index), th2.GetXaxis().GetBinUpEdge(index)
        projection.SetTitle(f"{tag} bin {index}: x in [{low:.4g}, {high:.4g}];#DeltaT [ns];events")
        fit = fit_slice(projection)
        projection_file.cd()
        projection.Write()
        if fit is None:
            continue
        in_bin = (x_values >= low) & (x_values < high)
        rows.append(dict(bin=index, x_low=low, x_high=high, x_mean=float(x_values[in_bin].mean()),
                         x_rms=float(x_values[in_bin].std()), n_events=int(in_bin.sum()),
                         mean_ns=fit["mean"], sigma_ns=fit["sigma"] * scale,
                         err_sigma_ns=fit["err_sigma"] * scale, chi2=fit["chi2"], ndf=fit["ndf"]))
    projection_file.Close()
    if len(rows) < 3:
        print(f"    {tag}: {len(rows)} usable slices, no resolution fit")
        return

    graph = common.make_graph([row["x_mean"] for row in rows], [row["sigma_ns"] for row in rows],
                              [row["err_sigma_ns"] for row in rows])
    graph.SetName(f"g_{tag}")
    graph.SetTitle(f";{X_TITLE[x_name]};{Y_TITLE[mode]}")
    curve = resolution_curve(f"f_{tag}", rows[0]["x_low"], rows[-1]["x_high"])
    result = common.fit_graph(graph, curve)
    noise, constant = result["values"]
    err_noise, err_constant = result["errors"]
    print(f"    {tag}: N = {noise:.3f} +- {err_noise:.3f} ns, C = {1000 * constant:.2f} +- "
          f"{1000 * err_constant:.2f} ps, chi2/ndf = {result['chi2']:.1f}/{result['ndf']}", flush=True)
    for row in rows:
        row.update(N_ns=noise, err_N_ns=err_noise, C_ps=1000 * constant,
                   err_C_ps=1000 * err_constant, fit_chi2=result["chi2"], fit_ndf=result["ndf"])
    common.write_csv(os.path.join(outdir, f"{tag}_sigma.csv"), rows,
                     ("bin", "x_low", "x_high", "x_mean", "x_rms", "n_events", "mean_ns", "sigma_ns",
                      "err_sigma_ns", "chi2", "ndf", "N_ns", "err_N_ns", "C_ps", "err_C_ps",
                      "fit_chi2", "fit_ndf"))
    draw_resolution(tag, graph, curve, result, th2, outdir)


def draw_resolution(tag, graph, curve, result, th2, outdir):
    canvas = ROOT.TCanvas(f"c_{tag}", tag, 1000, 650)
    canvas.SetLeftMargin(0.13)
    canvas.SetBottomMargin(0.14)
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(0.7)
    graph.SetMarkerColor(ROOT.kBlue + 1)
    graph.SetLineColor(ROOT.kBlue + 1)
    graph.Draw("AP")
    curve.SetLineColor(ROOT.kRed)
    curve.SetLineWidth(2)
    curve.Draw("same")
    noise, constant = result["values"]
    err_noise, err_constant = result["errors"]
    box = common.keep(common.text_box(
        [f"N = {noise:.2f} #pm {err_noise:.2f} ns", f"C = {1000 * constant:.1f} #pm {1000 * err_constant:.1f} ps",
         f"#chi^{{2}}/ndf = {result['chi2']:.0f}/{result['ndf']}"], 0.58, 0.68, 0.88, 0.88, size=0.035))
    box.Draw()
    output = ROOT.TFile(os.path.join(outdir, f"{tag}_sigma.root"), "RECREATE")
    graph.Write()
    curve.Write()
    canvas.Write()
    output.Close()
    for extension in ("png", "pdf"):
        common.save_canvas(canvas, os.path.join(outdir, f"{tag}_sigma.{extension}"))
    canvas_2d = ROOT.TCanvas(f"c2_{tag}", tag, 1000, 650)
    canvas_2d.SetLogz()
    canvas_2d.SetRightMargin(0.13)
    th2.GetYaxis().SetRangeUser(-1.5, 1.5)
    th2.Draw("colz")
    common.save_canvas(canvas_2d, os.path.join(outdir, f"{tag}_th2.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="directory of the per-run <run>_<E>_reco.root")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mode", nargs="+", choices=("crystals", "mcp-mcp", "ecal-mcp"),
                        default=["crystals", "mcp-mcp", "ecal-mcp"])
    parser.add_argument("--good-runs", default=os.path.join(common.HERE, "bookkeeping2025",
                                                            "good_run_list_2025.csv"))
    parser.add_argument("--per-bin", type=int, default=2000, help="events per x bin")
    parser.add_argument("--sigma-noise", type=float, default=2.5, help="ECAL noise sigma_n [ADC]")
    parser.add_argument("--mcp-min", type=float, default=60., help="cut on both mcp_A")
    parser.add_argument("--centroid-half", type=float, default=0.2)
    parser.add_argument("--hodoscope", choices=("optional", "required"), default="optional",
                        help="optional: when the parabola gives no window the point keeps the "
                             "base and centroid cuts only (column 'hodoscope' = 0)")
    parser.add_argument("--half", type=float, default=4., help="hodoscope half window [mm]")
    parser.add_argument("--yplane", choices=("y1", "y2"), default="y1")
    parser.add_argument("--fallback-file", default="fallback_none.py",
                        help="hand-set hodoscope vertices; the 2025 ones belong to other runs")
    parser.add_argument("--exclude-runs", nargs="*", type=int, default=[])
    runsets.add_argument(parser)
    args = parser.parse_args()

    common.style()
    files = run_files(args.base)
    print(len(files), "run files in", args.base)
    for mode in args.mode:
        analyse_mode(mode, files, args)


if __name__ == "__main__":
    main()
