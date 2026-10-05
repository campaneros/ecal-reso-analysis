#!/usr/bin/env python3
"""
Time resolution, after the selection shared with the energy resolution (selection.py).

Modes
  crystals  Delta T between the two crystals the beam was aimed between (runs with
            "5/6" in the eta or phi column of the good-run list). The hodoscope
            parabola profiles the 3x3 sum rebuilt from A, not the seed-based A_tot,
            so that the response is flat across the boundary; in the coordinate where
            it is flat (no maximum) the window is centred on the boundary, the zero of
            (A_1 - A_2)/(A_1 + A_2) (pol1 fit). x = A_eff / sigma_n =
            sqrt(2 / ((sigma_n/A_1)^2 + (sigma_n/A_2)^2)), y = sigma(Delta T)/sqrt(2).
  mcp-mcp   Delta T between the two MCPs, both with mcp_A > --mcp-min;
            x = 2 / (1/mcp_A[0] + 1/mcp_A[1]), y = sigma(Delta T)/sqrt(2).
  ecal-mcp  ECAL time of the target crystal minus the MCP time (mean of the two MCPs,
            both above --mcp-min), folded on the clock period as in var_bins.C; the
            fold offset is the circular mean of each run instead of a fixed constant.
            Two sets of outputs: x = A_ecal / sigma_n and x = MCP A_eff. y = sigma.

Chain, for every mode
  0. runs: every run of the good-run list with a reco file; "a/b" rows go to the
     crystals mode, single-crystal rows to the MCP modes, keeping only the runs with
     both MCPs alive (MCP_MIN_FRACTION); points = (crystals, table position, energy);
  1. per point: the selection of the energy resolution (selection.py): A_tot cut,
     run selection, hodoscope window from the parabola study; a point without a
     window is skipped, as in fit_dcb_per_run.py;
  2. per run and per gain state (gs of the crystals: the low gain moves the time by
     ~1.5 ns): the median (circular mean for ecal-mcp) of Delta T is subtracted;
  3. all runs and energies of the same crystal(s) together: TH2 of Delta T against x with variable x bins of
     --per-bin events each (as var_bins.C)                       -> <mode>_<crystals>_<x>_th2.root
  4. Y projection of every x bin, gaussian fit in +- 2 sigma     -> ..._projections.root
  5. sigma against x, fitted with sqrt((N/x)^2 + C^2)            -> ..._sigma.root/.png/.csv

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
# a run enters the MCP modes when at least this fraction of its events has both MCPs
# above --mcp-min (runs without MCP, or with the MCPs off, have ~0-30 %)
MCP_MIN_FRACTION = 0.5
MIN_EVENTS_PER_GROUP = 50         # events of a (run, gain state) needed to measure its offset
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


def has_events(path):
    """False for the reco files with no event (runs stopped at the start, e.g. 19702)."""
    handle = ROOT.TFile.Open(path)
    tree = handle.Get("h4_reco") if handle and not handle.IsZombie() else None
    entries = tree.GetEntries() if tree else 0
    if handle:
        handle.Close()
    return entries > 0


def good_runs(good_runs_csv):
    """{run: dict(crystals, table)} from the good-run list. crystals is the pair the beam
    was aimed between ("a/b" in the eta or phi column) or the same crystal twice."""
    runs = {}
    with open(good_runs_csv) as handle:
        for row in csv.DictReader(handle):
            try:
                etas = [int(value) for value in row["eta"].strip().split("/")]
                phis = [int(value) for value in row["phi"].strip().split("/")]
            except ValueError:
                continue
            etas, phis = etas * (2 // len(etas)), phis * (2 // len(phis))
            runs[int(row["Run"])] = dict(crystals=((etas[0], phis[0]), (etas[1], phis[1])),
                                         table=(row["Table X"].strip(), row["Table Y"].strip()))
    return runs


def crystal_label(crystals):
    (eta_1, phi_1), (eta_2, phi_2) = crystals
    if crystals[0] == crystals[1]:
        return f"eta{eta_1}_phi{phi_1}"
    eta = f"{eta_1}" if eta_1 == eta_2 else f"{eta_1}-{eta_2}"
    phi = f"{phi_1}" if phi_1 == phi_2 else f"{phi_1}-{phi_2}"
    return f"eta{eta}_phi{phi}"


def points_of(mode, files, good_runs_csv):
    """One point per (crystals, table position, energy): the hodoscope moves with the
    table, so runs at different positions get their own window. The two-crystal runs
    feed the crystals mode, the single-crystal runs the two MCP modes."""
    grouped = {}
    for run, info in good_runs(good_runs_csv).items():
        crystals = info["crystals"]
        if run not in files or (mode == "crystals") != (crystals[0] != crystals[1]):
            continue
        energy, path = files[run]
        grouped.setdefault((crystals, info["table"], energy), []).append((run, path))
    return [dict(energy=energy, crystals=crystals, table=table, label=crystal_label(crystals),
                 runs=[run for run, _path in sorted(members)],
                 paths=[path for _run, path in sorted(members)])
            for (crystals, table, energy), members in sorted(grouped.items())]


ASYMMETRY_BIN_MM, ASYMMETRY_FIT_RANGE, ASYMMETRY_MIN_SPAN, ASYMMETRY_MIN_PER_BIN = 0.5, 0.5, 0.3, 100


def boundary_vertices(events, base):
    """{coordinate: hodoscope position where A_1 = A_2}, for the coordinates in which
    the sharing between the two crystals changes sign (the other one is left out)."""
    hodo_x, hodo_y = common.hodoscope_xy(events)
    with np.errstate(divide="ignore", invalid="ignore"):
        asymmetry = (events["amp_1"] - events["amp_2"]) / (events["amp_1"] + events["amp_2"])
    vertices = {}
    for name, coordinate in (("x", hodo_x), ("y", hodo_y)):
        usable = base & np.isfinite(coordinate) & np.isfinite(asymmetry)
        if usable.sum() < 1000:
            continue
        edges = np.arange(np.percentile(coordinate[usable], 1), np.percentile(coordinate[usable], 99),
                          ASYMMETRY_BIN_MM)
        centres, means, errors = [], [], []
        for low in edges:
            in_bin = usable & (coordinate >= low) & (coordinate < low + ASYMMETRY_BIN_MM)
            if in_bin.sum() < ASYMMETRY_MIN_PER_BIN:
                continue
            centres.append(coordinate[in_bin].mean())
            means.append(asymmetry[in_bin].mean())
            errors.append(asymmetry[in_bin].std() / math.sqrt(in_bin.sum()))
        means = np.array(means)
        if len(means) < 4 or means.max() < ASYMMETRY_MIN_SPAN or means.min() > -ASYMMETRY_MIN_SPAN:
            continue
        near = np.abs(means) < ASYMMETRY_FIT_RANGE
        if near.sum() < 3:
            continue
        graph = common.make_graph(np.array(centres)[near], means[near], np.array(errors)[near])
        line = ROOT.TF1(common.unique_name("asymmetry"), "pol1")
        graph.Fit(line, "QN")
        if line.GetParameter(1) != 0:
            vertices[name] = -line.GetParameter(0) / line.GetParameter(1)
    return vertices


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
            "time_2": f"pipeline_crystal_value(t, sel_ieta, sel_iphi, {eta_2}, {phi_2})",
            "gain_1": f"pipeline_crystal_value(gs, sel_ieta, sel_iphi, {eta_1}, {phi_1})",
            "gain_2": f"pipeline_crystal_value(gs, sel_ieta, sel_iphi, {eta_2}, {phi_2})"})
        return columns
    columns.update({"mcp_amp_0": "(double)mcp_A[0]", "mcp_amp_1": "(double)mcp_A[1]",
                    "mcp_time_0": "(double)mcp_t[0]", "mcp_time_1": "(double)mcp_t[1]"})
    if mode == "ecal-mcp":
        columns.update({
            "phase": "(double)clk_phase",
            "amp_1": f"pipeline_crystal_value(A, sel_ieta, sel_iphi, {eta_1}, {phi_1})",
            "time_1": f"pipeline_crystal_value(t, sel_ieta, sel_iphi, {eta_1}, {phi_1})",
            "gain_1": f"pipeline_crystal_value(gs, sel_ieta, sel_iphi, {eta_1}, {phi_1})"})
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


def gain_state(mode, events):
    """0 = every crystal used in high gain; the gain switch (gs = 1) moves the time of
    a crystal by ~1.5 ns, so every combination gets its own offset.
    crystals: 2*gs_1 + gs_2; ecal-mcp: gs of the crystal; mcp-mcp: 0."""
    if mode == "crystals":
        return (2 * np.nan_to_num(events["gain_1"]) + np.nan_to_num(events["gain_2"])).astype(int)
    if mode == "ecal-mcp":
        return np.nan_to_num(events["gain_1"]).astype(int)
    return np.zeros(len(events["run"]), int)


def correct_per_run(mode, delta, run, clk, selected, state=None):
    """Subtract the centre of Delta T of every (run, gain state). Groups with fewer
    than MIN_EVENTS_PER_GROUP events are left out (NaN). Returns (corrected in ns, rows)."""
    corrected = np.full(len(delta), np.nan)
    state = np.zeros(len(delta), int) if state is None else state
    rows = []
    for this_run, this_state in sorted(set(zip(run[selected].tolist(), state[selected].tolist()))):
        in_run = selected & (run == this_run) & (state == this_state)
        if in_run.sum() < MIN_EVENTS_PER_GROUP:
            rows.append(dict(run=int(this_run), gain_state=int(this_state),
                             n_events=int(in_run.sum()), offset_ns=np.nan, rms_ns=np.nan))
            continue
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
        rows.append(dict(run=int(this_run), gain_state=int(this_state),
                         n_events=int(in_run.sum()), offset_ns=float(offset),
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
        if not result.Get() or result.Status() != 0 or function.GetParameter(2) <= 0:
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
def mcp_live_runs(events, base, mcp_min):
    """Runs where at least MCP_MIN_FRACTION of the base events have both MCPs above
    mcp_min, and the fraction of every run."""
    both = (events["mcp_amp_0"] > mcp_min) & (events["mcp_amp_1"] > mcp_min)
    fractions = {int(run): float(both[base & (events["run"] == run)].mean())
                 for run in np.unique(events["run"][base])}
    return [run for run, fraction in fractions.items() if fraction >= MCP_MIN_FRACTION], fractions


def analyse_point(mode, point, args, outdir, dropped, kept_only):
    """Selection, Delta T and per-run correction of one point. Returns (selection row,
    window row, per-run rows, {x name: (x, corrected Delta T)}) or the rows only when
    the point is skipped."""
    energy, crystals, label = point["energy"], point["crystals"], point["label"]
    matrix_eta, matrix_phi = crystals[1]
    paths = [path for path in point["paths"] if has_events(path)]
    empty = sorted(set(point["paths"]) - set(paths))
    if empty:
        print(f"    empty files left out: {[os.path.basename(path) for path in empty]}")
    if not paths:
        return dict(label=label, energy=energy, runs=" ".join(map(str, point["runs"])),
                    reason="no event in any file"), None, [], {}
    events = common.read_events(paths, matrix_eta, matrix_phi, "a3x3",
                                extra=time_columns(mode, crystals))
    base = ((events["A_tot"] > common.A_TOT_MIN)
            & common.runset_mask(events["run"], dropped, kept_only))
    row = dict(label=label, energy=energy, table=f"{point['table'][0]}/{point['table'][1]}",
               runs=" ".join(str(run) for run in point["runs"]), n_events=len(events["run"]),
               n_base=int(base.sum()), n_hodoscope=0, n_selected=0, runs_dropped="", reason="")
    usable = base
    if mode != "crystals":
        live, fractions = mcp_live_runs(events, base, args.mcp_min)
        dead = [run for run in fractions if run not in live]
        row["runs_dropped"] = " ".join(f"{run}(mcp {100 * fractions[run]:.0f}%)" for run in dead)
        usable = base & np.isin(events["run"], live)
        if not live:
            row["reason"] = "no run with both MCPs alive"
            return row, None, [], {}

    point_args = SimpleNamespace(**vars(args))
    point_args.outdir = os.path.join(outdir, "windows", f"{label}_table{point['table'][0]}-{point['table'][1]}")
    response, flat_vertices = None, None
    if mode == "crystals":
        response = events["amplitude"]                        # 3x3 sum, not A_tot
        flat_vertices = boundary_vertices(events, base)
        print(f"    boundary between the crystals on the hodoscope: {flat_vertices}")
    window_row, cuts = selection.select_point(events, RESISTANCE_2025, energy, dropped, kept_only,
                                              point_args, response=response,
                                              vertex_when_flat=flat_vertices)
    window_row.update(label=label, table=row["table"])
    if cuts is None:
        row["reason"] = window_row.get("reason", "")
        return row, window_row, [], {}
    position = cuts["nominal"] & usable
    delta, x_values, valid = delta_and_x(mode, events, args)
    selected = position & valid
    corrected, run_rows = correct_per_run(mode, delta, events["run"], events["clk"], selected,
                                          gain_state(mode, events))
    for run_row in run_rows:
        run_row.update(energy=energy, label=label)
    row.update(n_hodoscope=int(position.sum()), n_selected=int(selected.sum()))
    print(f"    {row['n_base']} base, {row['n_hodoscope']} in the hodoscope window, "
          f"{row['n_selected']} selected" + (f"; dropped {row['runs_dropped']}" if row["runs_dropped"] else ""),
          flush=True)
    pooled = {x_name: (x_array[selected], corrected[selected]) for x_name, x_array in x_values.items()}
    return row, window_row, run_rows, pooled


def analyse_mode(mode, files, args):
    outdir = os.path.join(args.outdir, mode)
    os.makedirs(outdir, exist_ok=True)
    dropped, kept_only = runsets.resolve(args.runset, args.exclude_runs)
    selection_rows, window_rows, run_rows = [], [], []
    pooled = {}                       # (crystal label, x name) -> (x list, y list)
    for point in points_of(mode, files, args.good_runs):
        print(f"[{mode} {point['label']} table {point['table']} {point['energy']:>4} GeV] "
              f"runs {point['runs']}", flush=True)
        try:
            row, window_row, rows, point_pooled = analyse_point(mode, point, args, outdir,
                                                                dropped, kept_only)
        except Exception as error:               # one bad file must not stop the night run
            print(f"    FAILED: {type(error).__name__}: {error}", flush=True)
            selection_rows.append(dict(label=point["label"], energy=point["energy"],
                                       runs=" ".join(map(str, point["runs"])),
                                       reason=f"{type(error).__name__}: {error}"))
            continue
        selection_rows.append(row)
        if window_row:
            window_rows.append(window_row)
        run_rows += rows
        for x_name, (x_array, y_array) in point_pooled.items():
            store = pooled.setdefault((point["label"], x_name), ([], []))
            store[0].append(x_array)
            store[1].append(y_array)

    common.write_csv(os.path.join(outdir, f"{mode}_selection.csv"), selection_rows,
                     ("label", "energy", "table", "runs", "n_events", "n_base", "n_hodoscope",
                      "n_selected", "runs_dropped", "reason"))
    common.write_csv(os.path.join(outdir, f"{mode}_windows.csv"), window_rows,
                     ("label", "table", "resistance", "energy", "window", "n_base", "n_selected",
                      "skipped", "reason", "x_lo", "x_hi", "y_lo", "y_hi", "x_vertex", "y_vertex",
                      "x_ok", "y_ok", "x_why", "y_why", "fallback"))
    common.write_csv(os.path.join(outdir, f"{mode}_per_run.csv"), run_rows,
                     ("label", "run", "energy", "gain_state", "n_events", "offset_ns", "rms_ns"))
    for (label, x_name), (x_parts, y_parts) in sorted(pooled.items()):
        resolution_vs_x(mode, label, x_name, np.concatenate(x_parts), np.concatenate(y_parts),
                        outdir, args)


X_TITLE = {"aeff_over_sigman": "#sqrt{2/((#sigma_{n}/A_{1})^{2}+(#sigma_{n}/A_{2})^{2})}",
           "aecal_over_sigman": "A_{ECAL}/#sigma_{n}",
           "mcp_aeff": "2/(1/A_{MCP0}+1/A_{MCP1}) [ADC]"}
Y_TITLE = {"crystals": "#sigma_{#DeltaT}/#sqrt{2} [ns]", "mcp-mcp": "#sigma_{#DeltaT}/#sqrt{2} [ns]",
           "ecal-mcp": "#sigma_{t_{ECAL}-t_{MCP}} [ns]"}


def resolution_vs_x(mode, label, x_name, x_values, y_values, outdir, args):
    tag = f"{mode}_{label}_{x_name}"
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
