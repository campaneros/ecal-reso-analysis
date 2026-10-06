"""python3 test_time_resolution.py -- checks of the binning, the per-run fold and the run sets."""

import os

import numpy as np

import time_resolution

HERE = os.path.dirname(os.path.abspath(__file__))


def test_bins_built_from_the_top_with_per_bin_events_each():
    x_values = np.random.default_rng(1).uniform(0, 100, 9500)
    counts = np.histogram(x_values, time_resolution.variable_edges(x_values, 3000))[0]
    assert list(counts) == [3500, 3000, 3000]          # the leftover joins the lowest bin


def test_fold_centres_a_distribution_across_the_clock_wrap():
    period, rng = 31.2, np.random.default_rng(2)
    delta = np.mod(30.5 + rng.normal(0, 0.5, 5000), period)
    run = np.repeat([1, 2], 2500)
    corrected, _rows = time_resolution.correct_per_run("ecal-mcp", delta, run, np.full(5000, period),
                                                       np.ones(5000, bool))
    assert abs(np.median(corrected)) < 0.02
    assert abs(np.std(corrected) - 0.5 * time_resolution.NS_PER_DIGITIZER_UNIT) < 0.01


def test_offset_measured_on_high_gain_and_applied_to_every_state():
    rng = np.random.default_rng(3)
    state = np.repeat([0, 1, 2], 1000)
    delta = 0.7 + rng.normal(0, 0.05, 3000) + np.array([0., 1.5, -1.5])[state]
    corrected, rows = time_resolution.correct_per_run("crystals", delta, np.ones(3000, int),
                                                      np.full(3000, 31.2), np.ones(3000, bool), state == 0)
    assert len(rows) == 1
    assert abs(np.median(corrected[state == 0])) < 0.01
    assert abs(np.median(corrected[state == 1]) - 1.5) < 0.01


def test_two_crystal_runs_pair_the_neighbours():
    files = {19402: (60, "a"), 19441: (120, "b"), 19573: (100, "c")}
    points = time_resolution.points_of("crystals", files,
                                       os.path.join(HERE, "bookkeeping2025", "good_run_list_2025.csv"))
    assert [point["crystals"] for point in points] == [((18, 5), (18, 6)), ((53, 5), (53, 6))]
    assert [point["label"] for point in points] == ["eta18_phi5-6", "eta53_phi5-6"]


def test_mcp_mcp_takes_every_run_ecal_mcp_the_single_crystal_ones():
    files = {19402: (60, "a"), 19573: (100, "c"), 19680: (150, "d")}
    csv_path = os.path.join(HERE, "bookkeeping2025", "good_run_list_2025.csv")
    ecal_mcp = time_resolution.points_of("ecal-mcp", files, csv_path)
    assert [(point["label"], point["runs"]) for point in ecal_mcp] == [("eta18_phi6", [19573]),
                                                                       ("eta53_phi6", [19680])]
    mcp_mcp = time_resolution.points_of("mcp-mcp", files, csv_path)
    assert sorted(run for point in mcp_mcp for run in point["runs"]) == [19402, 19573, 19680]
    assert {point["label"] for point in mcp_mcp} == {"all"}


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
