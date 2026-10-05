"""
The event selection shared by the energy and the time resolution.

  base cut     A_tot > A_TOT_MIN and the run selection of runsets
  hodoscope    window around the vertex of the response parabola on the hodoscope
               (hodoscope_window.py), with its four +- 1 mm variations
  centroid     |pos_eta - X| < half and |pos_phi - Y| < half around the point where the
               beam was aimed; X or Y ends in .5 when the beam sat between two crystals

select_point() applies the first two and records the diagnostics of the parabola
study; fit_dcb_per_run.py (energy) and time_resolution.py (time) both start from it.
"""

import numpy as np

import common
import hodoscope_window

MIN_EVENTS_IN_WINDOW = common.MIN_EVENTS_POOLED


def centroid_mask(events, centre_eta, centre_phi, half=0.2):
    return ((np.abs(events["pos_eta"] - centre_eta) < half)
            & (np.abs(events["pos_phi"] - centre_phi) < half))


def select_point(events, resistance, energy, dropped, kept_only, args):
    """Base cut and hodoscope window of one (resistance, energy) point.

    Returns (window_row, cuts): cuts is the dict of masks of
    hodoscope_window.window_masks (nominal + the four shifted windows), or None when
    the point is skipped, with the reason in window_row.
    """
    base = (events["A_tot"] > common.A_TOT_MIN) & common.runset_mask(events["run"], dropped,
                                                                      kept_only)
    window_row = dict(resistance=resistance, energy=energy,
                      energy_true=common.true_energy(energy),
                      n_base=int(base.sum()), skipped=0, reason="", fallback="")
    if base.sum() == 0:
        window_row.update(skipped=1, n_selected=0, reason="no event passes the base cut")
        return window_row, None

    hodo_x, hodo_y = common.hodoscope_xy(events, args.yplane)
    info = hodoscope_window.hodoscope_windows(hodo_x, hodo_y, events["A_tot"], base, resistance,
                                              energy, args.half, args.outdir, args.fallback_file)
    for coordinate in ("x", "y"):
        scan = info["scan"][coordinate]
        window_row.update({f"{coordinate}_vertex": scan["vertex"],
                           f"{coordinate}_width": scan["width"],
                           f"{coordinate}_ok": int(scan["ok"]),
                           f"{coordinate}_why": info["why"][coordinate]})
    window_row["fallback"] = "+".join(info["fallback"])
    window_row["window"] = hodoscope_window.window_label(info["fallback"])
    for coordinate in info["fallback"]:
        reason = info["why"][coordinate]
        print(f"    {coordinate}: hand-set vertex of resolution_hodo.py"
              + (f" (scan failed: {reason})" if reason else " (overrides a successful scan)"))

    missing = [coordinate for coordinate in ("x", "y") if info["windows"][coordinate] is None]
    if missing:
        reason = "; ".join(f"{coordinate}: {info['why'][coordinate]}" for coordinate in missing)
        print(f"    SKIPPED, no window in {'+'.join(missing)} ({reason})")
        window_row.update(skipped=1, reason=reason, n_selected=0)
        return window_row, None

    window_x, window_y = info["windows"]["x"], info["windows"]["y"]
    window_row.update(x_lo=window_x[0], x_hi=window_x[1], y_lo=window_y[0], y_hi=window_y[1])
    cuts = hodoscope_window.window_masks(hodo_x, hodo_y, base, window_x, window_y)
    n_selected = int(cuts["nominal"].sum())
    window_row["n_selected"] = n_selected
    if n_selected < MIN_EVENTS_IN_WINDOW:
        print(f"    SKIPPED, only {n_selected} events after the hodoscope cut")
        window_row.update(skipped=1, reason=f"fewer than {MIN_EVENTS_IN_WINDOW} events in the window")
        return window_row, None
    return window_row, cuts
