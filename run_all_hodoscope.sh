BASE=/eos/cms/store/group/dpg_ecal/comm_ecal/upgrade/testbeam/ECALTB_H4_Jun2026/Reco/


# 0  prerequisite: the crystal-unit curvature (skip if profili is up to date)
python3 profili_pernorm.py --base $BASE --outdir profili \
    --resistances 340 400 500

# 1  calibration on its own: offsets, W per energy, and the list of points
#    where the parabola scan finds nothing. Useful as a check before the rest
python3 hodoscope_calib.py --base $BASE --outdir hodo_scan --plotdir plot \
    --resistances 340 400 500

# 2  the resolution, once per window definition, into two directories
python3 resolution_hodo.py --base $BASE --outdir hodo_parab --besdir bes \
    --plotdir plot --resistances 340 400 500 --window parabola

# 3  the picture behind each cut (slow, about two minutes per resistance)
python3 hodo_windows.py --base $BASE --outdir hodo_parab --plotdir plot \
    --resistances 340 400 500 --window parabola
