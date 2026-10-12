FILTER_50MHZ = tuple(range(20552, 20600)) + tuple(range(20486,20501))
HIGH_275 = tuple(range(20652, 20660))
OUTLIERS = (20491,20788,21116)


def resolve(runset, extra_exclude=()):
    drop = list(extra_exclude)
    only = []
    if runset == "standard":
        drop += list(FILTER_50MHZ) + list(HIGH_275) + list(OUTLIERS)
    elif runset == "filter50":
        only = list(FILTER_50MHZ)
    elif runset != "all":
        raise ValueError(f"unknown runset: {runset}")
    return drop, only


def add_argument(ap):
    ap.add_argument("--runset", choices=("standard", "filter50", "all"),
                    default="standard",
                    help="which readout condition to analyse: 'standard' drops the "
                         "50 MHz runs and the 275 GeV high population, 'filter50' "
                         "keeps only the 50 MHz runs, 'all' filters nothing beyond "
                         "--exclude-runs")
