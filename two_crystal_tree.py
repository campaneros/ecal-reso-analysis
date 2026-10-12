import argparse
import csv
import glob
import os

import ROOT

ROOT.gInterpreter.Declare(r'''
template <typename Values, typename Index>
double crystal_value(const Values &values, const Index &ieta, const Index &iphi, int eta, int phi) {
  for (size_t index = 0; index < values.size() && index < ieta.size(); ++index)
    if ((int)ieta[index] == eta && (int)iphi[index] == phi) return (double)values[index];
  return std::nan("");
}

double sum_3x3(const ROOT::RVecD &amplitude, const ROOT::RVec<unsigned short> &ieta,
               const ROOT::RVec<unsigned short> &iphi, int eta, int phi) {
  double total = 0.;
  for (size_t index = 0; index < amplitude.size() && index < ieta.size(); ++index)
    if (std::abs((int)ieta[index] - eta) < 2 && std::abs((int)iphi[index] - phi) < 2) total += amplitude[index];
  return total;
}

double first_cluster(const ROOT::RVecF &positions) {
  return positions.size() ? (double)positions[0] : std::nan("");
}
''')

COLUMNS = ["run", "spill", "evt", "energy", "A_tot", "A_3x3", "A_1", "A_2", "t_1", "t_2",
           "lg_nsample_1", "lg_nsample_2", "fit_status_1", "fit_status_2", "pos_eta", "pos_phi",
           "hodo_x1", "hodo_x2", "hodo_y1", "hodo_y2",
           "hodo_x1_nclusters", "hodo_x2_nclusters", "hodo_y1_nclusters", "hodo_y2_nclusters"]


def two_crystal_runs(good_runs, eta):
    runs = []
    with open(good_runs) as handle:
        for row in csv.DictReader(handle):
            if row["eta"].strip() == str(eta) and row["phi"].strip() == "5/6":
                runs.append(int(row["Run"]))
    return sorted(runs)


def main():
    parser = argparse.ArgumentParser(
        description="Merge the two-crystal runs (phi 5/6) of one eta into a flat tree, without any cut.",
        epilog="example:\n  python3 two_crystal_tree.py --skim-dir skim_dir --eta 18 --out two_crystal_tree_eta18.root",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skim-dir", required=True)
    parser.add_argument("--eta", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--good-runs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "bookkeeping2025", "good_run_list_2025.csv"))
    args = parser.parse_args()

    files = ROOT.std.vector("string")()
    for run in two_crystal_runs(args.good_runs, args.eta):
        for path in glob.glob(os.path.join(args.skim_dir, f"{run}_*_reco.root")):
            files.push_back(path)
    print(files.size(), "files")

    frame = ROOT.RDataFrame("h4_reco", files)
    for number, phi in (("1", 5), ("2", 6)):
        for name in ("A", "t", "lg_nsample", "fit_status"):
            frame = frame.Define(f"{name}_{number}", f"crystal_value({name}, sel_ieta, sel_iphi, {args.eta}, {phi})")
    frame = (frame.Define("A_3x3", f"sum_3x3(A, sel_ieta, sel_iphi, {args.eta}, 6)")
             .Define("hodo_x1", "first_cluster(hodo_x1_pos)").Define("hodo_x2", "first_cluster(hodo_x2_pos)")
             .Define("hodo_y1", "first_cluster(hodo_y1_pos)").Define("hodo_y2", "first_cluster(hodo_y2_pos)"))
    frame.Snapshot("two_crystals", args.out, COLUMNS)


if __name__ == "__main__":
    main()
