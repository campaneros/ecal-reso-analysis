#!/usr/bin/env python3
"""
Confronto ROOT vs Python.

Legge root_fit.csv (prodotto da crosscheck_root.sh, con ROOT vero)
e i CSV di drift_dcb_all.py, e stampa la tabella delle differenze.
Se il fit Python riproduce fit.sh + dcb.cxx, le differenze devono essere
compatibili con zero entro gli errori.

Uso:  python3 confronta_root_python.py --plotdir .
"""
import argparse, csv, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plotdir", default=".")
    a = ap.parse_args()

    rf = os.path.join(a.plotdir, "root_fit.csv")
    if not os.path.exists(rf):
        print(f"manca {rf} -- lancia prima ./crosscheck_root.sh")
        return
    root = {}
    for r in csv.DictReader(open(rf)):
        root[(int(r["resistance"]), int(r["energy"]))] = r

    print("=" * 96)
    print("FIT double-CB  (riga ALL dei CSV Python contro ROOT)")
    print(f"{'R':>4} {'E':>4} | {'peak ROOT':>11} {'peak py':>11} {'diff':>9} {'diff/err':>8}"
          f" | {'sig ROOT':>9} {'sig py':>9} {'diff':>8} {'diff/err':>8}")
    worst = 0.
    for R in (340, 400, 500):
        f = os.path.join(a.plotdir, str(R), f"drift_per_run_{R}ohm.csv")
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            if r["run"] != "ALL":
                continue
            k = (R, int(r["energy"]))
            if k not in root:
                continue
            rr = root[k]
            pR, pP = float(rr["peak_abs"]), float(r["peak_abs"])
            sR, sP = float(rr["sigma_abs"]), float(r["sigma_abs"])
            epk = np.hypot(float(rr["err_peak_abs"]), float(r["err_peak_abs"]))
            esg = np.hypot(float(rr["err_sigma_abs"]), float(r["err_sigma_abs"]))
            zp = (pP - pR) / epk if epk else np.nan
            zs = (sP - sR) / esg if esg else np.nan
            worst = max(worst, abs(zs), abs(zp))
            flag = "  <<<" if max(abs(zs), abs(zp)) > 3 else ""
            print(f"{R:>4} {r['energy']:>4} | {pR:11.4f} {pP:11.4f} {pP-pR:+9.4f} {zp:+8.2f}"
                  f" | {sR:9.4f} {sP:9.4f} {sP-sR:+8.4f} {zs:+8.2f}{flag}")
    print(f"\nmassimo scarto in sigma di differenza: {worst:.2f}")
    print("(sotto ~3 vuol dire che le due implementazioni danno lo stesso risultato)")



if __name__ == "__main__":
    main()
