"""Distribuzione di A_tot run per run, per vedere se il problema e' fra run o dentro un run."""
import argparse, os
import numpy as np, uproot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETA0, PHI0, SEL, AMIN = 18., 6., 0.2, 100.
CUT = "$|\\mathrm{pos\\_eta}-18| \\leq 0.2$,  $|\\mathrm{pos\\_phi}-6| \\leq 0.2$"

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True); ap.add_argument("--energy", type=int, required=True)
ap.add_argument("--resistance", type=int, required=True); ap.add_argument("--outdir", default="plot/metodi")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)

t = uproot.open(a.file)["h4_reco"]
arr = t.arrays(["run","spill","evt","A_tot","pos_eta","pos_phi"], library="np")
k = ((np.abs(arr["pos_eta"]-ETA0)<=SEL)&(np.abs(arr["pos_phi"]-PHI0)<=SEL)&(arr["A_tot"]>AMIN))
run, at = arr["run"][k], arr["A_tot"][k]
o = np.lexsort((arr["evt"][k], arr["spill"][k], run)); run, at = run[o], at[o]
runs = sorted(np.unique(run))
med = {r: float(np.median(at[run==r])) for r in runs}
lo, hi = 0.93*min(med.values()), 1.06*max(med.values())

fig, axs = plt.subplots(2, 2, figsize=(15, 9))
b = np.linspace(lo, hi, 260)

ax = axs[0,0]
ax.hist(at, bins=b, histtype="step", color="k", lw=1.8, label=f"tutti,  $N={len(at)}$")
for i, r in enumerate(runs):
    v = at[run==r]
    ax.hist(v, bins=b, histtype="step", lw=1.2, color=f"C{i}",
            label=f"run {r},  $N={len(v)}$,  med $={med[r]:.1f}$")
ax.set_xlabel("$A_{tot}$ [ADC]"); ax.set_ylabel("eventi / bin")
ax.set_title(f"{a.resistance} $\\Omega$,  {a.energy} GeV   $\\quad$   "
             f"$A_{{tot}} > {AMIN:.0f}$ ADC,  {CUT}", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axs[0,1]
for i, r in enumerate(runs):
    v = at[run==r]
    ax.hist(v, bins=b, histtype="step", lw=1.4, color=f"C{i}", density=True, label=f"run {r}")
ax.set_xlabel("$A_{tot}$ [ADC]"); ax.set_ylabel("densita' normalizzata")
ax.set_title("ogni run normalizzato a se stesso", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axs[1,0]
for i, r in enumerate(runs):
    v = at[run==r]; s = med[r]
    ax.hist(v/s, bins=np.linspace(0.93, 1.06, 260), histtype="step", lw=1.4,
            color=f"C{i}", density=True, label=f"run {r}")
ax.set_xlabel("$A_{tot} \\, / \\, \\mathrm{mediana}$ del run"); ax.set_ylabel("densita' normalizzata")
ax.set_title("riscalato alla mediana di ogni run", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axs[1,1]
CH = 400
for i, r in enumerate(runs):
    v = at[run==r]; n = max(len(v)//CH, 1)
    x = [j*CH+CH/2 for j in range(n) if len(v[j*CH:(j+1)*CH])>50]
    y = [np.median(v[j*CH:(j+1)*CH]) for j in range(n) if len(v[j*CH:(j+1)*CH])>50]
    off = sum(len(at[run==q]) for q in runs[:i])
    ax.plot([off+xx for xx in x], y, ".-", ms=3, lw=.8, color=f"C{i}", label=f"run {r}")
    ax.axhline(med[r], color=f"C{i}", ls=":", lw=.8)
ax.set_xlabel("evento (ordine cronologico)"); ax.set_ylabel("mediana di $A_{tot}$ su 400 eventi")
ax.set_title("stabilita' dentro ogni run", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=.3)

fig.tight_layout()
p = os.path.join(a.outdir, f"diagnosi_{a.energy}GeV_{a.resistance}ohm.png")
fig.savefig(p, dpi=150); plt.close(fig)
print("->", p)
for r in runs:
    v = at[run==r]
    print(f"  run {r}: N={len(v):>7}  mediana={med[r]:>9.1f}  "
          f"scarto dal primo = {100*(med[r]/med[runs[0]]-1):+.2f}%")
