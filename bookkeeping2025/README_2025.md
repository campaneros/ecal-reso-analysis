# ECAL TB H4 — October 2025

- **Run list**
  - https://gitlab.cern.ch/ecal-daq-upgrade/DANTE/-/blob/dev-2026/logbook_db_2025.txt?ref_type=heads

- **Beam information**
  - Beam element logs:
    - Available in the `beamfiles/` folder.
  - Beamline energy spread details:
    - [CERN document](https://cds.cern.ch/record/702402/files/cer-000414329.pdf)
  - Syncrotron radiation beam energy spread RMS (in %): 1.92e-7* E**(5/2), Fig.4 of the paper in the link above

- **Reconstructed data**
  - Reconstructed files, using a 3×3 matrix:
    - `/eos/cms/store/group/dpg_ecal/comm_ecal/upgrade/testbeam/ECALTB_H4_Oct2025/Reco_v3_fix`
  - Reconstruction software:
    - [DANTE v2026-260819](https://gitlab.cern.ch/ecal-daq-upgrade/DANTE/-/tags/v2026-260819) [to be updated]
  - Template used:
    - From Marc, available on ecalgit lxplus

- **Runs for energy and MCP**
  - Follow: ```declare -A runs_by_en=( [20]="19582 19583" [30]="19579 19580 19581" [40]="19578" [60]="19576 19577" [80]="19574 19575" [100]="19572 19573 19614" [120]="19571" [175]="19565 19566" [200]="19564 19567 19568 19569" [225]="19632 19633" [250]="19626 19587")```
