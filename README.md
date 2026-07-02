# Analiza danych EEG 'different doors'

## Setup

``` shell
git clone "https://github.com/mutt00/projekt-dane-eeg.git" && cd projekt-dane-eeg/

datalad clone "https://github.com/OpenNeuroDatasets/ds007647.git" data
datalad get data/sub-01

python3 main.py
```


## Processing
- downsample from 1000 Hz to 500 Hz
- filter 0.1–100.0 Hz
- drop channels M1, M2
- set Fz as reference electrode, re-reference from average
- set montage as `easycap-M1` (brainvision 32-electrode)
- filter out bad impedance channels (>20kOms), mark T7 as bad
- drop TP9, TP10
- notch-filter at 60 Hz
- ICA wip
- Epochs events wip

## Bibliografia

- Badanie:
  - The Effects of Learnability and Reward Responsiveness on Reward Processing [https://www.biorxiv.org/content/10.64898/2026.04.13.718323v1]
- Dane:
  - OpenNeuro [https://openneuro.org/datasets/ds007647/]
  - GIT [https://github.com/OpenNeuroDatasets/ds007647.git]
- Analiza Danych:
  - EEG is better left alone [https://doi.org/10.1038/s41598-023-27528-0]
  - EEG is better left alone, but ERPs must be attended to [https://doi.org/10.1016/j.ijpsycho.2024.112441]
