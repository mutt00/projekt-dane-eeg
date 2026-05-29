# Analiza danych EEG 'different doors'

## Źródła

- Dane: https://openneuro.org/datasets/ds007647/versions/1.0.1
  - GIT: https://github.com/OpenNeuroDatasets/ds007647.git
- Badanie: https://www.biorxiv.org/content/10.64898/2026.04.13.718323v1

## Setup

``` shell
git clone "https://github.com/mutt00/projekt-dane-eeg.git" different-doors-eeg
cd different-doors-eeg/

datalad clone "https://github.com/OpenNeuroDatasets/ds007647.git" data
datalad get data/sub-01

# requires mne
python3 main.py
```

