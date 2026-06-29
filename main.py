import mne
from mne.preprocessing import ICA
from pathlib import Path


mne.viz.set_browser_backend("qt") # pip install pyqt5 mne-qt-browser
mne.set_log_level("WARNING")

PROJECT_PATH = Path(__file__).parent
DATA_DIR = "data"

### Read data
# data should live in 'data' dir adjacent to this script
raw = mne.io.read_raw_brainvision(
    PROJECT_PATH / DATA_DIR / "sub-01" / "eeg" / "sub-01_task-differentdoors_eeg.vhdr",
    preload = True
)

raw.resample(sfreq=500, npad="auto") # downsample to 500Hz
raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin") # high-pass (for drift removal)
raw.filter(l_freq=None, h_freq=80, fir_design="firwin") # low-pass (for EMG)
raw.add_reference_channels(ref_channels=["Fz"]) # reference electrode at Fz
raw.rename_channels({"M1": "TP9", "M2": "TP10"}) # rename Mastoid electrodes to TP9/TP10
raw.set_eeg_reference(ref_channels=["TP9", "TP10"], projection=False) # re-reference using mastoids, channels later dropped
raw.set_montage("easycap-M1", match_case=False, on_missing="warn") # set montage

# Remove bad impedance channels
# for every channel, if its impedance value is not None and is > threshold, then mark as bad
IMP_THRESH = 25
bads = [
    channel for channel, info in raw.impedances.items()
    if channel in raw.ch_names
    and info.get("imp") is not None
    and info["imp"] > IMP_THRESH
]
raw.info["bads"] = bads

# note: bad ref leaks noise to all channels, thence:
for mastoid in ["TP9", "TP10"]:
    if mastoid in bads:
        print(f"{mastoid} (reref) bad impedance!")

### Sanity checks
# All channels
raw_cropped = raw.copy().crop(0, 60)
# raw_cropped.plot(
#     title="All channels [t0–60]",
#     duration=60,
#     n_channels=33
# )

# Mastoids (TP9/10), plots be inverse of eachother
# raw_cropped.plot(
#     title="Mastoids (TP9/10) [t0–60]",
#     picks=["TP9", "TP10"],
#     scalings={"eeg": 75e-6}
# )

# Drop mastoids from analysis
raw.drop_channels(["TP9", "TP10"])

#print(raw.info)
"""
<Info | 9 non-empty values
 bads: ...
 ch_names: ...
 chs: 33 EEG
 custom_ref_applied: True
 dig: 36 items (3 Cardinal, 33 EEG)
 highpass: 0.0 Hz
 lowpass: 1000.0 Hz
 meas_date: 2025-03-31 12:01:00 UTC
 nchan: 33
 projs: []
 sfreq: 500.0 Hz
>
"""

# Sensors
# note: plots electrodes, not channels
# raw.plot_sensors(
#     title="Sensor plot",
#     ch_groups='position',
#     show_names=True,
#     sphere="auto"
# )

# PSD (power spectrum density)

spectrum = raw.compute_psd(method="welch", n_fft=int(4 * raw.info["sfreq"]))

# # pre-notch
# spectrum.plot(
#     average=False,
#     spatial_colors=True
# )

# from PSD
raw.notch_filter(
    freqs=[60], # US; harmonics (120, 180...) already covered by band-pass
    method="spectrum_fit",
    filter_length="10s"
)


# # post-notch
# spectrum = raw.compute_psd(method="welch", n_fft=int(4 * raw.info["sfreq"]))
# spectrum.plot(
#     average=False,
#     spatial_colors=True
# )

# spectrum.plot_topo(color="k", fig_facecolor="w", axis_facecolor="w")

# spectrum.plot_topomap(
#     bands={"Delta (1-4 Hz)": (1, 4),
#            "Theta (4-8 Hz)": (4, 8),
#            "Alpha (8-12 Hz)": (8, 12),
#            "Beta (13-30 Hz)": (13, 30),
#            "Gamma (30-45 Hz)": (30, 45)},
#     normalize=True,
# )

### ICA

# high-pass filtered copy of raw for ICA
raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, fir_design="firwin")

# each rank-reducing operation (eg. re-referencing) must be accounted for, otherwise ICA overfits
rank = mne.compute_rank(raw_for_ica, rank="info")
n_components = rank["eeg"]
#print(f"Data rank: {n_components}")

ica = ICA(
    n_components=n_components,
    method="picard",
    fit_params=dict(ortho=False, extended=True),
    max_iter=500,
    random_state=97 # keep this fixed
)
ica.fit(raw_for_ica, decim=5) # decim=12 for 1000Hz

# ica.plot_sources(raw_for_ica, show_scrollbars=True) # time courses
# ica.plot_components() # topographies

muscle_indices, muscle_scores = ica.find_bads_muscle(raw_for_ica)

ica.exclude = list(set(muscle_indices))
print(f"Marked for exclusion: {ica.exclude}")

ica.apply(raw)

spectrum.plot_topomap(
    bands={"Delta (1-4 Hz)": (1, 4),
           "Theta (4-8 Hz)": (4, 8),
           "Alpha (8-12 Hz)": (8, 12),
           "Beta (13-30 Hz)": (13, 30),
           "Gamma (30-45 Hz)": (30, 45)},
    normalize=True,
)


input()
