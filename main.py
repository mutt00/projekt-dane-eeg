import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pathlib import Path


mne.viz.set_browser_backend("qt") # pip install pyqt5 mne-qt-browser
mne.set_log_level("ERROR")


#==== Ładowanie danych ====#

#PROJECT_PATH = Path.cwd() # jupyter
PROJECT_PATH = Path(__file__).parent # bezpieczniejsze

DATA_DIR = "data"

raw = mne.io.read_raw_brainvision(
    PROJECT_PATH / DATA_DIR / "sub-01" / "eeg" / "sub-01_task-differentdoors_eeg.vhdr",
    preload = True
)


#==== Podstawowy preprocessing ====#

# downsample to 500Hz
raw.resample(sfreq=500, npad="auto")

# ogólny filtr
# - high-pass powyżej 0.1 Hz może zniekształcić ERPy
# - w badaniu użyto low-pass 30 Hz oraz ICLabel, lecz
#   ICLabel preferuje 1-100 Hz
raw.filter(l_freq=0.1, h_freq=100.0, fir_design="firwin")

# refka na Fz (z badania)
raw.add_reference_channels(ref_channels=["Fz"])

# re-reference na podstawie średniej ze wszystkich kanałów
raw.set_eeg_reference("average", projection=False)

# zmień nazwę M1/M2 na TP9/TP10 (najbliższy odpowiednik w montażu)
# raw.rename_channels({"M1": "TP9", "M2": "TP10"})

# usuń M1, M2 (nie używamy ich do re-reference)
raw.drop_channels(["M1", "M2"], on_missing="warn")

# ustaw montaż
raw.set_montage("easycap-M1", match_case=False, on_missing="warn")

raw.notch_filter(
    freqs=[60], # US; harmonics (120, 180...) already covered by low-pass
    method="fir",
    filter_length="10s"
)


#==== Oznaczanie kanałów powyżej progu impedancji ====#

# 20 kOhms (z badania)
IMP_THRESH = 20

# dla każdego kanału, jeśli wartość impedancji nie jest
# None i jest powyżej progu, zaznacz jako "bad"
bads = [
    channel for channel, info in raw.impedances.items()
    if channel in raw.ch_names
    and info.get("imp") is not None
    and info["imp"] > IMP_THRESH
]
raw.info["bads"] = bads

# ze względu na szum, raw.plot()
raw.info["bads"].append("T7")

print(f"Bads ({len(raw.info['bads'])}): {raw.info['bads']}")


#==== Złe segmenty (burst) ====#

# wyłapane w raw.plot(), wpisane ręcznie
bad_segments = [
    (144.449, 1.259, 'BAD_burst'),
    (594.367, 3.630, 'BAD_burst'),
    (599.392, 1.508, 'BAD_burst'),
    (694.231, 0.326, 'BAD_burst'),
    (745.447, 0.488, 'BAD_burst'),
    (805.968, 0.852, 'BAD_burst'),
    (814.383, 0.703, 'BAD_burst'),
    (1113.212, 2.989, 'BAD_burst'),
    (1408.841, 3.116, 'BAD_burst'),
    (1447.342, 3.826, 'BAD_burst'),
]

onsets, durations, descriptions = zip(*bad_segments)

raw.set_annotations(raw.annotations + mne.Annotations(
    onset=list(onsets),
    duration=list(durations),
    description=list(descriptions),
    orig_time=raw.info['meas_date'],
))

#print(raw.annotations)


#==== Sanity check pre-ICA ====#

print(raw.info)

# snippet of raw eeg before ICA
raw.copy().crop(tmin=140, tmax=160).plot(
    scalings=dict(eeg=20e-6),
    duration=20,
    n_channels=31
)

# plot sensors (electrodes)
raw.plot_sensors(
    title="Sensor plot",
    ch_groups='position',
    show_names=True,
    sphere="auto"
)

# PSD
spectrum = raw.compute_psd(
    method="welch",
    n_fft=int(4 * raw.info["sfreq"]),
    fmin=0,
    fmax=140.0 # force Qt
)

# zapobiega przeciągania wykresu w kierunku minus nieskończoności
# (Fz/ref == same zera)
picks_no_fz = [channel for channel in raw.ch_names if channel != "Fz"]
spectrum.plot(
    average=False,
    spatial_colors=True,
    picks=picks_no_fz
)


#==== ICA ====#
# Wymóg: filtr 1-100 Hz
# uwaga: low-pass 100 Hz już zastosowany na raw

raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, fir_design="firwin")

n_channels = len(raw.ch_names) - len(raw.info["bads"])
rank = mne.compute_rank(raw_for_ica, rank="info")
print(f"n_channels (w tym Fz): {n_channels}")
print(f"Ranga danych (auto): {rank['eeg']}")

# MNE nie loguje utraty rangi na skutek refowania (?), więc:
real_rank = rank['eeg'] - 1 # Fz reference
print(f"Prawdziwa ranga danych: {real_rank}")

# ICA na raw nie działało dobrze, ze względu charakterystyke artefaktów,
# można też zrobić na epokach
events, event_dict = mne.events_from_annotations(raw)
epochs_for_manual_ica = mne.Epochs(
    raw_for_ica, events, event_id=event_dict,
    tmin=-0.2, tmax=0.6,
    baseline=None, preload=True,
    reject_by_annotation=True,
)

# uwaga: nie koniecznie deterministyczne, reprodukowalność wymaga
# conajmniej tych samych danych (raw), tego samego random_state,
# oraz tych samych parametrów (np. picard, ortho=F, extended=T)
ica = ICA(
    n_components=real_rank,
    method="picard",
    fit_params=dict(ortho=False, extended=True),
    max_iter="auto",
    random_state=97
)
ica.fit(epochs_for_manual_ica, decim=5)

ica.plot_components() # topografia komponentów
ica.plot_sources(raw.copy().crop(tmin=140, tmax=160)) # przykładowy odcinek

# dobór komponentów na podstawie ica.plot_components() i ica.plot_sources(raw)
manual_excludes = sorted(list(set([0, 17, 6, 8, 12, 15, 16, 17, 18, 21, 22, 26, 28, 25, 27])))
print(f"Ręcznie wybrane komponenty ({len(manual_excludes)}/{ica.n_components_}): {manual_excludes}")

ica.exclude = manual_excludes

#ica.plot_properties(raw_for_ica, picks=ica.exclude)

# porównanie z raw
ica.plot_overlay(raw, exclude=ica.exclude)

raw_after_manual_ica = raw.copy()
ica.apply(raw_after_manual_ica)

# check
raw_after_manual_ica.copy().crop(tmin=140, tmax=160).plot(
    scalings=dict(eeg=20e-6),
    duration=20,
    n_channels=31
)


#==== ICLabel ====#

# ica = ICA(...)
# ica.fit(...)

# Algorytm ICLabel
ic_labels = label_components(raw_for_ica, ica, method="iclabel")

labels = ic_labels["labels"] # np. 'brain', 'eye blink', 'muscle artifact'
probs = ic_labels["y_pred_proba"] # pewność modelu od 0.0 do 1.0

# (Opcjonalnie) Wypisanie wyników do konsoli, żeby wiedzieć co się dzieje
print("\n--- Wyniki klasyfikacji ICLabel ---")
for idx, (label, prob) in enumerate(zip(labels, probs)):
    print(f"IC{idx:02d}: {label:15s} (pewność: {prob:.2f})")

# Oznaczenie artefaktów do usunięcia
# (zostawiamy sygnały sklasyfikowane jako "brain" i "other")
exclude_categories = ["muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"]
ica.exclude = [
    idx for idx, label in enumerate(labels)
    if label in exclude_categories
]
print(f"\nUsunięte komponenty (artefakty): {ica.exclude}")

raw_after_iclabel = raw.copy()
ica.apply(raw_after_iclabel)

# check
raw_after_iclabel.copy().crop(tmin=140, tmax=160).plot(
    scalings=dict(eeg=20e-6),
    duration=20,
    n_channels=31
)


#==== Epochs, events ====#

# załóżmy, że moje jest lepsze ;)
#raw = raw_after_iclabel.copy()
raw_after_ica = raw_after_manual_ica.copy()

events, event_dict = mne.events_from_annotations(raw)
print("\nEvent dictionary inferred from annotations:")
for k,v in event_dict.items():
    print(k, ".......",  v)

epochs = mne.Epochs(
    raw_after_ica,
    events, 
    event_id=event_dict,
    tmin=-0.2, # (z badania)
    tmax=0.6, # (z badania)
    baseline=(-0.2, 0.0),
    preload=True,
    reject=None # Na tym etapie nic nie wyrzucamy!
)

# Tworzymy symulację odrzucania na kopii danych, żeby zidentyfikować zepsute kanały
reject_criteria = dict(eeg=150e-6)
epochs_test = epochs.copy().drop_bad(reject=reject_criteria)

# Zliczamy, ile razy każda elektroda zepsuła epokę
dropped_counts = {ch: 0 for ch in epochs.ch_names}
total_epochs = len(epochs)

# drop_log to lista, która mówi nam, co było powodem usunięcia danej epoki
for drop_reason in epochs_test.drop_log:
    # Jeśli powodem był skok napięcia na kanale, dodajemy punkt karny dla tej elektrody
    for ch in drop_reason:
        if ch in dropped_counts:
            dropped_counts[ch] += 1

# Szukamy elektrod, które psują więcej niż 10% epok
threshold = 0.10 * total_epochs
noisy_electrodes = [ch for ch, count in dropped_counts.items() if count > threshold]

print(f"Zaszumione elektrody (>10% zepsutych epok): {noisy_electrodes}")

# Decyzja zgodnie z tekstem (Interpolacja vs Wyrzucenie badanego)
if len(noisy_electrodes) > 3:
    # W normalnym workflow tutaj skrypt by przerwał działanie dla tego pliku
    print("UWAGA: Badany ma więcej niż 3 zepsute elektrody! Zgodnie z artykułem wylatuje.")
else:
    # Oznaczamy elektrody jako zepsute na głównych danych
    epochs.info['bads'] = noisy_electrodes
    
    # Interpolujemy (odbudowujemy sygnał na podstawie dobrych sąsiadów)
    epochs.interpolate_bads(reset_bads=True)
    print("Złe elektrody zostały zinterpolowane.")
    
    # Właściwe drzucenie epok
    # Teraz, gdy złe kanały są naprawione, aplikujemy nasze odrzucanie napięciowe (150 uV)
    # na ostateczne dane, aby wyczyścić resztki (np. rzeczywiste ruchy głowy)
    epochs.drop_bad(reject=reject_criteria)
    print(f"Pozostało czystych epok do ERP: {len(epochs)}")
