import matplotlib
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

# Wczytanie
raw = mne.io.read_raw_brainvision(
    '/Users/natanstaron/ds007647-download/sub-01/eeg/sub-01_task-differentdoors_eeg.vhdr',
    preload=True
)

# Montaż
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# 1. Resample
raw.resample(sfreq=250)

# 2. Filtrowanie
raw.filter(l_freq=0.1, h_freq=30)
raw.notch_filter(freqs=60)

# 3. Referencja do mastoidów (jak w artykule)
raw.set_eeg_reference(['M1', 'M2'])

# 4. Kopia przed ICA
raw_clean = raw.copy()

# 5. ICA — wyklucz segmenty z amplitudą powyżej ±1000 µV (jak w artykule)
ica = ICA(
    n_components=20,
    method='fastica',
    random_state=42,
    max_iter=800
)

reject_criteria = dict(eeg=1000e-6)  # 1000 µV w woltach
ica.fit(raw, reject=reject_criteria)

# 6. Usuń składowe okularowe
ica.exclude = [0, 1]
ica.apply(raw_clean)

# 7. Sprawdź wynik
ica.plot_overlay(raw, exclude=[0, 1])
plt.show()

print("Preprocessing gotowy!")
# %%
print(raw_clean.info)

events, event_id = mne.events_from_annotations(raw_clean)
print(event_id)
