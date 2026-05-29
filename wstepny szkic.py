import mne


raw = mne.io.read_raw_brainvision('/Users/natanstaron/ds007647-download/sub-01/eeg/sub-01_task-differentdoors_eeg.vhdr', preload = True)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)


raw.set_eeg_reference(['M1','M2'])

raw.plot()

raw.resample(sfreq = 250)
raw.filter(l_freq = 0.1, h_freq = 30)
raw.notch_filter (freqs = 60)

print(raw)
print(raw.info)


raw_clean = raw.copy()

from mne.preprocessing import ICA


ica = ICA(
    n_components=20,
    method='fastica',
    random_state=42,
    max_iter=800
)


ica.fit(raw)
#ica.plot_components()
#plt.show()
print(ica)

#ica.exclude = [0, 1]
#ica.apply(raw_clean)
#ica.plot_overlay(raw, exclude=[0, 1])
