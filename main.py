import mne

from pathlib import Path


PROJECT_PATH = Path(__file__).parent
DATA_DIR = 'data'


# Read data
raw = mne.io.read_raw_brainvision(
    PROJECT_PATH / DATA_DIR / 'sub-01' / 'eeg' / 'sub-01_task-differentdoors_eeg.vhdr',
    preload = True
)
