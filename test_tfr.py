"""Test script to verify AverageTFR creation."""
import sys
sys.path.insert(0, 'src')
import mne
import numpy as np
from pathlib import Path

# Test creating AverageTFR
info = mne.create_info(ch_names=['ROI_AVG'], sfreq=500, ch_types=['eeg'])
data = np.random.randn(1, 10, 100)  # 1 channel, 10 freqs, 100 times
times = np.linspace(-1, 2, 100)
freqs = np.arange(8, 18)

try:
    avg_tfr = mne.time_frequency.AverageTFR(
        info=info,
        data=data,
        times=times,
        freqs=freqs,
        nave=10
    )
    print('SUCCESS: AverageTFR created')
    print(f'Shape: {avg_tfr.data.shape}')
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
