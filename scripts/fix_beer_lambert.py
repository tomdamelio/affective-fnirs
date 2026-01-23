#!/usr/bin/env python3
"""
Script para diagnosticar y arreglar el problema de Beer-Lambert.
"""

import numpy as np
import mne

print("=" * 70)
print("DIAGNÓSTICO BEER-LAMBERT")
print("=" * 70)

# Crear datos mínimos
info = mne.create_info(['S1_D1 760', 'S1_D1 850'], 10.0, ['fnirs_cw_amplitude']*2)

for i in range(2):
    info['chs'][i]['loc'][0:3] = [0, 0, 0.1]
    info['chs'][i]['loc'][3:6] = [0.03, 0, 0.1]
    info['chs'][i]['loc'][9] = 760.0 if i == 0 else 850.0

np.random.seed(42)
data = np.random.randn(2, 100) * 0.01 + 0.5
raw = mne.io.RawArray(data, info)
raw_od = mne.preprocessing.nirs.optical_density(raw)

print(f"OD data mean: {raw_od.get_data().mean():.6f}")

# Aplicar Beer-Lambert
raw_hb = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)

print(f"Hb data mean: {raw_hb.get_data().mean():.6e}")
print(f"Hb channels: {raw_hb.ch_names}")
print(f"Hb types: {raw_hb.get_channel_types()}")

# Verificar si el problema es el emparejamiento
print("\nVerificando emparejamiento de canales...")
from mne.io.pick import _picks_to_idx

# Verificar qué canales se emparejan
picks_hbo = _picks_to_idx(raw_hb.info, 'hbo')
picks_hbr = _picks_to_idx(raw_hb.info, 'hbr')
print(f"HbO picks: {picks_hbo}")
print(f"HbR picks: {picks_hbr}")

# Verificar los datos de cada canal
for i, ch_name in enumerate(raw_hb.ch_names):
    ch_data = raw_hb.get_data()[i]
    print(f"  {ch_name}: min={ch_data.min():.6e}, max={ch_data.max():.6e}")
