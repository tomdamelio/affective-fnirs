#!/usr/bin/env python3
"""
Diagnóstico: inspeccionar el código de beer_lambert_law.
"""

import numpy as np
import mne
from mne.preprocessing.nirs import beer_lambert_law
import inspect

print("=" * 70)
print("DIAGNÓSTICO BEER-LAMBERT V6 - Inspección del código")
print("=" * 70)

# Ver el código fuente
print("\n1. Ubicación del código de beer_lambert_law:")
print(f"   {inspect.getfile(beer_lambert_law)}")

# Crear datos de prueba
n_samples = 100
sfreq = 10.0

info = mne.create_info(
    ch_names=['S1_D1 760', 'S1_D1 850'],
    sfreq=sfreq,
    ch_types=['fnirs_cw_amplitude', 'fnirs_cw_amplitude']
)

for i, ch_name in enumerate(info.ch_names):
    wavelength_nm = 760.0 if '760' in ch_name else 850.0
    source_pos = np.array([0.0, 0.0, 0.1])
    detector_pos = np.array([0.03, 0.0, 0.1])
    
    info['chs'][i]['loc'][0:3] = source_pos
    info['chs'][i]['loc'][3:6] = detector_pos
    info['chs'][i]['loc'][9] = wavelength_nm

np.random.seed(42)
data = np.random.randn(2, n_samples) * 0.01 + 0.5

raw = mne.io.RawArray(data, info)
raw_od = mne.preprocessing.nirs.optical_density(raw)

print("\n2. Datos OD antes de Beer-Lambert:")
print(f"   - Shape: {raw_od.get_data().shape}")
print(f"   - Min: {raw_od.get_data().min():.6e}")
print(f"   - Max: {raw_od.get_data().max():.6e}")

# Verificar los picks que usa beer_lambert_law
from mne.preprocessing.nirs._beer_lambert_law import _validate_nirs_info

print("\n3. Validando info para Beer-Lambert...")
try:
    picks = _validate_nirs_info(raw_od.info, fnirs="od")
    print(f"   - Picks válidos: {picks}")
except Exception as e:
    print(f"   - Error: {e}")

# Verificar las frecuencias (wavelengths)
print("\n4. Verificando wavelengths detectados por MNE...")
from mne.io.pick import _picks_to_idx

picks_od = _picks_to_idx(raw_od.info, "fnirs_od")
print(f"   - Picks fnirs_od: {picks_od}")

freqs = []
for pick in picks_od:
    freq = raw_od.info['chs'][pick]['loc'][9]
    freqs.append(freq)
    print(f"   - Canal {pick} ({raw_od.ch_names[pick]}): wavelength = {freq}")

# Verificar el emparejamiento de canales
print("\n5. Verificando emparejamiento de canales...")
from mne.preprocessing.nirs._beer_lambert_law import _channel_frequencies

try:
    ch_freqs = _channel_frequencies(raw_od.info)
    print(f"   - Frecuencias por canal: {ch_freqs}")
except Exception as e:
    print(f"   - Error: {e}")

# Intentar ejecutar beer_lambert_law paso a paso
print("\n6. Ejecutando beer_lambert_law con debug...")

# Copiar el código relevante de MNE para debug
from mne.preprocessing.nirs._beer_lambert_law import (
    _validate_nirs_info,
    _channel_frequencies,
    _channel_distances,
    _load_absorption,
)

picks = _validate_nirs_info(raw_od.info, fnirs="od")
print(f"   - picks: {picks}")

freqs = np.array(_channel_frequencies(raw_od.info))
print(f"   - freqs: {freqs}")

distances = np.array(_channel_distances(raw_od.info))
print(f"   - distances: {distances}")

# Verificar si hay NaN en las distancias
print(f"   - NaN en distances: {np.isnan(distances).any()}")

# Cargar coeficientes de absorción
print("\n7. Cargando coeficientes de absorción...")
try:
    abs_coef = _load_absorption(freqs)
    print(f"   - abs_coef shape: {abs_coef.shape}")
    print(f"   - abs_coef: {abs_coef}")
except Exception as e:
    print(f"   - Error: {e}")

print("\n" + "=" * 70)
