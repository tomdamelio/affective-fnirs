#!/usr/bin/env python3
"""
Diagnóstico final: verificar unidades de wavelength.
"""

import numpy as np
import mne

print("=" * 70)
print("DIAGNÓSTICO BEER-LAMBERT V5 - Unidades de wavelength")
print("=" * 70)

# Crear datos mínimos
n_samples = 100
sfreq = 10.0

# Probar con wavelength en NANÓMETROS (760, 850)
print("\n1. Probando con wavelength en NANÓMETROS (760, 850)...")

info_nm = mne.create_info(
    ch_names=['S1_D1 760', 'S1_D1 850'],
    sfreq=sfreq,
    ch_types=['fnirs_cw_amplitude', 'fnirs_cw_amplitude']
)

for i, ch_name in enumerate(info_nm.ch_names):
    wavelength_nm = 760.0 if '760' in ch_name else 850.0  # En nanómetros
    
    # Posiciones en metros
    source_pos = np.array([0.0, 0.0, 0.1])
    detector_pos = np.array([0.03, 0.0, 0.1])
    
    info_nm['chs'][i]['loc'][0:3] = source_pos
    info_nm['chs'][i]['loc'][3:6] = detector_pos
    info_nm['chs'][i]['loc'][9] = wavelength_nm

# Crear datos de intensidad con variación
np.random.seed(42)
data = np.random.randn(2, n_samples) * 0.01 + 0.5  # Intensidad ~0.5 con ruido

raw_nm = mne.io.RawArray(data, info_nm)
print(f"   - Wavelength canal 0: {raw_nm.info['chs'][0]['loc'][9]}")
print(f"   - Wavelength canal 1: {raw_nm.info['chs'][1]['loc'][9]}")

# Convertir a OD
print("\n2. Convirtiendo a OD...")
raw_od_nm = mne.preprocessing.nirs.optical_density(raw_nm)
od_data = raw_od_nm.get_data()
print(f"   - OD min: {od_data.min():.6e}")
print(f"   - OD max: {od_data.max():.6e}")
print(f"   - Wavelength después de OD: {raw_od_nm.info['chs'][0]['loc'][9]}")

# Aplicar Beer-Lambert
print("\n3. Aplicando Beer-Lambert...")
try:
    raw_haemo_nm = mne.preprocessing.nirs.beer_lambert_law(raw_od_nm, ppf=6.0)
    hb_data = raw_haemo_nm.get_data()
    print(f"   - Hb min: {hb_data.min():.6e}")
    print(f"   - Hb max: {hb_data.max():.6e}")
    
    if hb_data.max() != 0:
        print(f"   ✅ Beer-Lambert funciona con wavelength en nm!")
    else:
        print(f"   ❌ Datos en cero")
except Exception as e:
    print(f"   ERROR: {e}")

# Verificar qué hace optical_density con el wavelength
print("\n4. Verificando transformación de wavelength en optical_density...")
print(f"   - Antes de OD: {raw_nm.info['chs'][0]['loc'][9]}")
print(f"   - Después de OD: {raw_od_nm.info['chs'][0]['loc'][9]}")

# Verificar el tipo de canal después de OD
print(f"   - Tipo antes de OD: {raw_nm.get_channel_types()[0]}")
print(f"   - Tipo después de OD: {raw_od_nm.get_channel_types()[0]}")

# Verificar coil_type
print(f"   - coil_type antes: {raw_nm.info['chs'][0]['coil_type']}")
print(f"   - coil_type después: {raw_od_nm.info['chs'][0]['coil_type']}")

print("\n" + "=" * 70)
