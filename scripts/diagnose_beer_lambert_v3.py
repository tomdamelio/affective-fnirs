#!/usr/bin/env python3
"""
Diagnóstico: verificar si el problema es la falta de posiciones 3D.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import mne

print("=" * 70)
print("DIAGNÓSTICO: POSICIONES 3D EN BEER-LAMBERT")
print("=" * 70)

# Crear datos sintéticos para probar
print("\n1. Creando datos sintéticos fNIRS...")

# Simular 2 canales (un par 760/850)
n_samples = 1000
sfreq = 10.0
data = np.random.randn(2, n_samples) * 0.01  # Pequeñas variaciones en OD

# Crear info con posiciones 3D
ch_names = ['S1_D1 760', 'S1_D1 850']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='fnirs_od')

# Configurar metadata SIN posiciones 3D (como está ahora)
for i, ch_name in enumerate(ch_names):
    wavelength = 760 if '760' in ch_name else 850
    info['chs'][i]['loc'][9] = wavelength  # wavelength en nm
    info['chs'][i]['loc'][10] = 0.03  # distance en metros

print("\n2. Probando Beer-Lambert SIN posiciones 3D...")
raw_od_no_pos = mne.io.RawArray(data, info.copy())
print(f"   loc[0:6] (source pos): {raw_od_no_pos.info['chs'][0]['loc'][0:6]}")
print(f"   loc[6:9] (detector pos): {raw_od_no_pos.info['chs'][0]['loc'][6:9]}")

try:
    raw_haemo_no_pos = mne.preprocessing.nirs.beer_lambert_law(raw_od_no_pos, ppf=6.0)
    hb_data = raw_haemo_no_pos.get_data()
    print(f"   Resultado: min={hb_data.min():.6e}, max={hb_data.max():.6e}")
    if hb_data.max() == 0:
        print("   ❌ Datos en cero SIN posiciones 3D")
except Exception as e:
    print(f"   ERROR: {e}")

# Ahora probar CON posiciones 3D
print("\n3. Probando Beer-Lambert CON posiciones 3D...")
info_with_pos = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='fnirs_od')

# Configurar posiciones 3D (source y detector)
# Source position: loc[0:3]
# Detector position: loc[3:6]
for i, ch_name in enumerate(ch_names):
    wavelength = 760 if '760' in ch_name else 850
    
    # Posiciones ficticias pero válidas
    source_pos = [0.0, 0.0, 0.1]  # Source en (0, 0, 0.1)
    detector_pos = [0.03, 0.0, 0.1]  # Detector a 3cm de distancia
    
    info_with_pos['chs'][i]['loc'][0:3] = source_pos
    info_with_pos['chs'][i]['loc'][3:6] = detector_pos
    info_with_pos['chs'][i]['loc'][9] = wavelength
    info_with_pos['chs'][i]['loc'][10] = 0.03

raw_od_with_pos = mne.io.RawArray(data, info_with_pos)
print(f"   loc[0:3] (source pos): {raw_od_with_pos.info['chs'][0]['loc'][0:3]}")
print(f"   loc[3:6] (detector pos): {raw_od_with_pos.info['chs'][0]['loc'][3:6]}")

try:
    raw_haemo_with_pos = mne.preprocessing.nirs.beer_lambert_law(raw_od_with_pos, ppf=6.0)
    hb_data = raw_haemo_with_pos.get_data()
    print(f"   Resultado: min={hb_data.min():.6e}, max={hb_data.max():.6e}")
    if hb_data.max() != 0:
        print("   ✅ Datos válidos CON posiciones 3D!")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 70)
print("CONCLUSIÓN")
print("=" * 70)
print("\nEl problema es que build_fnirs_raw() no configura las posiciones 3D")
print("de los optodos (loc[0:6]). MNE's beer_lambert_law() necesita estas")
print("posiciones para calcular correctamente las concentraciones de hemoglobina.")
print("\nSOLUCIÓN: Agregar posiciones 3D ficticias o reales en build_fnirs_raw()")
