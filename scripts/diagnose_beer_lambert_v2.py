#!/usr/bin/env python3
"""
Diagnóstico más profundo de Beer-Lambert.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import mne

print("=" * 70)
print("DIAGNÓSTICO BEER-LAMBERT V2")
print("=" * 70)

print(f"\nMNE version: {mne.__version__}")

# Verificar si mne-nirs está instalado
try:
    import mne_nirs
    print(f"MNE-NIRS version: {mne_nirs.__version__}")
except ImportError:
    print("MNE-NIRS: NO INSTALADO")

# Cargar datos OD desde el diagnóstico anterior
from affective_fnirs.config import SubjectConfig
from affective_fnirs.ingestion import load_xdf_file, identify_streams
from affective_fnirs.mne_builder import build_fnirs_raw
import json

config = SubjectConfig.from_yaml(Path("configs/sub-009.yml"))
xdf_path = Path("data/raw/sub-009/ses-001/sub-009_ses-001_task-fingertapping_recording.xdf")
streams, header = load_xdf_file(xdf_path)
stream_ids = identify_streams(streams)

fnirs_stream = stream_ids["fnirs"]
fnirs_data = fnirs_stream["time_series"]
fnirs_timestamps = fnirs_stream["time_stamps"]
fnirs_sfreq = 1.0 / np.median(np.diff(fnirs_timestamps))

json_path = Path("data/raw/sub-009/ses-001/sub-009_ses-001_task-fingertapping_nirs.json")
with open(json_path) as f:
    json_sidecar = json.load(f)
montage_from_json = json_sidecar.get("ChMontage", [])

fnirs_raw = build_fnirs_raw(fnirs_data, fnirs_sfreq, montage_from_json, fnirs_timestamps)

# Filtrar solo canales fNIRS
channel_types = fnirs_raw.get_channel_types()
fnirs_channels = [ch for ch, ct in zip(fnirs_raw.ch_names, channel_types) if ct == "fnirs_cw_amplitude"]
fnirs_raw = fnirs_raw.copy().pick(fnirs_channels)

# Convertir a OD
raw_od = mne.preprocessing.nirs.optical_density(fnirs_raw)

print(f"\n1. Datos OD antes de Beer-Lambert:")
od_data = raw_od.get_data()
print(f"   - Shape: {od_data.shape}")
print(f"   - Min: {od_data.min():.6e}")
print(f"   - Max: {od_data.max():.6e}")

# Verificar el emparejamiento de canales
print(f"\n2. Verificando emparejamiento de canales por wavelength:")
ch_names = raw_od.ch_names
pairs_760 = [ch for ch in ch_names if '760' in ch]
pairs_850 = [ch for ch in ch_names if '850' in ch]
print(f"   - Canales 760nm: {len(pairs_760)}")
print(f"   - Canales 850nm: {len(pairs_850)}")

# Verificar que los pares coinciden
print(f"\n3. Verificando pares source-detector:")
for ch_760 in pairs_760[:5]:
    base_name = ch_760.replace(' 760', '')
    ch_850 = base_name + ' 850'
    if ch_850 in pairs_850:
        print(f"   ✓ Par encontrado: {ch_760} <-> {ch_850}")
    else:
        print(f"   ✗ Par NO encontrado para: {ch_760}")

# Intentar Beer-Lambert con verbose
print(f"\n4. Aplicando Beer-Lambert con verbose...")
try:
    # Verificar los datos antes
    print(f"   Datos OD antes:")
    print(f"   - raw_od._data.shape: {raw_od._data.shape}")
    print(f"   - raw_od._data.min(): {raw_od._data.min():.6e}")
    print(f"   - raw_od._data.max(): {raw_od._data.max():.6e}")
    
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
    
    print(f"\n   Datos Haemo después:")
    print(f"   - raw_haemo._data.shape: {raw_haemo._data.shape}")
    print(f"   - raw_haemo._data.min(): {raw_haemo._data.min():.6e}")
    print(f"   - raw_haemo._data.max(): {raw_haemo._data.max():.6e}")
    
    # Verificar si hay algún canal con datos
    for i, ch_name in enumerate(raw_haemo.ch_names[:10]):
        ch_data = raw_haemo._data[i]
        if ch_data.max() != 0 or ch_data.min() != 0:
            print(f"   Canal con datos: {ch_name}")
            
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Verificar la estructura interna de los canales
print(f"\n5. Verificando estructura interna de canales OD:")
for i in range(min(4, len(raw_od.ch_names))):
    ch = raw_od.info['chs'][i]
    print(f"   Canal {i} ({ch['ch_name']}):")
    print(f"     - kind: {ch['kind']}")
    print(f"     - coil_type: {ch['coil_type']}")
    print(f"     - loc[:12]: {ch['loc'][:12]}")

# Verificar si el problema es con la función de MNE
print(f"\n6. Probando Beer-Lambert manualmente...")

# Obtener coeficientes de extinción de MNE
from mne.preprocessing.nirs import _get_extinction_coefficients

try:
    ext_coef = _get_extinction_coefficients()
    print(f"   Coeficientes de extinción disponibles: {list(ext_coef.keys())[:5]}...")
except Exception as e:
    print(f"   Error obteniendo coeficientes: {e}")

# Verificar si hay un problema con las unidades
print(f"\n7. Verificando unidades de wavelength:")
for i in range(min(4, len(raw_od.ch_names))):
    ch = raw_od.info['chs'][i]
    wl = ch['loc'][9]
    print(f"   {ch['ch_name']}: wavelength = {wl} (esperado: 760 o 850 en nm)")
    
    # MNE espera wavelength en nm, no en metros
    if wl < 100:
        print(f"   ⚠️ PROBLEMA: wavelength parece estar en metros, no en nm!")

print("\n" + "=" * 70)
print("FIN DEL DIAGNÓSTICO")
print("=" * 70)
