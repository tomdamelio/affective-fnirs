#!/usr/bin/env python3
"""
Diagnóstico específico de la conversión Beer-Lambert.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import mne
import json

print("=" * 70)
print("DIAGNÓSTICO BEER-LAMBERT LAW")
print("=" * 70)

# Cargar datos preprocesados hasta OD
from affective_fnirs.config import SubjectConfig
from affective_fnirs.ingestion import load_xdf_file, identify_streams
from affective_fnirs.mne_builder import build_fnirs_raw

config = SubjectConfig.from_yaml(Path("configs/sub-009.yml"))

# Cargar XDF
xdf_path = Path("data/raw/sub-009/ses-001/sub-009_ses-001_task-fingertapping_recording.xdf")
streams, header = load_xdf_file(xdf_path)
stream_ids = identify_streams(streams)

# Extraer datos
fnirs_stream = stream_ids["fnirs"]
fnirs_data = fnirs_stream["time_series"]
fnirs_timestamps = fnirs_stream["time_stamps"]
fnirs_sfreq = 1.0 / np.median(np.diff(fnirs_timestamps))

# Cargar JSON
json_path = Path("data/raw/sub-009/ses-001/sub-009_ses-001_task-fingertapping_nirs.json")
with open(json_path) as f:
    json_sidecar = json.load(f)
montage_from_json = json_sidecar.get("ChMontage", [])

# Construir Raw
fnirs_raw = build_fnirs_raw(fnirs_data, fnirs_sfreq, montage_from_json, fnirs_timestamps)

# Filtrar solo canales fNIRS
channel_types = fnirs_raw.get_channel_types()
fnirs_channels = [ch for ch, ct in zip(fnirs_raw.ch_names, channel_types) if ct == "fnirs_cw_amplitude"]
fnirs_raw = fnirs_raw.copy().pick(fnirs_channels)

print(f"\n1. Raw fNIRS (solo fnirs_cw_amplitude):")
print(f"   - Canales: {len(fnirs_raw.ch_names)}")
print(f"   - Tipos: {set(fnirs_raw.get_channel_types())}")
data = fnirs_raw.get_data()
print(f"   - Min: {data.min():.6e}")
print(f"   - Max: {data.max():.6e}")
print(f"   - Mean: {data.mean():.6e}")

# Convertir a OD
print(f"\n2. Convirtiendo a Optical Density...")
raw_od = mne.preprocessing.nirs.optical_density(fnirs_raw)
od_data = raw_od.get_data()
print(f"   - Canales: {len(raw_od.ch_names)}")
print(f"   - Tipos: {set(raw_od.get_channel_types())}")
print(f"   - Min: {od_data.min():.6e}")
print(f"   - Max: {od_data.max():.6e}")
print(f"   - Mean: {od_data.mean():.6e}")

# Verificar info de canales antes de Beer-Lambert
print(f"\n3. Verificando metadata de canales OD:")
for i, ch_name in enumerate(raw_od.ch_names[:4]):
    ch_info = raw_od.info['chs'][i]
    loc = ch_info['loc']
    print(f"   - {ch_name}:")
    print(f"     - loc[9] (wavelength): {loc[9]}")
    print(f"     - loc[10] (distance): {loc[10]}")
    print(f"     - coil_type: {ch_info.get('coil_type', 'N/A')}")

# Verificar que hay pares de longitudes de onda
print(f"\n4. Verificando pares de longitudes de onda:")
wavelengths = []
for ch_info in raw_od.info['chs']:
    wl = ch_info['loc'][9]
    if wl > 0:
        wavelengths.append(wl)
unique_wl = set(wavelengths)
print(f"   - Longitudes de onda únicas: {unique_wl}")
print(f"   - Conteo: {len(wavelengths)} canales con wavelength > 0")

# Intentar Beer-Lambert
print(f"\n5. Aplicando Beer-Lambert Law...")
try:
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
    hb_data = raw_haemo.get_data()
    print(f"   - Canales: {len(raw_haemo.ch_names)}")
    print(f"   - Tipos: {set(raw_haemo.get_channel_types())}")
    print(f"   - Min: {hb_data.min():.6e}")
    print(f"   - Max: {hb_data.max():.6e}")
    print(f"   - Mean: {hb_data.mean():.6e}")
    
    # Verificar canales individuales
    print(f"\n6. Verificando canales HbO individuales:")
    hbo_channels = [ch for ch, ct in zip(raw_haemo.ch_names, raw_haemo.get_channel_types()) if ct == "hbo"]
    for ch_name in hbo_channels[:5]:
        ch_idx = raw_haemo.ch_names.index(ch_name)
        ch_data = raw_haemo.get_data(picks=[ch_idx])[0]
        print(f"   - {ch_name}: min={ch_data.min():.6e}, max={ch_data.max():.6e}, mean={ch_data.mean():.6e}")
    
    # Verificar si el problema es la distancia source-detector
    print(f"\n7. Verificando distancias source-detector en HbO:")
    for ch_name in hbo_channels[:5]:
        ch_idx = raw_haemo.ch_names.index(ch_name)
        ch_info = raw_haemo.info['chs'][ch_idx]
        loc = ch_info['loc']
        print(f"   - {ch_name}: distance={loc[10]:.6f} m")
        
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("FIN DEL DIAGNÓSTICO")
print("=" * 70)
