#!/usr/bin/env python3
"""
Diagnóstico paso a paso del pipeline fNIRS.

Ejecuta cada paso del preprocesamiento y verifica los datos en cada etapa.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import mne
import json

print("=" * 70)
print("DIAGNÓSTICO PASO A PASO DEL PIPELINE fNIRS")
print("=" * 70)

# Cargar configuración
from affective_fnirs.config import SubjectConfig
config = SubjectConfig.from_yaml(Path("configs/sub-009.yml"))

# Cargar montage config
montage_path = Path("configs/montage_config.json")
with open(montage_path) as f:
    montage_config = json.load(f)

print(f"\n1. Cargando datos desde XDF...")
from affective_fnirs.ingestion import load_xdf_file, identify_streams
from affective_fnirs.mne_builder import build_fnirs_raw, embed_events

xdf_path = Path("data/raw/sub-009/ses-001/sub-009_ses-001_task-fingertapping_recording.xdf")
streams, header = load_xdf_file(xdf_path)
stream_ids = identify_streams(streams)

print(f"   Streams: {list(stream_ids.keys())}")

# Construir Raw fNIRS
print(f"\n2. Construyendo MNE Raw fNIRS...")

# Extraer datos del stream fNIRS
fnirs_stream = stream_ids["fnirs"]
fnirs_data = fnirs_stream["time_series"]
fnirs_timestamps = fnirs_stream["time_stamps"]

# Calcular sfreq
if len(fnirs_timestamps) > 1:
    fnirs_sfreq = 1.0 / np.median(np.diff(fnirs_timestamps))
else:
    fnirs_sfreq = float(fnirs_stream["info"]["nominal_srate"][0])

print(f"   - Datos extraídos: {fnirs_data.shape}")
print(f"   - Sfreq: {fnirs_sfreq:.2f} Hz")

# Cargar JSON sidecar para montage
json_path = Path("data/raw/sub-009/ses-001/sub-009_ses-001_task-fingertapping_nirs.json")
with open(json_path) as f:
    json_sidecar = json.load(f)

montage_from_json = json_sidecar.get("ChMontage", [])
print(f"   - Montage JSON: {len(montage_from_json)} canales")

fnirs_raw = build_fnirs_raw(fnirs_data, fnirs_sfreq, montage_from_json, fnirs_timestamps)

print(f"   - Canales: {len(fnirs_raw.ch_names)}")
print(f"   - Tipos: {set(fnirs_raw.get_channel_types())}")

# Verificar datos
data = fnirs_raw.get_data()
print(f"\n   DATOS RAW (después de build_fnirs_raw):")
print(f"   - Shape: {data.shape}")
print(f"   - Min: {data.min():.6e}")
print(f"   - Max: {data.max():.6e}")
print(f"   - Mean: {data.mean():.6e}")

if data.max() == 0:
    print("\n   ❌ ERROR: Los datos ya están en cero después de build_fnirs_raw!")
    print("   El problema está en la construcción del Raw, no en el preprocesamiento.")
    sys.exit(1)

# Embed events
print(f"\n3. Embebiendo eventos...")
from affective_fnirs.mne_builder import embed_events
event_mapping = {"LEFT": 1, "RIGHT": 2, "NOTHING": 3}
fnirs_raw = embed_events(fnirs_raw, stream_ids["markers"], event_mapping)
print(f"   - Anotaciones: {len(fnirs_raw.annotations)}")

# Ahora ejecutar el pipeline paso a paso
print(f"\n4. Ejecutando pipeline paso a paso...")

from affective_fnirs.fnirs_processing import (
    convert_to_optical_density,
    correct_motion_artifacts,
    identify_short_channels,
    apply_short_channel_regression,
    convert_to_hemoglobin,
    filter_hemoglobin_data,
)

# Paso 1: Intensidad → OD
print(f"\n   PASO 1: Intensidad → Optical Density")
try:
    raw_od = convert_to_optical_density(fnirs_raw)
    od_data = raw_od.get_data()
    print(f"   - Shape: {od_data.shape}")
    print(f"   - Min: {od_data.min():.6e}")
    print(f"   - Max: {od_data.max():.6e}")
    print(f"   - Mean: {od_data.mean():.6e}")
    print(f"   - Tipos: {set(raw_od.get_channel_types())}")
    
    if od_data.max() == 0 and od_data.min() == 0:
        print("   ❌ ERROR: Datos en cero después de OD conversion!")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Paso 2: Motion correction (TDDR)
print(f"\n   PASO 2: Motion Correction (TDDR)")
try:
    raw_od_corrected = correct_motion_artifacts(raw_od, method="tddr")
    mc_data = raw_od_corrected.get_data()
    print(f"   - Shape: {mc_data.shape}")
    print(f"   - Min: {mc_data.min():.6e}")
    print(f"   - Max: {mc_data.max():.6e}")
    print(f"   - Mean: {mc_data.mean():.6e}")
    
    if mc_data.max() == 0 and mc_data.min() == 0:
        print("   ❌ ERROR: Datos en cero después de motion correction!")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Paso 3: Short channel identification
print(f"\n   PASO 3: Identificación de canales cortos")
try:
    short_channels, long_channels = identify_short_channels(
        raw_od_corrected, montage_config, short_threshold_mm=15.0
    )
    print(f"   - Short channels: {len(short_channels)}")
    print(f"   - Long channels: {len(long_channels)}")
    if short_channels:
        print(f"   - Short: {short_channels[:5]}...")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    short_channels = []
    long_channels = list(raw_od_corrected.ch_names)

# Paso 4: Short channel regression (si hay canales cortos)
print(f"\n   PASO 4: Short Channel Regression")
if short_channels and long_channels:
    try:
        raw_od_regressed = apply_short_channel_regression(
            raw_od_corrected, short_channels, long_channels
        )
        scr_data = raw_od_regressed.get_data()
        print(f"   - Shape: {scr_data.shape}")
        print(f"   - Min: {scr_data.min():.6e}")
        print(f"   - Max: {scr_data.max():.6e}")
        print(f"   - Mean: {scr_data.mean():.6e}")
        
        if scr_data.max() == 0 and scr_data.min() == 0:
            print("   ❌ ERROR: Datos en cero después de SCR!")
        
        raw_od_final = raw_od_regressed
    except Exception as e:
        print(f"   ⚠️ SCR falló: {e}")
        raw_od_final = raw_od_corrected
else:
    print("   - Saltando SCR (no hay canales cortos)")
    raw_od_final = raw_od_corrected

# Paso 5: OD → Hemoglobina
print(f"\n   PASO 5: OD → Hemoglobina (Beer-Lambert)")
try:
    raw_haemo = convert_to_hemoglobin(raw_od_final, dpf=6.0)
    hb_data = raw_haemo.get_data()
    print(f"   - Shape: {hb_data.shape}")
    print(f"   - Min: {hb_data.min():.6e}")
    print(f"   - Max: {hb_data.max():.6e}")
    print(f"   - Mean: {hb_data.mean():.6e}")
    print(f"   - Tipos: {set(raw_haemo.get_channel_types())}")
    
    if hb_data.max() == 0 and hb_data.min() == 0:
        print("   ❌ ERROR: Datos en cero después de Beer-Lambert!")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Paso 6: Filtrado
print(f"\n   PASO 6: Filtrado (0.01-0.5 Hz)")
try:
    raw_haemo_filtered = filter_hemoglobin_data(raw_haemo, l_freq=0.01, h_freq=0.5)
    filt_data = raw_haemo_filtered.get_data()
    print(f"   - Shape: {filt_data.shape}")
    print(f"   - Min: {filt_data.min():.6e}")
    print(f"   - Max: {filt_data.max():.6e}")
    print(f"   - Mean: {filt_data.mean():.6e}")
    
    if filt_data.max() == 0 and filt_data.min() == 0:
        print("   ❌ ERROR: Datos en cero después del filtrado!")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("FIN DEL DIAGNÓSTICO")
print("=" * 70)
