#!/usr/bin/env python3
"""
Diagnóstico simplificado de datos fNIRS.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pyxdf

print("=" * 70)
print("DIAGNÓSTICO DE DATOS fNIRS")
print("=" * 70)

# Cargar XDF directamente
xdf_path = Path("data/raw/sub-009/ses-001/sub-009_ses-001_task-fingertapping_recording.xdf")
print(f"\n1. Cargando XDF: {xdf_path}")

streams, header = pyxdf.load_xdf(str(xdf_path))

# Encontrar stream fNIRS
fnirs_stream = None
for stream in streams:
    name = stream['info']['name'][0].lower()
    if 'photon' in name or 'nirs' in name or 'fnirs' in name:
        fnirs_stream = stream
        break

if fnirs_stream is None:
    print("ERROR: No se encontró stream fNIRS")
    sys.exit(1)

print(f"\n2. Stream fNIRS encontrado: {fnirs_stream['info']['name'][0]}")

# Datos crudos
raw_data = fnirs_stream['time_series']
print(f"\n3. Datos CRUDOS del XDF:")
print(f"   - Shape: {raw_data.shape}")
print(f"   - Min: {raw_data.min():.6e}")
print(f"   - Max: {raw_data.max():.6e}")
print(f"   - Mean: {raw_data.mean():.6e}")
print(f"   - Std: {raw_data.std():.6e}")
print(f"   - Zeros: {(raw_data == 0).sum()} de {raw_data.size} ({100*(raw_data == 0).sum()/raw_data.size:.1f}%)")

# Verificar canales individuales
print(f"\n4. Primeros 10 canales:")
for i in range(min(10, raw_data.shape[1])):
    ch_data = raw_data[:, i]
    zeros_pct = 100 * (ch_data == 0).sum() / len(ch_data)
    print(f"   - Canal {i}: min={ch_data.min():.4e}, max={ch_data.max():.4e}, zeros={zeros_pct:.1f}%")

# Ahora cargar el archivo preprocesado
print("\n" + "=" * 70)
print("DATOS PREPROCESADOS")
print("=" * 70)

import mne

fnirs_path = Path("data/derivatives/validation-pipeline/sub-009/ses-001/sub-009_ses-001_task-fingertapping_desc-preprocessed_fnirs.fif")
print(f"\n5. Cargando preprocesado: {fnirs_path}")

fnirs_raw = mne.io.read_raw_fif(fnirs_path, preload=True)
prep_data = fnirs_raw.get_data()

print(f"\n6. Datos PREPROCESADOS:")
print(f"   - Shape: {prep_data.shape}")
print(f"   - Min: {prep_data.min():.6e}")
print(f"   - Max: {prep_data.max():.6e}")
print(f"   - Mean: {prep_data.mean():.6e}")
print(f"   - Std: {prep_data.std():.6e}")
print(f"   - Zeros: {(prep_data == 0).sum()} de {prep_data.size} ({100*(prep_data == 0).sum()/prep_data.size:.1f}%)")

# Conclusión
print("\n" + "=" * 70)
print("CONCLUSIÓN")
print("=" * 70)

if raw_data.max() > 0 and prep_data.max() == 0:
    print("\n❌ PROBLEMA IDENTIFICADO:")
    print("   - Los datos CRUDOS tienen valores válidos")
    print("   - Los datos PREPROCESADOS están todos en CERO")
    print("   - El problema está en el PREPROCESAMIENTO fNIRS")
    print("\n   Posibles causas:")
    print("   1. Error en la conversión de intensidad a concentración (Beer-Lambert)")
    print("   2. Error en el filtrado que elimina toda la señal")
    print("   3. Error en la corrección de movimiento")
    print("   4. Problema con los canales cortos (short channels)")
elif raw_data.max() == 0:
    print("\n❌ Los datos CRUDOS ya están en cero - problema en la adquisición")
else:
    print("\n✅ Ambos datasets tienen valores válidos")
