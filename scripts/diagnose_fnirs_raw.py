#!/usr/bin/env python3
"""
Diagnóstico de datos fNIRS CRUDOS (antes del preprocesamiento).

Este script examina los datos fNIRS originales del archivo XDF
para verificar si el problema está en los datos crudos o en el preprocesamiento.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from affective_fnirs.config import SubjectConfig
from affective_fnirs.ingestion import load_xdf_file, identify_streams, extract_stream_data

# Cargar configuración
config_path = Path("configs/sub-009.yml")
config = SubjectConfig.from_yaml(config_path)

print("=" * 70)
print("DIAGNÓSTICO DE DATOS fNIRS CRUDOS")
print("=" * 70)

# Construir ruta al archivo XDF
data_root = Path(config.data_root)
xdf_path = data_root / f"sub-{config.subject.id}" / f"ses-{config.subject.session}" / f"sub-{config.subject.id}_ses-{config.subject.session}_task-{config.subject.task}_recording.xdf"
print(f"\n1. Buscando archivo XDF: {xdf_path}")

if not xdf_path.exists():
    print(f"ERROR: No se encontró archivo XDF: {xdf_path}")
    sys.exit(1)

streams, header = load_xdf_file(xdf_path)
print(f"   - Streams encontrados: {len(streams)}")

# Identificar streams
stream_ids = identify_streams(streams)
print(f"\n2. Streams identificados:")
for stream_type, stream_id in stream_ids.items():
    print(f"   - {stream_type}: {stream_id}")

# Extraer datos fNIRS
fnirs_stream_id = stream_ids.get("fnirs")
if fnirs_stream_id is None:
    print("ERROR: No se encontró stream fNIRS")
    sys.exit(1)

print(f"\n3. Extrayendo datos fNIRS (stream_id={fnirs_stream_id})...")
fnirs_data = extract_stream_data(streams, fnirs_stream_id)

print(f"\n4. Información del stream fNIRS:")
print(f"   - Shape de datos: {fnirs_data['data'].shape}")
print(f"   - Frecuencia de muestreo: {fnirs_data['sfreq']} Hz")
print(f"   - Número de canales: {fnirs_data['n_channels']}")
print(f"   - Nombres de canales: {fnirs_data['ch_names'][:10]}...")

# Estadísticas de datos crudos
data = fnirs_data['data']
print(f"\n5. Estadísticas de datos crudos fNIRS:")
print(f"   - Min: {data.min():.6e}")
print(f"   - Max: {data.max():.6e}")
print(f"   - Mean: {data.mean():.6e}")
print(f"   - Std: {data.std():.6e}")
print(f"   - NaN count: {np.isnan(data).sum()}")
print(f"   - Inf count: {np.isinf(data).sum()}")
print(f"   - Zero count: {(data == 0).sum()} de {data.size} ({100*(data == 0).sum()/data.size:.1f}%)")

# Verificar por canal
print(f"\n6. Estadísticas por canal (primeros 10):")
for i, ch_name in enumerate(fnirs_data['ch_names'][:10]):
    ch_data = data[i, :]
    print(f"   - {ch_name}: min={ch_data.min():.4e}, max={ch_data.max():.4e}, mean={ch_data.mean():.4e}")

# Verificar si hay datos válidos
if data.max() > 0 or data.min() < 0:
    print("\n✅ Los datos crudos fNIRS tienen valores no-cero")
    print("   El problema está en el PREPROCESAMIENTO")
else:
    print("\n❌ Los datos crudos fNIRS están todos en CERO")
    print("   El problema está en los DATOS ORIGINALES o en la EXTRACCIÓN")

# Verificar el stream original
print(f"\n7. Verificando stream original...")
for stream in streams:
    if stream['info']['stream_id'] == fnirs_stream_id:
        info = stream['info']
        print(f"   - Nombre: {info.get('name', ['Unknown'])[0]}")
        print(f"   - Tipo: {info.get('type', ['Unknown'])[0]}")
        print(f"   - Formato: {info.get('channel_format', ['Unknown'])[0]}")
        
        # Verificar datos del stream directamente
        raw_data = stream['time_series']
        print(f"   - Shape time_series: {raw_data.shape}")
        print(f"   - Min: {raw_data.min():.6e}")
        print(f"   - Max: {raw_data.max():.6e}")
        print(f"   - Mean: {raw_data.mean():.6e}")
        break

print("\n" + "=" * 70)
print("FIN DEL DIAGNÓSTICO")
print("=" * 70)
