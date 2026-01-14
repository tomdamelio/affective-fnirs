#!/usr/bin/env python3
"""
Script para verificar los eventos en el archivo XDF de sub-010.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.ingestion import load_xdf_file, identify_streams

# Cargar XDF
xdf_path = Path("data/raw/sub-010/sub-010_ses-001_task-fingertapping_recording.xdf")
print(f"Cargando: {xdf_path}")

streams, header = load_xdf_file(xdf_path)
print(f"\nStreams encontrados: {len(streams)}")

# Identificar streams
identified = identify_streams(streams)

# Buscar el stream de markers
markers_stream = identified.get("markers")

if markers_stream:
    print(f"\n{'='*70}")
    print("EVENTOS ENCONTRADOS EN EL STREAM DE MARKERS")
    print(f"{'='*70}")
    
    # Extraer eventos únicos
    marker_data = markers_stream["time_series"]
    unique_events = set()
    
    for marker in marker_data:
        if isinstance(marker, list) and len(marker) > 0:
            unique_events.add(marker[0])
        elif isinstance(marker, str):
            unique_events.add(marker)
    
    print(f"\nEventos únicos ({len(unique_events)}):")
    for event in sorted(unique_events):
        # Contar cuántas veces aparece
        count = sum(1 for m in marker_data if (isinstance(m, list) and len(m) > 0 and m[0] == event) or m == event)
        print(f"  - '{event}': {count} veces")
    
    print(f"\n{'='*70}")
    print("PRIMEROS 10 EVENTOS (con timestamps):")
    print(f"{'='*70}")
    
    timestamps = markers_stream["time_stamps"]
    for i, (marker, ts) in enumerate(zip(marker_data[:10], timestamps[:10])):
        event_name = marker[0] if isinstance(marker, list) and len(marker) > 0 else marker
        print(f"{i+1}. t={ts:.2f}s: '{event_name}'")
    
else:
    print("\n❌ No se encontró stream de markers")
