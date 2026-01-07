"""
Script para investigar los 6 canales extras de fNIRS.

Compara los canales en el XDF con los documentados en el JSON.
"""

import json
from pathlib import Path

from affective_fnirs.ingestion import (
    extract_stream_data,
    identify_streams,
    load_xdf_file,
)

xdf_path = Path(
    "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
)
fnirs_json_path = Path(
    "data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_nirs.json"
)

print("=" * 80)
print("Investigación de Canales Extras de fNIRS")
print("=" * 80)

# Cargar XDF
streams, header = load_xdf_file(xdf_path)
identified = identify_streams(streams)

# Extraer datos fNIRS
fnirs_stream = identified["fnirs"]
fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(fnirs_stream)

print(f"\n1. Información del Stream fNIRS:")
print(f"   - Nombre: {fnirs_stream['info']['name'][0]}")
print(f"   - Canales en XDF: {fnirs_data.shape[1]}")
print(f"   - Frecuencia de muestreo: {fnirs_sfreq:.2f} Hz")

# Cargar JSON
with open(fnirs_json_path, "r") as f:
    fnirs_json = json.load(f)

montage_config = fnirs_json["ChMontage"]
print(f"\n2. Información del JSON:")
print(f"   - Canales documentados: {len(montage_config)}")
print(f"   - NIRSChannelCount: {fnirs_json.get('NIRSChannelCount', 'N/A')}")

# Analizar los canales documentados
print(f"\n3. Análisis de Canales Documentados (JSON):")
long_channels = [ch for ch in montage_config if ch["type"] == "Long"]
short_channels = [ch for ch in montage_config if ch["type"] == "Short"]

print(f"   - Canales largos (30mm): {len(long_channels)}")
print(f"   - Canales cortos (8mm): {len(short_channels)}")

# Contar por longitud de onda
wl_760 = [ch for ch in montage_config if ch["wavelength"] == 760]
wl_850 = [ch for ch in montage_config if ch["wavelength"] == 850]
print(f"   - Canales 760nm: {len(wl_760)}")
print(f"   - Canales 850nm: {len(wl_850)}")

# Identificar los índices documentados
documented_indices = set(ch["channel_idx"] for ch in montage_config)
print(f"\n4. Índices de Canales:")
print(f"   - Documentados en JSON: {min(documented_indices)} a {max(documented_indices)}")
print(f"   - Total en XDF: 0 a {fnirs_data.shape[1] - 1}")

# Identificar los canales NO documentados
all_indices = set(range(fnirs_data.shape[1]))
undocumented_indices = all_indices - documented_indices

print(f"\n5. Canales NO Documentados en JSON:")
print(f"   - Cantidad: {len(undocumented_indices)}")
print(f"   - Índices: {sorted(undocumented_indices)}")

# Analizar los datos de los canales no documentados
if undocumented_indices:
    print(f"\n6. Análisis de Datos de Canales No Documentados:")
    for idx in sorted(undocumented_indices):
        channel_data = fnirs_data[:, idx]
        mean_val = channel_data.mean()
        std_val = channel_data.std()
        min_val = channel_data.min()
        max_val = channel_data.max()
        
        print(f"   - Canal {idx}:")
        print(f"     * Media: {mean_val:.2f}")
        print(f"     * Std: {std_val:.2f}")
        print(f"     * Rango: [{min_val:.2f}, {max_val:.2f}]")
        
        # Verificar si parece un canal de intensidad fNIRS típico
        if mean_val > 1000 and std_val > 10:
            print(f"     * Tipo probable: Canal de intensidad fNIRS activo")
        elif mean_val < 100 and std_val < 10:
            print(f"     * Tipo probable: Canal auxiliar o inactivo")
        else:
            print(f"     * Tipo probable: Desconocido")

# Verificar si hay información adicional en el stream
print(f"\n7. Información Adicional del Stream:")
if "desc" in fnirs_stream["info"]:
    desc = fnirs_stream["info"]["desc"][0]
    print(f"   - Descripción disponible: Sí")
    
    # Buscar información de canales
    if isinstance(desc, dict):
        for key in desc.keys():
            print(f"   - Clave en desc: {key}")
else:
    print(f"   - Descripción disponible: No")

print("\n" + "=" * 80)
print("Conclusión:")
print("=" * 80)
print(f"El stream fNIRS tiene {fnirs_data.shape[1]} canales totales.")
print(f"El JSON documenta {len(montage_config)} canales (índices 0-35).")
print(f"Hay {len(undocumented_indices)} canales adicionales (índices {sorted(undocumented_indices)}).")
print("\nPosibles explicaciones:")
print("1. Canales cortos adicionales no documentados")
print("2. Canales de referencia o calibración")
print("3. Canales auxiliares del dispositivo Photon Cap")
print("4. Canales de control de calidad en tiempo real")
print("=" * 80)
