#!/usr/bin/env python3
"""
Verificación detallada de canales EEG por sujeto.

Extrae y compara los nombres de canales EEG de cada sujeto.
"""

from pathlib import Path
import pyxdf
import json


def extract_eeg_channels(subject_id: str, session_id: str = "001") -> dict:
    """
    Extrae información detallada de canales EEG.

    Args:
        subject_id: ID del sujeto
        session_id: ID de sesión

    Returns:
        Diccionario con información de canales
    """
    # Construir ruta
    if int(subject_id) <= 6:
        base_path = Path(f"data/raw/sub-{subject_id}")
    else:
        base_path = Path(f"data/raw/sub-{subject_id}/ses-{session_id}")

    xdf_file = list(base_path.glob("*.xdf"))

    if not xdf_file:
        return {"subject": subject_id, "error": "No XDF file found"}

    # Cargar XDF
    streams, _ = pyxdf.load_xdf(str(xdf_file[0]))

    # Buscar stream EEG (excluir marcadores)
    for stream in streams:
        stream_type = stream["info"]["type"][0]
        stream_name = stream["info"]["name"][0]

        # Saltar streams de marcadores
        if "Markers" in stream_name or stream_type == "Markers":
            continue

        if "BrainVision" in stream_name or "actiCHamp" in stream_name or stream_type == "EEG":
            # Extraer nombres de canales
            channels = stream["info"]["desc"][0]["channels"][0]["channel"]
            channel_names = []
            channel_info = []

            for ch in channels:
                ch_label = ch["label"][0]
                ch_type = ch.get("type", [None])[0] if "type" in ch else None
                ch_unit = ch.get("unit", [None])[0] if "unit" in ch else None

                channel_names.append(ch_label)
                channel_info.append({
                    "label": ch_label,
                    "type": ch_type,
                    "unit": ch_unit,
                })

            return {
                "subject": subject_id,
                "stream_name": stream_name,
                "n_channels": len(channel_names),
                "channel_names": channel_names,
                "channel_info": channel_info,
            }

    return {"subject": subject_id, "error": "No EEG stream found"}


def main():
    """Ejecuta verificación de canales para todos los sujetos."""
    # Sujetos con EEG
    subjects = ["002", "003", "004", "005", "006", "009", "010"]

    print("=" * 80)
    print("VERIFICACIÓN DETALLADA DE CANALES EEG")
    print("=" * 80)
    print()

    all_results = []

    for subject_id in subjects:
        print(f"Extrayendo canales de sub-{subject_id}...")
        result = extract_eeg_channels(subject_id)
        all_results.append(result)

        if "error" in result:
            print(f"  ❌ {result['error']}")
            continue

        print(f"  Stream: {result['stream_name']}")
        print(f"  Canales: {result['n_channels']}")
        print(f"  Nombres: {', '.join(result['channel_names'])}")
        print()

    # Análisis comparativo
    print("=" * 80)
    print("ANÁLISIS COMPARATIVO")
    print("=" * 80)
    print()

    # Agrupar por número de canales
    by_n_channels = {}
    for result in all_results:
        if "error" not in result:
            n_ch = result["n_channels"]
            if n_ch not in by_n_channels:
                by_n_channels[n_ch] = []
            by_n_channels[n_ch].append(result)

    for n_ch, subjects_group in sorted(by_n_channels.items()):
        print(f"\nSujetos con {n_ch} canales:")
        for result in subjects_group:
            print(f"  - sub-{result['subject']}")

    # Comparar canales entre grupos
    if 32 in by_n_channels and 30 in by_n_channels:
        print("\n" + "=" * 80)
        print("DIFERENCIAS ENTRE GRUPOS")
        print("=" * 80)

        # Tomar un representante de cada grupo
        ch_32 = set(by_n_channels[32][0]["channel_names"])
        ch_30 = set(by_n_channels[30][0]["channel_names"])

        missing_in_30 = ch_32 - ch_30
        extra_in_30 = ch_30 - ch_32

        print(f"\nCanales en grupo de 32 pero NO en grupo de 30:")
        if missing_in_30:
            print(f"  {', '.join(sorted(missing_in_30))}")
        else:
            print("  (ninguno)")

        print(f"\nCanales en grupo de 30 pero NO en grupo de 32:")
        if extra_in_30:
            print(f"  {', '.join(sorted(extra_in_30))}")
        else:
            print("  (ninguno)")

        print(f"\nCanales comunes: {len(ch_32 & ch_30)}")

    # Guardar resultados
    output_file = Path("outputs/eeg_channels_detail.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResultados guardados en: {output_file}")


if __name__ == "__main__":
    main()
