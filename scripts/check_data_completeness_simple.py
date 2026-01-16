#!/usr/bin/env python3
"""
Verificación mínima de completitud de datos multimodales.

Verifica:
- Presencia de streams fNIRS y EEG
- Duración de grabación
- Número de muestras por modalidad
"""

from pathlib import Path
import pyxdf
import json


def check_subject_data(subject_id: str, session_id: str = "001") -> dict:
    """
    Verifica completitud de datos para un sujeto.

    Args:
        subject_id: ID del sujeto (e.g., "001", "009", "010")
        session_id: ID de sesión (default: "001")

    Returns:
        Diccionario con información de completitud
    """
    # Construir rutas según estructura BIDS
    if int(subject_id) <= 6:
        base_path = Path(f"data/raw/sub-{subject_id}")
    else:
        base_path = Path(f"data/raw/sub-{subject_id}/ses-{session_id}")

    xdf_file = list(base_path.glob("*.xdf"))

    if not xdf_file:
        return {
            "subject": subject_id,
            "exists": False,
            "error": "No XDF file found",
        }

    xdf_file = xdf_file[0]

    # Cargar datos XDF
    streams, _ = pyxdf.load_xdf(str(xdf_file))

    # Identificar streams
    stream_info = {}
    has_fnirs = False
    has_eeg = False
    has_markers = False

    for stream in streams:
        stream_type = stream["info"]["type"][0]
        stream_name = stream["info"]["name"][0]
        n_samples = len(stream["time_stamps"])

        if "NIRSport2" in stream_name or "Photon" in stream_name or stream_type == "NIRS":
            has_fnirs = True
            sfreq = float(stream["info"]["nominal_srate"][0])
            duration_sec = n_samples / sfreq if sfreq > 0 else 0
            stream_info["fnirs"] = {
                "name": stream_name,
                "channels": int(stream["info"]["channel_count"][0]),
                "samples": n_samples,
                "sfreq": sfreq,
                "duration_sec": duration_sec,
            }

        elif "BrainVision" in stream_name or "actiCHamp" in stream_name or stream_type == "EEG":
            has_eeg = True
            sfreq = float(stream["info"]["nominal_srate"][0])
            duration_sec = n_samples / sfreq if sfreq > 0 else 0
            stream_info["eeg"] = {
                "name": stream_name,
                "channels": int(stream["info"]["channel_count"][0]),
                "samples": n_samples,
                "sfreq": sfreq,
                "duration_sec": duration_sec,
            }

        elif stream_type == "Markers":
            has_markers = True
            # Contar eventos en el stream
            n_events = len(stream["time_series"])
            stream_info["markers"] = {
                "name": stream_name,
                "events": n_events,
            }

    # Resumen
    result = {
        "subject": subject_id,
        "session": session_id,
        "exists": True,
        "has_fnirs": has_fnirs,
        "has_eeg": has_eeg,
        "has_markers": has_markers,
        "streams": stream_info,
    }

    return result


def main():
    """Ejecuta verificación de completitud para todos los sujetos."""
    subjects = ["001", "002", "003", "004", "005", "006", "009", "010"]

    print("=" * 80)
    print("VERIFICACIÓN DE COMPLETITUD DE DATOS MULTIMODALES")
    print("=" * 80)
    print()

    all_results = []

    for subject_id in subjects:
        print(f"Verificando sub-{subject_id}...")
        result = check_subject_data(subject_id)
        all_results.append(result)

        if not result["exists"]:
            print(f"  ❌ {result['error']}")
            continue

        # Mostrar resultados
        fnirs_status = "✅" if result["has_fnirs"] else "❌"
        eeg_status = "✅" if result["has_eeg"] else "❌"
        markers_status = "✅" if result["has_markers"] else "❌"

        print(f"  fNIRS:   {fnirs_status}")
        if result["has_fnirs"]:
            fnirs = result["streams"]["fnirs"]
            print(
                f"    - {fnirs['channels']} canales, {fnirs['samples']} muestras, "
                f"{fnirs['sfreq']:.1f} Hz, {fnirs['duration_sec']:.1f}s"
            )

        print(f"  EEG:     {eeg_status}")
        if result["has_eeg"]:
            eeg = result["streams"]["eeg"]
            print(
                f"    - {eeg['channels']} canales, {eeg['samples']} muestras, "
                f"{eeg['sfreq']:.1f} Hz, {eeg['duration_sec']:.1f}s"
            )

        print(f"  Markers: {markers_status}")
        if result["has_markers"]:
            markers = result["streams"]["markers"]
            print(f"    - {markers['events']} eventos")

        print()

    # Resumen final
    print("=" * 80)
    print("RESUMEN")
    print("=" * 80)

    complete_subjects = [
        r for r in all_results if r["exists"] and r["has_fnirs"] and r["has_eeg"]
    ]

    print(f"\nSujetos con datos completos (fNIRS + EEG): {len(complete_subjects)}/{len(subjects)}")
    print()

    if complete_subjects:
        print("Detalles de sujetos completos:")
        print(f"{'Sujeto':<10} {'fNIRS (s)':<12} {'EEG (s)':<12} {'Eventos':<10}")
        print("-" * 50)
        for result in complete_subjects:
            fnirs_dur = result["streams"]["fnirs"]["duration_sec"]
            eeg_dur = result["streams"]["eeg"]["duration_sec"]
            n_events = (
                result["streams"]["markers"]["events"]
                if result["has_markers"]
                else 0
            )
            print(
                f"sub-{result['subject']:<6} {fnirs_dur:<12.1f} {eeg_dur:<12.1f} {n_events:<10}"
            )

    # Identificar sujetos incompletos
    incomplete_subjects = [
        r for r in all_results if r["exists"] and not (r["has_fnirs"] and r["has_eeg"])
    ]

    if incomplete_subjects:
        print("\n⚠️  Sujetos con datos incompletos:")
        for result in incomplete_subjects:
            missing = []
            if not result["has_fnirs"]:
                missing.append("fNIRS")
            if not result["has_eeg"]:
                missing.append("EEG")
            print(f"  - sub-{result['subject']}: Falta {', '.join(missing)}")

    # Guardar resultados en JSON
    output_file = Path("outputs/data_completeness_summary.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResultados guardados en: {output_file}")


if __name__ == "__main__":
    main()
