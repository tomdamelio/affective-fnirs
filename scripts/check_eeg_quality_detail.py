"""Análisis detallado de calidad EEG para los trials disponibles."""

from pathlib import Path

import json
import matplotlib.pyplot as plt
import mne
import numpy as np
import pyxdf
from scipy import signal


def main():
    """Analizar calidad EEG en detalle."""
    xdf_path = Path(
        "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
    )
    eeg_json_path = Path(
        "data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_eeg.json"
    )

    print("=" * 70)
    print("ANÁLISIS DETALLADO DE CALIDAD EEG - sub-002")
    print("=" * 70)

    # Cargar XDF directamente
    print("\n[1] Cargando datos...")
    streams, _ = pyxdf.load_xdf(str(xdf_path))

    # Identificar streams
    eeg_stream = None
    marker_stream = None

    for s in streams:
        stype = s["info"]["type"][0]
        name = s["info"]["name"][0]
        if stype == "EEG" and int(s["info"]["channel_count"][0]) > 10:
            eeg_stream = s
        elif name == "eeg_markers":
            marker_stream = s

    # Cargar config EEG
    with open(eeg_json_path) as f:
        eeg_config = json.load(f)

    # Crear Raw EEG manualmente
    eeg_data = np.array(eeg_stream["time_series"]).T  # (n_channels, n_times)
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])
    
    # Obtener nombres de canales del JSON o del stream
    if "channel_names" in eeg_config:
        ch_names = eeg_config["channel_names"]
    elif "Channels" in eeg_config:
        ch_names = [ch.strip() for ch in eeg_config["Channels"].split(",")]
    else:
        # Extraer del stream XDF
        ch_names = [
            ch["label"][0] for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        ]
    
    # Filtrar canales AUX si hay más canales en data que en ch_names
    n_data_channels = eeg_data.shape[0]
    if len(ch_names) > n_data_channels:
        ch_names = ch_names[:n_data_channels]
    elif len(ch_names) < n_data_channels:
        # Añadir nombres genéricos para canales extra
        for i in range(len(ch_names), n_data_channels):
            ch_names.append(f"AUX_{i}")
    
    # Filtrar solo canales EEG (no AUX)
    eeg_ch_mask = [not ch.startswith("AUX") for ch in ch_names]
    ch_names_eeg = [ch for ch, mask in zip(ch_names, eeg_ch_mask) if mask]
    eeg_data_filtered = eeg_data[eeg_ch_mask, :]
    
    print(f"Canales totales: {len(ch_names)}, EEG: {len(ch_names_eeg)}")

    # Escalar a Volts si es necesario (datos en µV)
    if np.abs(eeg_data).max() > 1:
        eeg_data = eeg_data * 1e-6  # µV -> V

    info = mne.create_info(ch_names=ch_names_eeg, sfreq=sfreq, ch_types="eeg")
    raw_eeg = mne.io.RawArray(eeg_data_filtered, info)

    # Añadir montage estándar
    montage = mne.channels.make_standard_montage("standard_1020")
    raw_eeg.set_montage(montage, on_missing="warn")

    print(f"EEG Raw: {raw_eeg.info['nchan']} canales, {raw_eeg.n_times} muestras")
    print(f"Duración: {raw_eeg.times[-1]:.2f}s")

    # Añadir annotations desde markers
    eeg_start_time = eeg_stream["time_stamps"][0]
    eeg_end_time = eeg_stream["time_stamps"][-1]

    onsets = []
    descriptions = []
    for marker, ts in zip(
        marker_stream["time_series"], marker_stream["time_stamps"]
    ):
        if eeg_start_time <= ts <= eeg_end_time:
            onset = ts - eeg_start_time
            onsets.append(onset)
            descriptions.append(marker[0])

    annotations = mne.Annotations(onset=onsets, duration=[0] * len(onsets), description=descriptions)
    raw_eeg.set_annotations(annotations)

    print(f"Eventos embebidos: {len(onsets)}")

    # Análisis de calidad por canal
    print("\n[2] ANÁLISIS DE CALIDAD POR CANAL:")
    print("-" * 70)

    data = raw_eeg.get_data()

    print(
        f"{'Canal':<8} {'Std(µV)':>10} {'PkPk(µV)':>12} {'Flat%':>8} {'60Hz':>10} {'Status'}"
    )
    print("-" * 70)

    bad_channels = []
    for i, ch_name in enumerate(raw_eeg.ch_names):
        ch_data = data[i, :]

        # Convertir a µV para display
        ch_data_uv = ch_data * 1e6

        std_uv = np.std(ch_data_uv)
        pkpk_uv = np.ptp(ch_data_uv)

        # Detectar señal plana
        diff = np.diff(ch_data)
        flat_pct = np.sum(np.abs(diff) < 1e-10) / len(diff) * 100

        # Detectar ruido de línea (50/60 Hz)
        freqs, psd = signal.welch(ch_data, fs=sfreq, nperseg=int(sfreq * 2))
        idx_60 = np.argmin(np.abs(freqs - 60))
        idx_50 = np.argmin(np.abs(freqs - 50))
        line_noise = max(psd[idx_50], psd[idx_60])
        total_power = np.sum(psd)
        line_ratio = line_noise / total_power * 100 if total_power > 0 else 0

        # Evaluar calidad
        issues = []
        if std_uv < 1:
            issues.append("low_var")
        if std_uv > 200:
            issues.append("high_var")
        if flat_pct > 10:
            issues.append("flat")
        if line_ratio > 5:
            issues.append("line_noise")

        status = "BAD" if issues else "OK"
        if issues:
            bad_channels.append(ch_name)

        print(
            f"{ch_name:<8} {std_uv:>10.2f} {pkpk_uv:>12.2f} {flat_pct:>7.1f}% {line_ratio:>9.2f}% {status} {issues}"
        )

    print(f"\nCanales malos detectados: {len(bad_channels)}/{len(raw_eeg.ch_names)}")

    # Análisis de espectro en canales motores
    print("\n[3] ESPECTRO EN CANALES MOTORES (C3, C4, Cz):")
    print("-" * 70)

    motor_channels = ["C3", "C4", "Cz"]
    available_motor = [ch for ch in motor_channels if ch in raw_eeg.ch_names]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, ch_name in enumerate(available_motor):
        ch_idx = raw_eeg.ch_names.index(ch_name)
        ch_data = data[ch_idx, :]

        freqs, psd = signal.welch(ch_data, fs=sfreq, nperseg=int(sfreq * 4))

        ax = axes[i // 2, i % 2]
        ax.semilogy(freqs, psd)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"Espectro: {ch_name}")
        ax.set_xlim([0, 50])
        ax.axvspan(8, 13, alpha=0.3, color="green", label="Alpha (8-13 Hz)")
        ax.axvspan(13, 30, alpha=0.3, color="blue", label="Beta (13-30 Hz)")
        ax.legend()

        # Calcular potencia en bandas
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        alpha_power = np.sum(psd[alpha_mask])
        beta_power = np.sum(psd[beta_mask])
        total_power = np.sum(psd[(freqs >= 1) & (freqs <= 40)])

        print(f"{ch_name}:")
        print(
            f"  Alpha power: {alpha_power:.2e} ({alpha_power/total_power*100:.1f}%)"
        )
        print(f"  Beta power: {beta_power:.2e} ({beta_power/total_power*100:.1f}%)")

    # Crear epochs y analizar ERD
    print("\n[4] ANÁLISIS DE ERD EN EPOCHS:")
    print("-" * 70)

    # Filtrar primero
    raw_filt = raw_eeg.copy().filter(1, 40, fir_design="firwin", verbose=False)

    events, event_id = mne.events_from_annotations(raw_filt)
    print(f"Eventos encontrados: {len(events)}")
    print(f"Event IDs: {event_id}")

    # Filtrar solo LEFT y RIGHT
    task_event_ids = {k: v for k, v in event_id.items() if "LEFT" in k or "RIGHT" in k}
    print(f"Eventos de tarea: {task_event_ids}")

    alpha_erd = 0
    beta_erd = 0

    if task_event_ids:
        # Crear epochs
        epochs = mne.Epochs(
            raw_filt,
            events,
            event_id=task_event_ids,
            tmin=-2,
            tmax=10,
            baseline=(-2, 0),
            preload=True,
            reject=None,
            verbose=False,
        )
        print(f"Epochs creados: {len(epochs)}")

        # Calcular TFR para C3
        if "C3" in raw_filt.ch_names:
            print("\nCalculando TFR para C3...")
            freqs_tfr = np.arange(4, 35, 1)
            n_cycles = freqs_tfr / 2.0

            tfr = mne.time_frequency.tfr_morlet(
                epochs,
                freqs=freqs_tfr,
                n_cycles=n_cycles,
                picks=["C3"],
                return_itc=False,
                average=True,
                verbose=False,
            )

            # Aplicar baseline
            tfr.apply_baseline(baseline=(-2, -0.5), mode="percent")

            # Extraer ERD en alpha y beta
            alpha_idx = (freqs_tfr >= 8) & (freqs_tfr <= 13)
            beta_idx = (freqs_tfr >= 13) & (freqs_tfr <= 30)

            # Durante movimiento (0.5-3s)
            time_idx = (tfr.times >= 0.5) & (tfr.times <= 3)

            tfr_data = tfr.data[0, :, :]  # Canal C3

            alpha_erd = np.mean(tfr_data[alpha_idx, :][:, time_idx]) * 100
            beta_erd = np.mean(tfr_data[beta_idx, :][:, time_idx]) * 100

            print(f"\nERD durante movimiento (0.5-3s):")
            print(f"  Alpha ERD: {alpha_erd:.1f}%")
            print(f"  Beta ERD: {beta_erd:.1f}%")

            if alpha_erd > -10:
                print("  ⚠️ Alpha ERD muy débil (esperado: -20% a -50%)")
            if beta_erd > -10:
                print("  ⚠️ Beta ERD muy débil (esperado: -20% a -40%)")

            # Plot TFR
            ax = axes[1, 1]
            im = ax.imshow(
                tfr_data * 100,
                aspect="auto",
                origin="lower",
                extent=[tfr.times[0], tfr.times[-1], freqs_tfr[0], freqs_tfr[-1]],
                cmap="RdBu_r",
                vmin=-50,
                vmax=50,
            )
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Frecuencia (Hz)")
            ax.set_title("TFR C3 (% cambio vs baseline)")
            ax.axvline(0, color="black", linestyle="--", label="Onset")
            plt.colorbar(im, ax=ax, label="% cambio")

    plt.tight_layout()
    output_path = Path("outputs/sub-002_eeg_quality.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nFigura guardada: {output_path}")
    plt.close()

    # Resumen
    print("\n" + "=" * 70)
    print("CONCLUSIONES")
    print("=" * 70)

    if len(bad_channels) > len(raw_eeg.ch_names) * 0.5:
        print("⚠️ Más del 50% de canales EEG tienen problemas de calidad")
        print("   Posibles causas: mal contacto de electrodos, impedancias altas")

    if alpha_erd > -10 and beta_erd > -10:
        print("⚠️ No se detecta ERD significativo")
        print("   Posibles causas:")
        print("   - El sujeto no ejecutó la tarea correctamente")
        print("   - Mala calidad de señal en canales motores")
        print("   - Parámetros de análisis inadecuados")
    else:
        print("✅ Se detecta ERD en rangos esperados")


if __name__ == "__main__":
    main()
