"""
Diagnóstico de datos crudos para sub-002.

Revisa:
1. Estructura del XDF y streams
2. Alineación temporal de markers
3. Calidad de señales EEG y fNIRS crudas
4. Distribución de eventos
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyxdf


def main():
    """Diagnóstico completo de sub-002."""
    xdf_path = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
    
    print("=" * 70)
    print("DIAGNÓSTICO DE DATOS CRUDOS - sub-002")
    print("=" * 70)
    
    # 1. Cargar XDF
    print("\n[1] CARGANDO XDF...")
    streams, header = pyxdf.load_xdf(str(xdf_path))
    print(f"Archivo: {xdf_path}")
    print(f"Streams encontrados: {len(streams)}")
    
    # 2. Analizar cada stream
    print("\n[2] ANÁLISIS DE STREAMS:")
    print("-" * 70)
    
    eeg_stream = None
    fnirs_stream = None
    marker_streams = []
    
    for stream in streams:
        info = stream["info"]
        name = info["name"][0]
        stream_type = info["type"][0]
        n_channels = int(info["channel_count"][0])
        srate = float(info["nominal_srate"][0])
        n_samples = len(stream["time_stamps"])
        
        if n_samples > 0:
            duration = stream["time_stamps"][-1] - stream["time_stamps"][0]
            first_ts = stream["time_stamps"][0]
            last_ts = stream["time_stamps"][-1]
        else:
            duration = 0
            first_ts = last_ts = 0
        
        print(f"\nStream: {name}")
        print(f"  Tipo: {stream_type}")
        print(f"  Canales: {n_channels}")
        print(f"  Srate nominal: {srate} Hz")
        print(f"  Muestras: {n_samples}")
        print(f"  Duración: {duration:.2f}s")
        print(f"  Timestamps: [{first_ts:.2f}, {last_ts:.2f}]")
        
        # Identificar streams principales
        if stream_type == "EEG" and n_channels > 10:
            eeg_stream = stream
            print("  >>> IDENTIFICADO COMO EEG")
        elif "NIR" in stream_type.upper() or "FNIRS" in name.upper():
            fnirs_stream = stream
            print("  >>> IDENTIFICADO COMO fNIRS")
        elif stream_type == "Markers" or n_channels == 1 and srate == 0:
            marker_streams.append(stream)
            print("  >>> IDENTIFICADO COMO MARKERS")
    
    # 3. Análisis de markers
    print("\n[3] ANÁLISIS DE MARKERS:")
    print("-" * 70)
    
    all_events = []
    for marker_stream in marker_streams:
        name = marker_stream["info"]["name"][0]
        markers = marker_stream["time_series"]
        timestamps = marker_stream["time_stamps"]
        
        print(f"\nMarker stream: {name}")
        print(f"  Total eventos: {len(timestamps)}")
        
        if len(timestamps) > 0:
            # Contar tipos de eventos
            event_types = {}
            for marker, ts in zip(markers, timestamps):
                event_name = marker[0] if isinstance(marker, (list, np.ndarray)) else str(marker)
                if event_name not in event_types:
                    event_types[event_name] = []
                event_types[event_name].append(ts)
                all_events.append((ts, event_name))
            
            print(f"  Tipos de eventos:")
            for event_name, times in sorted(event_types.items()):
                print(f"    - '{event_name}': {len(times)} eventos")
                if len(times) <= 5:
                    for t in times:
                        print(f"        @ {t:.2f}s")
                else:
                    print(f"        Primeros: {times[0]:.2f}s, {times[1]:.2f}s, ...")
                    print(f"        Últimos: ..., {times[-2]:.2f}s, {times[-1]:.2f}s")
    
    # 4. Verificar alineación temporal
    print("\n[4] ALINEACIÓN TEMPORAL:")
    print("-" * 70)
    
    if eeg_stream is not None:
        eeg_start = eeg_stream["time_stamps"][0]
        eeg_end = eeg_stream["time_stamps"][-1]
        print(f"EEG: [{eeg_start:.2f}s, {eeg_end:.2f}s] (duración: {eeg_end - eeg_start:.2f}s)")
    
    if fnirs_stream is not None:
        fnirs_start = fnirs_stream["time_stamps"][0]
        fnirs_end = fnirs_stream["time_stamps"][-1]
        print(f"fNIRS: [{fnirs_start:.2f}s, {fnirs_end:.2f}s] (duración: {fnirs_end - fnirs_start:.2f}s)")
    
    # Verificar eventos dentro del rango de datos
    if all_events and eeg_stream is not None:
        events_in_range = 0
        events_out_range = 0
        
        print(f"\nEventos vs rango de datos:")
        for ts, event_name in sorted(all_events):
            in_eeg = eeg_start <= ts <= eeg_end
            in_fnirs = fnirs_start <= ts <= fnirs_end if fnirs_stream else False
            
            if in_eeg or in_fnirs:
                events_in_range += 1
            else:
                events_out_range += 1
                print(f"  ⚠️  '{event_name}' @ {ts:.2f}s - FUERA DE RANGO")
        
        print(f"\n  Eventos dentro del rango: {events_in_range}")
        print(f"  Eventos fuera del rango: {events_out_range}")
        
        if events_out_range > 0:
            print("\n  ⚠️  PROBLEMA: Hay eventos fuera del rango de grabación!")
            print("     Esto puede indicar problemas de sincronización.")
    
    # 5. Análisis de señal EEG cruda
    print("\n[5] CALIDAD DE SEÑAL EEG:")
    print("-" * 70)
    
    if eeg_stream is not None:
        eeg_data = np.array(eeg_stream["time_series"])
        print(f"Shape: {eeg_data.shape}")
        
        # Estadísticas por canal
        print("\nEstadísticas por canal (primeros 10):")
        print(f"{'Canal':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Flat%':>8}")
        
        n_flat_channels = 0
        for i in range(min(10, eeg_data.shape[1])):
            ch_data = eeg_data[:, i]
            ch_mean = np.mean(ch_data)
            ch_std = np.std(ch_data)
            ch_min = np.min(ch_data)
            ch_max = np.max(ch_data)
            
            # Detectar señal plana
            diff = np.diff(ch_data)
            flat_pct = np.sum(diff == 0) / len(diff) * 100
            
            if flat_pct > 50:
                n_flat_channels += 1
                flag = "⚠️"
            else:
                flag = ""
            
            print(f"Ch{i:<7} {ch_mean:>12.2f} {ch_std:>12.2f} {ch_min:>12.2f} {ch_max:>12.2f} {flat_pct:>7.1f}% {flag}")
        
        # Rango dinámico global
        global_range = np.max(eeg_data) - np.min(eeg_data)
        print(f"\nRango dinámico global: {global_range:.2f}")
        print(f"Canales con >50% señal plana: {n_flat_channels}")
    
    # 6. Análisis de señal fNIRS cruda
    print("\n[6] CALIDAD DE SEÑAL fNIRS:")
    print("-" * 70)
    
    if fnirs_stream is not None:
        fnirs_data = np.array(fnirs_stream["time_series"])
        print(f"Shape: {fnirs_data.shape}")
        
        # Estadísticas por canal
        print("\nEstadísticas por canal (primeros 10):")
        print(f"{'Canal':<10} {'Mean':>12} {'Std':>12} {'CV%':>10} {'Status'}")
        
        for i in range(min(10, fnirs_data.shape[1])):
            ch_data = fnirs_data[:, i]
            ch_mean = np.mean(ch_data)
            ch_std = np.std(ch_data)
            cv = (ch_std / ch_mean * 100) if ch_mean != 0 else 0
            
            # Evaluar calidad
            if ch_mean <= 0:
                status = "⚠️ Negativo/Zero"
            elif cv > 10:
                status = "⚠️ Alta variabilidad"
            else:
                status = "OK"
            
            print(f"Ch{i:<7} {ch_mean:>12.2f} {ch_std:>12.4f} {cv:>9.2f}% {status}")
    
    # 7. Generar figura de diagnóstico
    print("\n[7] GENERANDO FIGURA DE DIAGNÓSTICO...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Diagnóstico sub-002: Datos Crudos", fontsize=14)
    
    # EEG: primeros 5 canales
    if eeg_stream is not None:
        eeg_times = np.array(eeg_stream["time_stamps"]) - eeg_stream["time_stamps"][0]
        ax = axes[0, 0]
        for i in range(min(5, eeg_data.shape[1])):
            ax.plot(eeg_times[::10], eeg_data[::10, i] + i * 100, alpha=0.7, linewidth=0.5)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Canales EEG (offset)")
        ax.set_title("EEG: Primeros 5 canales")
        ax.set_xlim([0, min(60, eeg_times[-1])])
    
    # EEG: espectro de potencia
    if eeg_stream is not None:
        ax = axes[0, 1]
        srate = float(eeg_stream["info"]["nominal_srate"][0])
        # Canal C3 o similar (índice ~10)
        ch_idx = min(10, eeg_data.shape[1] - 1)
        from scipy import signal
        freqs, psd = signal.welch(eeg_data[:, ch_idx], fs=srate, nperseg=int(srate * 2))
        ax.semilogy(freqs, psd)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"EEG: Espectro canal {ch_idx}")
        ax.set_xlim([0, 50])
        ax.axvspan(8, 13, alpha=0.3, color="green", label="Alpha")
        ax.axvspan(13, 30, alpha=0.3, color="blue", label="Beta")
        ax.legend()
    
    # fNIRS: primeros 5 canales
    if fnirs_stream is not None:
        fnirs_times = np.array(fnirs_stream["time_stamps"]) - fnirs_stream["time_stamps"][0]
        ax = axes[1, 0]
        for i in range(min(5, fnirs_data.shape[1])):
            normalized = (fnirs_data[:, i] - np.mean(fnirs_data[:, i])) / np.std(fnirs_data[:, i])
            ax.plot(fnirs_times, normalized + i * 5, alpha=0.7, linewidth=0.5)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Canales fNIRS (z-score, offset)")
        ax.set_title("fNIRS: Primeros 5 canales")
    
    # fNIRS: histograma de valores
    if fnirs_stream is not None:
        ax = axes[1, 1]
        ax.hist(fnirs_data.flatten(), bins=100, alpha=0.7)
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Frecuencia")
        ax.set_title("fNIRS: Distribución de intensidades")
        ax.set_yscale("log")
    
    # Timeline de eventos
    ax = axes[2, 0]
    if all_events and eeg_stream is not None:
        event_colors = {"LEFT": "red", "RIGHT": "blue", "NOTHING": "gray", "REST": "green"}
        for ts, event_name in all_events:
            # Normalizar timestamp al inicio de EEG
            ts_rel = ts - eeg_start
            color = event_colors.get(event_name.split("/")[0], "black")
            ax.axvline(ts_rel, color=color, alpha=0.5, linewidth=1)
        
        ax.set_xlabel("Tiempo relativo a EEG (s)")
        ax.set_title("Timeline de eventos")
        ax.set_xlim([-10, eeg_end - eeg_start + 10])
        
        # Leyenda manual
        for name, color in event_colors.items():
            ax.axvline(-100, color=color, label=name)
        ax.legend(loc="upper right")
    
    # ISI (Inter-Stimulus Interval)
    ax = axes[2, 1]
    if all_events:
        # Filtrar solo eventos de tarea (LEFT, RIGHT)
        task_events = [(ts, name) for ts, name in all_events if "LEFT" in name or "RIGHT" in name]
        if len(task_events) > 1:
            task_times = sorted([ts for ts, _ in task_events])
            isis = np.diff(task_times)
            ax.hist(isis, bins=20, alpha=0.7)
            ax.axvline(np.mean(isis), color="red", linestyle="--", label=f"Mean: {np.mean(isis):.1f}s")
            ax.set_xlabel("ISI (s)")
            ax.set_ylabel("Frecuencia")
            ax.set_title("Inter-Stimulus Interval (eventos de tarea)")
            ax.legend()
    
    plt.tight_layout()
    output_path = Path("outputs/sub-002_diagnostic.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Figura guardada: {output_path}")
    plt.close()
    
    # 8. Resumen y recomendaciones
    print("\n" + "=" * 70)
    print("RESUMEN Y RECOMENDACIONES")
    print("=" * 70)
    
    issues = []
    
    if events_out_range > 0:
        issues.append("- Eventos fuera del rango de grabación (problema de sincronización)")
    
    if eeg_stream is not None and n_flat_channels > 5:
        issues.append(f"- {n_flat_channels} canales EEG con señal plana (mal contacto)")
    
    if fnirs_stream is not None:
        neg_channels = np.sum(np.mean(fnirs_data, axis=0) <= 0)
        if neg_channels > 0:
            issues.append(f"- {neg_channels} canales fNIRS con valores negativos/cero")
    
    if issues:
        print("\n⚠️  PROBLEMAS DETECTADOS:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ No se detectaron problemas críticos en los datos crudos.")
    
    print("\n")


if __name__ == "__main__":
    main()
