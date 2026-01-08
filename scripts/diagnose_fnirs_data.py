"""Diagnóstico de datos fNIRS para sub-002."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyxdf

# Load XDF
streams, _ = pyxdf.load_xdf(
    "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
)

# Find fNIRS stream
fnirs_stream = None
for s in streams:
    if s["info"]["type"][0] == "NIRS" and "RAW" in s["info"]["name"][0]:
        fnirs_stream = s
        break

if fnirs_stream is None:
    print("No fNIRS stream found!")
    exit(1)

data = np.array(fnirs_stream["time_series"])
timestamps = np.array(fnirs_stream["time_stamps"])
srate = float(fnirs_stream["info"]["nominal_srate"][0])

print("=" * 70)
print("DIAGNÓSTICO DE DATOS fNIRS - sub-002")
print("=" * 70)

print(f"\nShape: {data.shape}")
print(f"Sampling rate: {srate:.2f} Hz")
print(f"Duration: {(timestamps[-1] - timestamps[0]):.2f}s")
print(f"N samples: {len(timestamps)}")

# Estadísticas globales
print(f"\n=== ESTADÍSTICAS GLOBALES ===")
print(f"Min: {np.min(data):.6f}")
print(f"Max: {np.max(data):.6f}")
print(f"Mean: {np.mean(data):.6f}")
print(f"Std: {np.std(data):.6f}")

# Analizar por canal
print(f"\n=== ANÁLISIS POR CANAL (primeros 10) ===")
print(f"{'Ch':<4} {'Mean':>12} {'Std':>12} {'CV%':>8} {'Min':>12} {'Max':>12} {'Status'}")
print("-" * 70)

n_negative = 0
n_zero = 0
n_high_cv = 0

for i in range(min(10, data.shape[1])):
    ch_data = data[:, i]
    ch_mean = np.mean(ch_data)
    ch_std = np.std(ch_data)
    ch_min = np.min(ch_data)
    ch_max = np.max(ch_data)
    cv = (ch_std / ch_mean * 100) if ch_mean != 0 else 0
    
    status = []
    if ch_mean <= 0:
        status.append("NEGATIVE")
        n_negative += 1
    if ch_std < 1e-6:
        status.append("FLAT")
        n_zero += 1
    if cv > 10:
        status.append("HIGH_CV")
        n_high_cv += 1
    
    status_str = ",".join(status) if status else "OK"
    
    print(f"{i:<4} {ch_mean:>12.6f} {ch_std:>12.6f} {cv:>7.2f}% {ch_min:>12.6f} {ch_max:>12.6f} {status_str}")

print(f"\nCanales con problemas:")
print(f"  - Negativos/cero: {n_negative}")
print(f"  - Planos: {n_zero}")
print(f"  - Alta variabilidad (CV>10%): {n_high_cv}")

# Verificar unidades
print(f"\n=== VERIFICACIÓN DE UNIDADES ===")
print(f"Valores típicos de intensidad fNIRS:")
print(f"  - Rango esperado: 0.01 - 0.05 (unidades arbitrarias)")
print(f"  - CV esperado: 0.1% - 5%")

if np.mean(data) < 0.001:
    print(f"\n⚠️ Valores muy bajos - posible problema de calibración")
elif np.mean(data) > 0.1:
    print(f"\n⚠️ Valores muy altos - verificar unidades")
else:
    print(f"\n✅ Valores en rango esperado")

# Analizar eventos y respuesta hemodinámica
print(f"\n=== ANÁLISIS DE RESPUESTA HEMODINÁMICA ===")

# Encontrar marker stream
marker_stream = None
for s in streams:
    if s["info"]["name"][0] == "eeg_markers":
        marker_stream = s
        break

if marker_stream:
    fnirs_start = timestamps[0]
    fnirs_end = timestamps[-1]
    
    # Contar eventos dentro del rango fNIRS
    events_in_range = 0
    left_events = []
    
    for marker, ts in zip(marker_stream["time_series"], marker_stream["time_stamps"]):
        if fnirs_start <= ts <= fnirs_end:
            events_in_range += 1
            if "LEFT" in marker[0]:
                left_events.append(ts - fnirs_start)
    
    print(f"Eventos en rango fNIRS: {events_in_range}")
    print(f"Eventos LEFT: {len(left_events)}")
    
    if len(left_events) > 0:
        # Analizar respuesta promedio en un canal
        # Usar canal 0 como ejemplo
        ch_idx = 0
        ch_data = data[:, ch_idx]
        
        # Crear epochs alrededor de eventos LEFT
        epoch_duration = 30  # segundos
        epoch_samples = int(epoch_duration * srate)
        
        epochs = []
        for event_time in left_events[:5]:  # Primeros 5 eventos
            event_sample = int(event_time * srate)
            if event_sample + epoch_samples < len(ch_data):
                epoch = ch_data[event_sample : event_sample + epoch_samples]
                epochs.append(epoch)
        
        if epochs:
            epochs_array = np.array(epochs)
            mean_response = np.mean(epochs_array, axis=0)
            
            # Calcular baseline y cambio
            baseline = np.mean(mean_response[:int(5 * srate)])  # Primeros 5s
            peak = np.max(mean_response)
            change_pct = (peak - baseline) / baseline * 100 if baseline != 0 else 0
            
            print(f"\nRespuesta promedio (canal {ch_idx}, {len(epochs)} trials):")
            print(f"  Baseline: {baseline:.6f}")
            print(f"  Peak: {peak:.6f}")
            print(f"  Cambio: {change_pct:.2f}%")
            
            if abs(change_pct) < 0.5:
                print(f"  ⚠️ Cambio muy pequeño - posible problema de calidad")
            elif change_pct > 0:
                print(f"  ✅ Aumento detectado (esperado para HbO)")
            else:
                print(f"  ⚠️ Disminución detectada (inesperado para HbO)")

# Generar figura
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Primeros 5 canales (primeros 60s)
ax = axes[0, 0]
time_rel = timestamps - timestamps[0]
plot_duration = min(60, time_rel[-1])
plot_mask = time_rel <= plot_duration

for i in range(min(5, data.shape[1])):
    ax.plot(time_rel[plot_mask], data[plot_mask, i], alpha=0.7, linewidth=0.5, label=f"Ch{i}")
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Intensidad")
ax.set_title("fNIRS: Primeros 5 canales (60s)")
ax.legend(loc="upper right", fontsize=8)

# Plot 2: Distribución de valores
ax = axes[0, 1]
ax.hist(data.flatten(), bins=100, alpha=0.7)
ax.set_xlabel("Intensidad")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de intensidades")
ax.set_yscale("log")

# Plot 3: CV por canal
ax = axes[1, 0]
cvs = []
for i in range(data.shape[1]):
    ch_mean = np.mean(data[:, i])
    ch_std = np.std(data[:, i])
    cv = (ch_std / ch_mean * 100) if ch_mean != 0 else 0
    cvs.append(cv)
ax.bar(range(len(cvs)), cvs, alpha=0.7)
ax.axhline(5, color="red", linestyle="--", label="CV=5% (threshold)")
ax.set_xlabel("Canal")
ax.set_ylabel("CV (%)")
ax.set_title("Coeficiente de Variación por canal")
ax.legend()

# Plot 4: Respuesta promedio si hay eventos
ax = axes[1, 1]
if len(left_events) > 0 and epochs:
    time_epoch = np.arange(len(mean_response)) / srate
    ax.plot(time_epoch, mean_response, linewidth=2)
    ax.axvline(0, color="red", linestyle="--", label="Evento")
    ax.axhline(baseline, color="gray", linestyle=":", label="Baseline")
    ax.set_xlabel("Tiempo desde evento (s)")
    ax.set_ylabel("Intensidad")
    ax.set_title(f"Respuesta promedio (canal {ch_idx}, n={len(epochs)})")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No hay eventos para analizar", ha="center", va="center")
    ax.set_title("Respuesta promedio")

plt.tight_layout()
output_path = Path("outputs/sub-002_fnirs_diagnostic.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150)
print(f"\nFigura guardada: {output_path}")
plt.close()

print("\n" + "=" * 70)
