"""Check event coverage across EEG and fNIRS streams."""

import pyxdf

streams, _ = pyxdf.load_xdf(
    "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
)

# Encontrar streams
eeg_stream = None
marker_stream = None
fnirs_stream = None

for s in streams:
    name = s["info"]["name"][0]
    stype = s["info"]["type"][0]
    if stype == "EEG" and int(s["info"]["channel_count"][0]) > 10:
        eeg_stream = s
    elif name == "eeg_markers":
        marker_stream = s
    elif stype == "NIRS" and "RAW" in name:
        fnirs_stream = s

eeg_start = eeg_stream["time_stamps"][0]
eeg_end = eeg_stream["time_stamps"][-1]
fnirs_start = fnirs_stream["time_stamps"][0]
fnirs_end = fnirs_stream["time_stamps"][-1]

print("=== ANÁLISIS DE COBERTURA TEMPORAL ===")
print(f"EEG: {eeg_start:.2f}s - {eeg_end:.2f}s (duración: {eeg_end-eeg_start:.2f}s)")
print(f"fNIRS: {fnirs_start:.2f}s - {fnirs_end:.2f}s (duración: {fnirs_end-fnirs_start:.2f}s)")
print()

# Analizar eventos
markers = marker_stream["time_series"]
timestamps = marker_stream["time_stamps"]

print("=== EVENTOS Y SU COBERTURA ===")
events_in_eeg = 0
events_in_fnirs_only = 0
events_outside = 0

left_in_eeg = 0
right_in_eeg = 0
nothing_in_eeg = 0

for marker, ts in zip(markers, timestamps):
    event_name = marker[0]
    in_eeg = eeg_start <= ts <= eeg_end
    in_fnirs = fnirs_start <= ts <= fnirs_end

    if in_eeg:
        events_in_eeg += 1
        if "LEFT" in event_name:
            left_in_eeg += 1
        elif "RIGHT" in event_name:
            right_in_eeg += 1
        elif "NOTHING" in event_name:
            nothing_in_eeg += 1
    elif in_fnirs:
        events_in_fnirs_only += 1
    else:
        events_outside += 1

print(f"Eventos dentro del rango EEG: {events_in_eeg}")
print(f"  - LEFT: {left_in_eeg}")
print(f"  - RIGHT: {right_in_eeg}")
print(f"  - NOTHING: {nothing_in_eeg}")
print(f"Eventos solo en fNIRS (sin EEG): {events_in_fnirs_only}")
print(f"Eventos fuera de ambos: {events_outside}")
print()

# Listar eventos con su estado
print("=== DETALLE DE EVENTOS ===")
for marker, ts in zip(markers, timestamps):
    event_name = marker[0]
    ts_rel_eeg = ts - eeg_start
    in_eeg = "✓" if eeg_start <= ts <= eeg_end else "✗"
    in_fnirs = "✓" if fnirs_start <= ts <= fnirs_end else "✗"
    print(f"{event_name:<10} @ {ts_rel_eeg:>8.2f}s (EEG:{in_eeg} fNIRS:{in_fnirs})")
