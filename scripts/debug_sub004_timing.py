"""Deep investigation of sub-004 timing/synchronization issues."""

from pathlib import Path
import pyxdf
import numpy as np

xdf_file = Path("data/raw/sub-004/sub-004_ses-001_task-fingertapping_recording.xdf")

print("=" * 70)
print("SUB-004 TIMING INVESTIGATION")
print("=" * 70)

streams, header = pyxdf.load_xdf(str(xdf_file))

print(f"\nTotal streams: {len(streams)}")
print("\n" + "=" * 70)
print("STREAM TIMING ANALYSIS")
print("=" * 70)

for stream in streams:
    name = stream["info"]["name"][0]
    stype = stream["info"]["type"][0]
    
    if "time_stamps" in stream and len(stream["time_stamps"]) > 0:
        ts = stream["time_stamps"]
        first_ts = ts[0]
        last_ts = ts[-1]
        duration = last_ts - first_ts
        n_samples = len(ts)
        
        # Calculate effective sample rate
        if n_samples > 1:
            effective_srate = (n_samples - 1) / duration if duration > 0 else 0
        else:
            effective_srate = 0
        
        nominal_srate = float(stream["info"]["nominal_srate"][0]) if "nominal_srate" in stream["info"] else 0
        
        print(f"\n{name} ({stype}):")
        print(f"  Samples: {n_samples}")
        print(f"  First timestamp: {first_ts:.3f}s")
        print(f"  Last timestamp: {last_ts:.3f}s")
        print(f"  Duration: {duration:.3f}s ({duration/60:.2f} min)")
        print(f"  Nominal srate: {nominal_srate:.2f} Hz")
        print(f"  Effective srate: {effective_srate:.2f} Hz")
    else:
        print(f"\n{name} ({stype}): NO TIMESTAMPS")

# Find EEG and fNIRS streams
eeg_stream = None
fnirs_stream = None
marker_stream = None

for stream in streams:
    name = stream["info"]["name"][0]
    stype = stream["info"]["type"][0]
    
    if stype == "EEG" and "actiCHamp" in name:
        eeg_stream = stream
    elif stype == "NIRS" and "RAW" in name:
        fnirs_stream = stream
    elif stype == "Markers" and "cortivision_markers" == name:
        marker_stream = stream

print("\n" + "=" * 70)
print("DETAILED COMPARISON: EEG vs fNIRS vs Markers")
print("=" * 70)

if eeg_stream and fnirs_stream and marker_stream:
    eeg_ts = eeg_stream["time_stamps"]
    fnirs_ts = fnirs_stream["time_stamps"]
    marker_ts = marker_stream["time_stamps"]
    
    eeg_start = eeg_ts[0]
    eeg_end = eeg_ts[-1]
    fnirs_start = fnirs_ts[0]
    fnirs_end = fnirs_ts[-1]
    
    print(f"\nEEG:")
    print(f"  Start: {eeg_start:.3f}s")
    print(f"  End: {eeg_end:.3f}s")
    print(f"  Duration: {eeg_end - eeg_start:.3f}s")
    
    print(f"\nfNIRS:")
    print(f"  Start: {fnirs_start:.3f}s")
    print(f"  End: {fnirs_end:.3f}s")
    print(f"  Duration: {fnirs_end - fnirs_start:.3f}s")
    
    print(f"\nMarkers:")
    print(f"  First marker: {marker_ts[0]:.3f}s")
    print(f"  Last marker: {marker_ts[-1]:.3f}s")
    print(f"  Total markers: {len(marker_ts)}")
    
    print(f"\n--- OFFSET ANALYSIS ---")
    print(f"EEG starts {eeg_start - fnirs_start:.3f}s after fNIRS")
    print(f"EEG ends {eeg_end - fnirs_end:.3f}s before fNIRS ends")
    print(f"First marker at {marker_ts[0] - eeg_start:.3f}s relative to EEG start")
    print(f"First marker at {marker_ts[0] - fnirs_start:.3f}s relative to fNIRS start")
    
    # Check which markers fall within EEG recording
    markers_in_eeg = sum(1 for t in marker_ts if eeg_start <= t <= eeg_end)
    markers_in_fnirs = sum(1 for t in marker_ts if fnirs_start <= t <= fnirs_end)
    
    print(f"\n--- MARKER COVERAGE ---")
    print(f"Markers within EEG window: {markers_in_eeg}/{len(marker_ts)}")
    print(f"Markers within fNIRS window: {markers_in_fnirs}/{len(marker_ts)}")
    
    # Show marker timing relative to EEG
    print(f"\n--- MARKER TIMING (relative to EEG) ---")
    markers_data = marker_stream["time_series"]
    for i, (ts, marker) in enumerate(zip(marker_ts, markers_data)):
        rel_to_eeg = ts - eeg_start
        in_eeg = "✓" if eeg_start <= ts <= eeg_end else "✗ OUTSIDE EEG"
        print(f"  {i+1:2d}. {marker[0]:8s} at {ts:.3f}s (EEG+{rel_to_eeg:7.3f}s) {in_eeg}")

print("\n" + "=" * 70)
print("RAW XDF HEADER INFO")
print("=" * 70)
print(f"Header: {header}")

# Check for clock offsets in stream info
print("\n" + "=" * 70)
print("CLOCK OFFSET INFORMATION")
print("=" * 70)
for stream in streams:
    name = stream["info"]["name"][0]
    if "clock_offsets" in stream:
        offsets = stream["clock_offsets"]
        if len(offsets) > 0:
            print(f"\n{name}:")
            print(f"  Number of clock offset samples: {len(offsets)}")
            times = [o[0] for o in offsets]
            values = [o[1] for o in offsets]
            print(f"  First offset: {values[0]:.6f}s at time {times[0]:.3f}s")
            print(f"  Last offset: {values[-1]:.6f}s at time {times[-1]:.3f}s")
            print(f"  Offset range: {min(values):.6f}s to {max(values):.6f}s")
