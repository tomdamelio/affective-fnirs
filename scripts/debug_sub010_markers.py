"""Debug script to check markers for sub-010."""
import pyxdf
from pathlib import Path
from affective_fnirs.ingestion import identify_streams, extract_stream_data
from affective_fnirs.mne_builder import build_eeg_raw, embed_events

xdf_file = Path("data/raw/sub-010/sub-010_ses-001_task-fingertapping_recording.xdf")
streams, _ = pyxdf.load_xdf(str(xdf_file))
stream_info = identify_streams(streams)

eeg_stream = stream_info["eeg"]
eeg_data, eeg_srate, eeg_ts = extract_stream_data(eeg_stream)
marker_stream = stream_info["markers"]

# Check marker content
markers = [m[0] for m in marker_stream["time_series"]]
print(f"Markers found: {set(markers)}")
print(f"Total markers: {len(markers)}")
print(f"First 10 markers: {markers[:10]}")

# Build raw and embed events
raw_eeg = build_eeg_raw(eeg_data, eeg_srate, eeg_stream["info"], eeg_ts)
event_mapping = {"LEFT": 1, "RIGHT": 2, "NOTHING": 3}
raw_eeg = embed_events(raw_eeg, marker_stream, event_mapping)

print(f"\nAnnotations: {len(raw_eeg.annotations)}")
print(f"Annotation descriptions: {set(raw_eeg.annotations.description)}")
print(f"First 10 annotation descriptions: {list(raw_eeg.annotations.description[:10])}")
