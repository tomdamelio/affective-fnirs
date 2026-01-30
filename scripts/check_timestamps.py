
import pyxdf
import numpy as np

def check_timestamps(path):
    print(f"Checking timestamps for: {path}")
    streams, header = pyxdf.load_xdf(path)
    
    eeg_stream = None
    marker_stream = None
    

    for s in streams:
        name = s['info']['name'][0]
        stype = s['info'].get('type', [''])[0]
        
        # More robust EEG check
        if 'eeg' in name.lower() or 'biosemi' in name.lower() or 'actichamp' in name.lower():
            eeg_stream = s
        
        # Robust marker check
        if 'markers' in name.lower() or 'event' in name.lower() or 'trigger' in name.lower():
            # specifically look for the one we want to test: eeg_markers
            if 'eeg_markers' in name:
                 marker_stream = s
            
    if not eeg_stream:
        print("EEG stream not found")
        return
    if not marker_stream:
        print("eeg_markers stream not found")
        return
        
    eeg_ts = eeg_stream['time_stamps']
    marker_ts = marker_stream['time_stamps']
    
    eeg_start = eeg_ts[0]
    eeg_end = eeg_ts[-1]
    duration = eeg_end - eeg_start
    
    print(f"EEG Range: {eeg_start:.4f} to {eeg_end:.4f} (Duration: {duration:.2f}s)")
    print(f"Marker Range: {marker_ts[0]:.4f} to {marker_ts[-1]:.4f}")
    
    # Check overlap
    valid_markers = [t for t in marker_ts if eeg_start <= t <= eeg_end]
    print(f"Total Markers: {len(marker_ts)}")
    print(f"Markers in EEG Range: {len(valid_markers)}")
    
    if len(valid_markers) == 0:
        print("CRITICAL: No markers found within EEG recording time range!")
        diff_start = marker_ts[0] - eeg_start
        print(f"Offset: First marker is {diff_start:.2f}s relative to EEG start")

if __name__ == "__main__":
    check_timestamps("data/raw/sub-011/ses-001/sub-011_ses-001_task-fingertapping_recording.xdf")
