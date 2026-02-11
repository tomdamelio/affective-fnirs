
import sys
import json
from pathlib import Path
import logging

# Add src to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent / "src"))

print("Starting Debug Script...", flush=True)

from affective_fnirs.config import SubjectConfig
from affective_fnirs.ingestion import load_xdf_file, identify_streams, extract_stream_data, DataIngestionError
from affective_fnirs.mne_builder import build_eeg_raw, build_fnirs_raw, embed_events, MNEConstructionError
# from affective_fnirs.reporting import run_quality_assessment

# Configure logging
logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def build_mne_objects_local(streams, config):
    raw_eeg = None
    raw_fnirs = None
    
    # Event mapping
    event_mapping = {
        "LEFT": 1,
        "RIGHT": 2,
        "NOTHING": 3,
        "task_start": 10,
        "task_end": 11,
    }

    # EEG Logic (Simplified)
    if config.modalities.eeg_enabled and streams["eeg"]:
        try:
             eeg_data, sfreq, ts = extract_stream_data(streams["eeg"])
             raw_eeg = build_eeg_raw(eeg_data, sfreq, streams["eeg"]["info"], ts)
             if streams["markers"]:
                 raw_eeg = embed_events(raw_eeg, streams["markers"], event_mapping)
        except Exception as e:
             print(f"EEG Build Failed: {e}", flush=True)

    # fNIRS Logic
    if config.modalities.fnirs_enabled:
        if streams["fnirs"] is None:
            print("WARNING: fNIRS enabled but stream not found!", flush=True)
        else:
            try:
                print("Building fNIRS Raw object...", flush=True)
                data, sfreq, timestamps = extract_stream_data(streams["fnirs"])
                print(f"Extracted fNIRS data: {data.shape} @ {sfreq}Hz", flush=True)
                
                # Load JSON Montage
                json_filename = f"sub-{config.subject.id}_ses-{config.subject.session}_task-{config.subject.task}_nirs.json"
                json_path = config.data_root / f"sub-{config.subject.id}" / json_filename
                
                # Check session subdir
                if not json_path.exists():
                     json_path = config.data_root / f"sub-{config.subject.id}" / f"ses-{config.subject.session}" / json_filename
                
                if not json_path.exists():
                     json_filename_lower = f"sub-{config.subject.id}_Tomi_ses-{config.subject.session}_task-{config.subject.task}_nirs.json"
                     path_lower = config.data_root / f"sub-{config.subject.id}" / json_filename_lower
                     if path_lower.exists():
                         json_path = path_lower
                
                if not json_path.exists():
                    print(f"ERROR: JSON Sidecar not found! Tried: {json_path}", flush=True)
                    return raw_eeg, None

                with open(json_path, "r") as f:
                    sidecar = json.load(f)
                montage = sidecar.get("ChMontage", [])
                print(f"Loaded montage: {len(montage)} channels", flush=True)
                
                raw_fnirs = build_fnirs_raw(data, sfreq, montage, timestamps)
                print(f"Built raw_fnirs: {len(raw_fnirs.ch_names)} channels", flush=True)
                
                if streams["markers"]:
                    raw_fnirs = embed_events(raw_fnirs, streams["markers"], event_mapping)
                    print(f"Embedded events: {len(raw_fnirs.annotations)}")
                    
            except Exception as e:
                print(f"ERROR: fNIRS Build Failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raw_fnirs = None
                
    return raw_eeg, raw_fnirs

def test_fnirs_loading():
    print("Testing fNIRS Loading for Sub-012...", flush=True)
    
    # 1. Load Config
    config_path = Path("configs/sub-012.yml")
    config = SubjectConfig.from_yaml(config_path)
    print(f"Config loaded. fNIRS Enabled: {config.modalities.fnirs_enabled}", flush=True)
    
    # 2. Load Streams
    xdf_filename = f"sub-12_ses-{config.subject.session}_task-{config.subject.task}_recording.xdf"
    xdf_path = config.data_root / f"sub-{config.subject.id}" / f"ses-{config.subject.session}" / xdf_filename
    
    # Handle path variations
    if not xdf_path.exists():
         xdf_path = config.data_root / f"sub-{config.subject.id}" / xdf_filename
    
    print(f"Loading XDF: {xdf_path}", flush=True)
    if not xdf_path.exists():
        print(f"CRITICAL: XDF FILE NOT FOUND AT {xdf_path}", flush=True)
        return

    streams, _ = load_xdf_file(xdf_path)
    streams = identify_streams(streams)
    
    if streams['fnirs']:
        print(f"Found fNIRS stream: {streams['fnirs']['info']['name'][0]}", flush=True)
    else:
        print("ERROR: fNIRS stream NOT found in identify_streams!", flush=True)

    # 3. Build MNE Objects
    print("Building MNE objects...", flush=True)
    raw_eeg, raw_fnirs = build_mne_objects_local(streams, config)
    
    if raw_fnirs:
        print(f"fNIRS Raw successfully built!", flush=True)
        print(f"  Channels: {len(raw_fnirs.ch_names)}", flush=True)
        print(f"  Duration: {raw_fnirs.times[-1]:.2f}s", flush=True)
        
        # Check QA (Optional if it imports heavy stuff)
        # print("Running QA on fNIRS...", flush=True)
        # qa_results = run_quality_assessment(raw_eeg, raw_fnirs, config)
        
        # fnirs_dur = qa_results.recording_duration.get('fnirs', 'MISSING')
        # fnirs_trials = qa_results.valid_trial_count.get('fnirs', 'MISSING')
        # print(f"QA Duration (fNIRS): {fnirs_dur}", flush=True)
        # print(f"QA Valid Trials (fNIRS): {fnirs_trials}", flush=True)
    else:
        print("ERROR: raw_fnirs is None after construction!", flush=True)

if __name__ == "__main__":
    test_fnirs_loading()
