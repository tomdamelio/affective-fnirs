#!/usr/bin/env python
"""
EEG-only analysis for sub-001 to compare with sub-002 results.

This script runs the same EEG analysis pipeline as sub-002 but without fNIRS,
to verify if the RIGHT condition issue is reproducible across subjects.
"""

import sys
from pathlib import Path
import numpy as np
import mne

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.ingestion import load_xdf_file, identify_streams, extract_stream_data
from affective_fnirs.mne_builder import build_eeg_raw, embed_events
from affective_fnirs.eeg_processing import preprocess_eeg_pipeline
from affective_fnirs.lateralization_analysis import compute_lateralization_analysis
from affective_fnirs.config import PipelineConfig


def main():
    print("=" * 80)
    print("EEG ANALYSIS FOR SUB-001")
    print("=" * 80)
    
    # Load data
    xdf_path = Path("data/raw/sub-001/sub-001_tomi_ses-001_task-fingertapping_recording.xdf")
    
    if not xdf_path.exists():
        print(f"\nERROR: File not found: {xdf_path}")
        print("Please verify the file path.")
        return
    
    print(f"\nLoading XDF file: {xdf_path}")
    streams, header = load_xdf_file(xdf_path)
    identified_streams = identify_streams(streams)
    
    # Extract EEG data
    eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(identified_streams["eeg"])
    marker_stream = identified_streams["markers"]
    
    print(f"\nEEG Stream:")
    print(f"  Duration: {eeg_timestamps[-1] - eeg_timestamps[0]:.1f}s")
    print(f"  Samples: {len(eeg_timestamps)}")
    print(f"  Sample rate: {eeg_sfreq} Hz")
    
    # Count markers
    markers = marker_stream["time_series"]
    marker_list = [str(m[0]) if isinstance(m, (list, np.ndarray)) else str(m) for m in markers]
    
    left_count = sum(1 for m in marker_list if "LEFT" in m.upper())
    right_count = sum(1 for m in marker_list if "RIGHT" in m.upper())
    nothing_count = sum(1 for m in marker_list if "NOTHING" in m.upper())
    
    print(f"\nMarkers:")
    print(f"  LEFT: {left_count}")
    print(f"  RIGHT: {right_count}")
    print(f"  NOTHING: {nothing_count}")
    print(f"  Total: {len(markers)}")
    
    # Build EEG Raw
    print("\n" + "=" * 80)
    print("BUILDING EEG RAW OBJECT")
    print("=" * 80)
    
    raw_eeg = build_eeg_raw(
        eeg_data, eeg_sfreq, identified_streams["eeg"]["info"], eeg_timestamps
    )
    
    # Embed events
    event_mapping = {
        "LEFT": 1,
        "RIGHT": 2,
        "NOTHING": 3,
    }
    raw_eeg = embed_events(raw_eeg, marker_stream, event_mapping)
    
    print(f"\nAnnotations embedded:")
    ann_counts = {}
    for ann in raw_eeg.annotations:
        desc = ann['description']
        ann_counts[desc] = ann_counts.get(desc, 0) + 1
    
    for desc, count in sorted(ann_counts.items()):
        print(f"  '{desc}': {count}")
    
    # Process EEG
    print("\n" + "=" * 80)
    print("PROCESSING EEG")
    print("=" * 80)
    
    config = PipelineConfig()
    
    # Disable ICA for faster processing
    config.preprocessing.ica_enabled = False
    
    raw_eeg_processed = preprocess_eeg_pipeline(raw_eeg, config)
    
    print(f"\nEEG processed successfully")
    
    # Lateralization analysis
    print("\n" + "=" * 80)
    print("LATERALIZATION ANALYSIS")
    print("=" * 80)
    
    lat_result = compute_lateralization_analysis(
        raw_eeg_processed,
        alpha_band=(8, 12),
        beta_band=(13, 30),
        tmin=config.epochs.eeg_tmin_sec,
        tmax=config.epochs.eeg_tmax_sec,
        baseline=(config.epochs.baseline_tmin_sec, config.epochs.baseline_tmax_sec),
        task_window=(0, 10),
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\nTrials: LEFT={lat_result.n_left_trials}, RIGHT={lat_result.n_right_trials}, NOTHING={lat_result.n_nothing_trials}")
    
    print(f"\nAlpha ERD (%):")
    print(f"  LEFT:    C3={lat_result.left_c3_alpha_erd:+.1f}, C4={lat_result.left_c4_alpha_erd:+.1f} (Lat: {lat_result.left_lateralization_alpha:+.1f})")
    print(f"  RIGHT:   C3={lat_result.right_c3_alpha_erd:+.1f}, C4={lat_result.right_c4_alpha_erd:+.1f} (Lat: {lat_result.right_lateralization_alpha:+.1f})")
    print(f"  NOTHING: C3={lat_result.nothing_c3_alpha_erd:+.1f}, C4={lat_result.nothing_c4_alpha_erd:+.1f}")
    
    print(f"\nBeta ERD (%):")
    print(f"  LEFT:    C3={lat_result.left_c3_beta_erd:+.1f}, C4={lat_result.left_c4_beta_erd:+.1f} (Lat: {lat_result.left_lateralization_beta:+.1f})")
    print(f"  RIGHT:   C3={lat_result.right_c3_beta_erd:+.1f}, C4={lat_result.right_c4_beta_erd:+.1f} (Lat: {lat_result.right_lateralization_beta:+.1f})")
    print(f"  NOTHING: C3={lat_result.nothing_c3_beta_erd:+.1f}, C4={lat_result.nothing_c4_beta_erd:+.1f}")
    
    print(f"\nStatistical Tests (p-values):")
    print(f"  LEFT C4 vs NOTHING C4:  p={lat_result.left_vs_nothing_c4_p:.4f}")
    print(f"  RIGHT C3 vs NOTHING C3: p={lat_result.right_vs_nothing_c3_p:.4f}")
    print(f"  LEFT C4 vs C3 (paired): p={lat_result.left_contralateral_vs_ipsilateral_p:.4f}")
    print(f"  RIGHT C3 vs C4 (paired): p={lat_result.right_contralateral_vs_ipsilateral_p:.4f}")
    
    print(f"\nValidation:")
    print(f"  LEFT shows contralateral ERD (C4):  {'✓' if lat_result.left_shows_contralateral_erd else '✗'}")
    print(f"  RIGHT shows contralateral ERD (C3): {'✓' if lat_result.right_shows_contralateral_erd else '✗'}")
    print(f"  Overall pattern valid: {'✓ YES' if lat_result.lateralization_pattern_valid else '✗ NO'}")
    
    # Comparison with sub-002
    print("\n" + "=" * 80)
    print("COMPARISON WITH SUB-002")
    print("=" * 80)
    
    print(f"""
SUB-002 Results (for reference):
  Trials: LEFT=7, RIGHT=7, NOTHING=7
  
  Alpha ERD (%):
    LEFT:    C3=-28.0, C4=-38.4 (Lat: -10.4)
    RIGHT:   C3=+16.2, C4=+3.9  (Lat: +12.3)
    NOTHING: C3=+11.7, C4=+9.7
  
  Beta ERD (%):
    LEFT:    C3=-30.4, C4=-39.7 (Lat: -9.3)
    RIGHT:   C3=-31.2, C4=-31.4 (Lat: +0.3)
    NOTHING: C3=+15.6, C4=+28.4
  
  Validation:
    LEFT shows contralateral ERD (C4):  ✓
    RIGHT shows contralateral ERD (C3): ✗
    Overall pattern valid: ✗ NO

SUB-001 Results (current):
  Trials: LEFT={lat_result.n_left_trials}, RIGHT={lat_result.n_right_trials}, NOTHING={lat_result.n_nothing_trials}
  
  Alpha ERD (%):
    LEFT:    C3={lat_result.left_c3_alpha_erd:+.1f}, C4={lat_result.left_c4_alpha_erd:+.1f} (Lat: {lat_result.left_lateralization_alpha:+.1f})
    RIGHT:   C3={lat_result.right_c3_alpha_erd:+.1f}, C4={lat_result.right_c4_alpha_erd:+.1f} (Lat: {lat_result.right_lateralization_alpha:+.1f})
    NOTHING: C3={lat_result.nothing_c3_alpha_erd:+.1f}, C4={lat_result.nothing_c4_alpha_erd:+.1f}
  
  Beta ERD (%):
    LEFT:    C3={lat_result.left_c3_beta_erd:+.1f}, C4={lat_result.left_c4_beta_erd:+.1f} (Lat: {lat_result.left_lateralization_beta:+.1f})
    RIGHT:   C3={lat_result.right_c3_beta_erd:+.1f}, C4={lat_result.right_c4_beta_erd:+.1f} (Lat: {lat_result.right_lateralization_beta:+.1f})
    NOTHING: C3={lat_result.nothing_c3_beta_erd:+.1f}, C4={lat_result.nothing_c4_beta_erd:+.1f}
  
  Validation:
    LEFT shows contralateral ERD (C4):  {'✓' if lat_result.left_shows_contralateral_erd else '✗'}
    RIGHT shows contralateral ERD (C3): {'✓' if lat_result.right_shows_contralateral_erd else '✗'}
    Overall pattern valid: {'✓ YES' if lat_result.lateralization_pattern_valid else '✗ NO'}
""")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
