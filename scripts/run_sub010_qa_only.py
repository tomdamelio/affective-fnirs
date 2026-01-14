"""
Run QA-only report for sub-010.

Only generates Experiment QA metrics without ERD/HRF analysis.
NOTE: fNIRS data not available for this subject - EEG QA only.
"""

import sys
from pathlib import Path
import pyxdf
import json

from affective_fnirs.ingestion import identify_streams, extract_stream_data


def main() -> int:
    """Run QA-only pipeline for sub-010 (EEG only)."""
    xdf_file = Path("data/raw/sub-010/sub-010_ses-001_task-fingertapping_recording.xdf")
    eeg_json = Path("data/raw/sub-010/sub-010_ses-001_task-fingertapping_eeg.json")
    output_dir = Path("data/derivatives/validation-pipeline/sub-010/ses-001")
    output_dir.mkdir(parents=True, exist_ok=True)

    for f in [xdf_file, eeg_json]:
        if not f.exists():
            print(f"ERROR: File not found: {f}")
            return 1

    print("=" * 70)
    print("sub-010 QA-ONLY REPORT (EEG ONLY)")
    print("NOTE: fNIRS data not available")
    print("=" * 70)

    # Load XDF
    print("\n[1/2] Loading XDF file...")
    streams, _ = pyxdf.load_xdf(str(xdf_file))
    stream_info = identify_streams(streams)
    
    # Extract streams
    eeg_stream = stream_info["eeg"]
    eeg_data, eeg_srate, eeg_ts = extract_stream_data(eeg_stream)
    
    marker_stream = stream_info["markers"]
    markers = [m[0] for m in marker_stream["time_series"]]
    marker_ts = marker_stream["time_stamps"]

    # Timing analysis
    eeg_duration = eeg_ts[-1] - eeg_ts[0]
    
    # Get channel counts from data shape (data is channels x samples)
    eeg_n_channels = eeg_data.shape[0]
    eeg_n_samples = eeg_data.shape[1]
    
    print(f"\n" + "=" * 70)
    print("TIMING ANALYSIS")
    print("=" * 70)
    print(f"EEG: {eeg_duration:.1f}s ({eeg_duration/60:.1f} min)")
    print(f"  Channels: {eeg_n_channels}")
    print(f"  Sample rate: {eeg_srate:.0f} Hz")
    print(f"  Samples: {eeg_n_samples}")
    
    # Marker coverage
    eeg_start, eeg_end = eeg_ts[0], eeg_ts[-1]
    markers_in_eeg = sum(1 for t in marker_ts if eeg_start <= t <= eeg_end)
    
    print(f"\n" + "=" * 70)
    print("MARKER ANALYSIS")
    print("=" * 70)
    print(f"Total markers: {len(markers)}")
    print(f"Markers in EEG window: {markers_in_eeg}/{len(markers)} ({100*markers_in_eeg/len(markers):.0f}%)")

    # Count by condition
    left_count = sum(1 for m in markers if m == "LEFT")
    right_count = sum(1 for m in markers if m == "RIGHT")
    nothing_count = sum(1 for m in markers if m == "NOTHING")
    print(f"\nTrials by condition:")
    print(f"  LEFT: {left_count}")
    print(f"  RIGHT: {right_count}")
    print(f"  NOTHING: {nothing_count}")

    # Show marker timing
    print(f"\n[2/2] Marker timing details...")
    print(f"\nFirst 10 markers (relative to EEG start):")
    for i, (m, t) in enumerate(zip(markers[:10], marker_ts[:10])):
        rel_time = t - eeg_start
        in_eeg = "✓" if eeg_start <= t <= eeg_end else "✗"
        print(f"  {i+1:2d}. {m:8s} at {rel_time:6.2f}s {in_eeg}")
    
    if len(markers) > 10:
        print(f"  ... ({len(markers) - 10} more markers)")

    # QA Summary
    print(f"\n" + "=" * 70)
    print("QA SUMMARY")
    print("=" * 70)
    
    expected_duration = 36 * 30  # 36 trials * 30s per trial = 1080s = 18 min
    print(f"Expected duration: {expected_duration:.0f}s ({expected_duration/60:.1f} min)")
    print(f"Actual EEG duration: {eeg_duration:.0f}s ({eeg_duration/60:.1f} min)")
    print(f"Coverage: {eeg_duration/expected_duration*100:.1f}%")
    
    # Issues
    issues = []
    if markers_in_eeg < len(markers):
        issues.append(f"EEG missing {len(markers) - markers_in_eeg} trials ({100*(len(markers)-markers_in_eeg)/len(markers):.0f}%)")
    if eeg_duration < expected_duration * 0.8:
        issues.append(f"EEG recording shorter than expected ({eeg_duration/expected_duration*100:.0f}%)")
    if eeg_duration < 30:
        issues.append(f"Very short EEG recording ({eeg_duration:.1f}s)")
    
    if issues:
        print("\n⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No major timing issues detected")

    # Save summary to JSON
    summary = {
        "subject": "sub-010",
        "session": "ses-001",
        "task": "fingertapping",
        "protocol": "standard_15s_extended",
        "note": "fNIRS data not available - EEG only",
        "eeg": {
            "duration_sec": round(eeg_duration, 2),
            "n_channels": int(eeg_n_channels),
            "sample_rate_hz": round(eeg_srate, 2),
            "n_samples": int(eeg_n_samples)
        },
        "markers": {
            "total": len(markers),
            "in_eeg_window": markers_in_eeg,
            "left": left_count,
            "right": right_count,
            "nothing": nothing_count
        },
        "issues": issues
    }
    
    summary_path = output_dir / "sub-010_ses-001_task-fingertapping_desc-qa_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved: {summary_path}")
    print(f"Output: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
