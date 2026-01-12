"""
Run complete validation pipeline for sub-005.

Short protocol: 12 trials, 1s task duration, 1s rest between trials.
"""

import sys
import numpy as np
from pathlib import Path

from affective_fnirs.config import PipelineConfig
from affective_fnirs.pipeline import run_validation_pipeline


def main() -> int:
    """Run the complete pipeline for sub-005."""
    # Paths (BIDS-compliant naming)
    xdf_file = Path("data/raw/sub-005/sub-005_ses-001_task-fingertapping_recording.xdf")
    eeg_json = Path("data/raw/sub-005/sub-005_ses-001_task-fingertapping_eeg.json")
    fnirs_json = Path("data/raw/sub-005/sub-005_ses-001_task-fingertapping_nirs.json")
    output_dir = Path("data/derivatives/validation-pipeline/sub-005/ses-001")

    # Check files exist
    if not xdf_file.exists():
        print(f"ERROR: XDF file not found: {xdf_file}")
        return 1
    if not eeg_json.exists():
        print(f"ERROR: EEG JSON not found: {eeg_json}")
        return 1
    if not fnirs_json.exists():
        print(f"ERROR: fNIRS JSON not found: {fnirs_json}")
        return 1

    # Load optimized configuration for sub-005
    config_path = Path("configs/sub005_optimized.yml")
    if config_path.exists():
        print(f"Loading configuration from: {config_path}")
        config = PipelineConfig.from_yaml(config_path)
    else:
        print("WARNING: sub005_optimized.yml not found, using default configuration")
        config = PipelineConfig.default()
    
    print("=" * 70)
    print("Running complete validation pipeline for sub-005")
    print("Protocol: 12 trials, 1s task, 1s rest")
    print(f"EEG epochs: {config.epochs.eeg_tmin_sec}s to {config.epochs.eeg_tmax_sec}s")
    print(f"fNIRS epochs: {config.epochs.fnirs_tmin_sec}s to {config.epochs.fnirs_tmax_sec}s")
    print("=" * 70)
    sys.stdout.flush()

    results = run_validation_pipeline(xdf_file, eeg_json, fnirs_json, config, output_dir)

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE - RESULTS SUMMARY")
    print("=" * 70)
    print(f"Subject: {results.subject_id}")
    print(f"Session: {results.session_id}")
    print(f"Task: {results.task}")
    print()
    print("--- Quality Report ---")
    print(f"Total channels: {results.quality_report.n_total_channels}")
    print(f"Bad channels: {results.quality_report.n_bad_channels}")
    print(f"Mean SCI: {results.quality_report.mean_sci:.3f}")
    print(f"Mean CV: {results.quality_report.mean_cv:.2f}%")
    print()
    print("--- EEG ERD/ERS ---")
    print(f"C3 Alpha ERD: {results.erd_metrics.alpha_erd_percent:.2f}%")
    print(f"C3 Beta ERD: {results.erd_metrics.beta_erd_percent:.2f}%")
    
    if results.erd_metrics_c4 is not None:
        print(f"C4 Alpha ERD: {results.erd_metrics_c4.alpha_erd_percent:.2f}%")
        print(f"C4 Beta ERD: {results.erd_metrics_c4.beta_erd_percent:.2f}%")
    
    # Lateralization Analysis Results
    if results.lateralization_metrics is not None:
        lat = results.lateralization_metrics
        print()
        print("--- Lateralization Analysis ---")
        print(f"Trials: LEFT={lat.n_left_trials}, RIGHT={lat.n_right_trials}, NOTHING={lat.n_nothing_trials}")
        print(f"LEFT shows contralateral ERD (C4): {'✓' if lat.left_shows_contralateral_erd else '✗'}")
        print(f"RIGHT shows contralateral ERD (C3): {'✓' if lat.right_shows_contralateral_erd else '✗'}")
        print(f"Overall pattern valid: {'✓ YES' if lat.lateralization_pattern_valid else '✗ NO'}")
    
    print()
    print("--- fNIRS HRF ---")
    print(f"Channel: {results.hrf_validation.channel}")
    print(f"Onset detected: {results.hrf_validation.onset_detected}")
    print(f"Time to peak: {results.hrf_validation.time_to_peak_sec:.2f}s")
    print()
    print("--- Output Files ---")
    print(f"Output directory: {output_dir}")
    
    if output_dir.exists():
        print("Generated files:")
        for f in sorted(output_dir.iterdir()):
            print(f"  - {f.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
