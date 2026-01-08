"""
Run complete validation pipeline for sub-002.

This script executes the full multimodal EEG + fNIRS validation pipeline
and prints a summary of results.
"""

import numpy as np
from pathlib import Path

from affective_fnirs.config import PipelineConfig
from affective_fnirs.pipeline import run_validation_pipeline


def main():
    """Run the complete pipeline for sub-002."""
    # Paths
    xdf_file = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
    eeg_json = Path("data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_eeg.json")
    fnirs_json = Path("data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_nirs.json")
    output_dir = Path("data/derivatives/validation-pipeline/sub-002/ses-001")

    # Load optimized configuration for sub-002
    config_path = Path("configs/sub002_optimized.yml")
    if config_path.exists():
        print(f"Loading optimized configuration from: {config_path}")
        config = PipelineConfig.from_yaml(config_path)
    else:
        print("Using default configuration")
        config = PipelineConfig.default()
    
    print("=" * 70)
    print("Running complete validation pipeline for sub-002")
    print(f"ICA enabled: {config.ica.enabled}")
    print("=" * 70)

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
    print("--- EEG ERD/ERS (All Conditions Combined) ---")
    print(f"C3 (Left Motor Cortex):")
    print(f"  Alpha ERD: {results.erd_metrics.alpha_erd_percent:.2f}%")
    print(f"  Beta ERD: {results.erd_metrics.beta_erd_percent:.2f}%")
    if not np.isnan(results.erd_metrics.beta_rebound_percent):
        print(f"  Beta Rebound (15-20s): {results.erd_metrics.beta_rebound_percent:.2f}%")
    else:
        print(f"  Beta Rebound: N/A (check epoch range)")
    
    if results.erd_metrics_c4 is not None:
        print(f"C4 (Right Motor Cortex):")
        print(f"  Alpha ERD: {results.erd_metrics_c4.alpha_erd_percent:.2f}%")
        print(f"  Beta ERD: {results.erd_metrics_c4.beta_erd_percent:.2f}%")
        if not np.isnan(results.erd_metrics_c4.beta_rebound_percent):
            print(f"  Beta Rebound (15-20s): {results.erd_metrics_c4.beta_rebound_percent:.2f}%")
        else:
            print(f"  Beta Rebound: N/A (check epoch range)")
    
    print()
    print("NOTE: Values above are near-zero because they average LEFT + RIGHT + NOTHING.")
    print("      See Lateralization Analysis below for condition-specific ERD patterns.")
    print()
    
    # Lateralization Analysis Results
    if results.lateralization_metrics is not None:
        lat = results.lateralization_metrics
        print("--- Lateralization Analysis ---")
        print(f"Trials: LEFT={lat.n_left_trials}, RIGHT={lat.n_right_trials}, NOTHING={lat.n_nothing_trials}")
        print()
        print("Alpha ERD (%):")
        print(f"  LEFT:    C3={lat.left_c3_alpha_erd:>7.1f}  C4={lat.left_c4_alpha_erd:>7.1f}  (Lat: {lat.left_lateralization_alpha:>6.1f})")
        print(f"  RIGHT:   C3={lat.right_c3_alpha_erd:>7.1f}  C4={lat.right_c4_alpha_erd:>7.1f}  (Lat: {lat.right_lateralization_alpha:>6.1f})")
        print(f"  NOTHING: C3={lat.nothing_c3_alpha_erd:>7.1f}  C4={lat.nothing_c4_alpha_erd:>7.1f}")
        print()
        print("Beta ERD (%):")
        print(f"  LEFT:    C3={lat.left_c3_beta_erd:>7.1f}  C4={lat.left_c4_beta_erd:>7.1f}  (Lat: {lat.left_lateralization_beta:>6.1f})")
        print(f"  RIGHT:   C3={lat.right_c3_beta_erd:>7.1f}  C4={lat.right_c4_beta_erd:>7.1f}  (Lat: {lat.right_lateralization_beta:>6.1f})")
        print(f"  NOTHING: C3={lat.nothing_c3_beta_erd:>7.1f}  C4={lat.nothing_c4_beta_erd:>7.1f}")
        print()
        print("Statistical Tests (p-values):")
        print(f"  LEFT C4 vs NOTHING C4:  p={lat.left_vs_nothing_c4_p:.4f}")
        print(f"  RIGHT C3 vs NOTHING C3: p={lat.right_vs_nothing_c3_p:.4f}")
        print(f"  LEFT C4 vs C3 (paired): p={lat.left_contralateral_vs_ipsilateral_p:.4f}")
        print(f"  RIGHT C3 vs C4 (paired): p={lat.right_contralateral_vs_ipsilateral_p:.4f}")
        print()
        print("Validation:")
        print(f"  LEFT shows contralateral ERD (C4):  {'✓' if lat.left_shows_contralateral_erd else '✗'}")
        print(f"  RIGHT shows contralateral ERD (C3): {'✓' if lat.right_shows_contralateral_erd else '✗'}")
        print(f"  Overall pattern valid: {'✓ YES' if lat.lateralization_pattern_valid else '✗ NO'}")
        print()
    else:
        print("--- Lateralization Analysis ---")
        print("  (Not available)")
        print()
    
    print("--- fNIRS HRF ---")
    print(f"Channel: {results.hrf_validation.channel}")
    print(f"Onset detected: {results.hrf_validation.onset_detected}")
    print(f"Time to peak: {results.hrf_validation.time_to_peak_sec:.2f}s")
    print(f"Plateau significant: {results.hrf_validation.plateau_significant}")
    print()
    print("--- Neurovascular Coupling ---")
    print(f"Max correlation: {results.coupling_metrics.max_correlation}")
    print(f"Lag: {results.coupling_metrics.lag_seconds:.2f}s")
    print(f"EEG precedes fNIRS: {results.coupling_metrics.eeg_precedes_fnirs}")
    print()
    print("--- Output Files ---")
    print(f"Output directory: {output_dir}")
    
    # List generated files
    if output_dir.exists():
        print("Generated files:")
        for f in sorted(output_dir.iterdir()):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
