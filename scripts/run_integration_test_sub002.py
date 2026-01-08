"""
Simple integration test script for sub-002 pipeline execution.

This script runs the complete pipeline on sub-002 data and reports results.
It's designed to be run manually to verify pipeline functionality.

Usage:
    micromamba run -n affective-fnirs python scripts/run_integration_test_sub002.py
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.config import PipelineConfig
from affective_fnirs.pipeline import run_validation_pipeline

def main():
    """Run integration test on sub-002."""
    print("=" * 70)
    print("INTEGRATION TEST: sub-002 Complete Pipeline")
    print("=" * 70)
    print()
    
    # Define paths
    data_root = Path("data/raw/sub-002")
    xdf_file = data_root / "sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
    eeg_json = data_root / "sub-002_Tomi_ses-001_task-fingertapping_eeg.json"
    fnirs_json = data_root / "sub-002_Tomi_ses-001_task-fingertapping_nirs.json"
    
    output_dir = Path("data/derivatives/validation-pipeline-integration-test/sub-002/ses-001")
    
    # Verify files exist
    print("Checking input files...")
    for file_path in [xdf_file, eeg_json, fnirs_json]:
        if not file_path.exists():
            print(f"❌ MISSING: {file_path}")
            return 1
        print(f"✓ Found: {file_path.name}")
    print()
    
    # Create configuration
    config = PipelineConfig.default()
    print(f"Using configuration with random seed: {config.random_seed}")
    print()
    
    # Run pipeline
    try:
        print("Running validation pipeline...")
        print("-" * 70)
        results = run_validation_pipeline(
            xdf_file=xdf_file,
            eeg_json=eeg_json,
            fnirs_json=fnirs_json,
            config=config,
            output_dir=output_dir,
        )
        print("-" * 70)
        print()
        
        # Print summary
        print("=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print()
        print(f"Subject: sub-{results.subject_id}")
        print(f"Session: ses-{results.session_id}")
        print(f"Task: {results.task}")
        print()
        
        print("Quality Assessment:")
        print(f"  Total channels: {results.quality_report.n_total_channels}")
        print(f"  Bad channels: {results.quality_report.n_bad_channels}")
        print(f"  Mean SCI: {results.quality_report.mean_sci:.3f}")
        print(f"  Mean CV: {results.quality_report.mean_cv:.2f}%")
        print()
        
        print("EEG Analysis:")
        print(f"  Channel: {results.erd_metrics.channel}")
        print(f"  Alpha ERD: {results.erd_metrics.alpha_erd_percent:.1f}% (p={results.erd_metrics.alpha_p_value:.4f})")
        print(f"  Alpha significant: {results.erd_metrics.alpha_significant}")
        print(f"  Beta ERD: {results.erd_metrics.beta_erd_percent:.1f}% (p={results.erd_metrics.beta_p_value:.4f})")
        print(f"  Beta significant: {results.erd_metrics.beta_significant}")
        print()
        
        print("fNIRS Analysis:")
        print(f"  Channel: {results.hrf_validation.channel}")
        print(f"  HRF onset: {results.hrf_validation.onset_time_sec:.2f}s (detected={results.hrf_validation.onset_detected})")
        print(f"  Time-to-peak: {results.hrf_validation.time_to_peak_sec:.2f}s (plausible={results.hrf_validation.peak_plausible})")
        print(f"  Plateau significant: {results.hrf_validation.plateau_significant}")
        print(f"  Trial consistency: r={results.hrf_validation.trial_consistency_r:.3f}")
        print()
        
        print("Neurovascular Coupling:")
        print(f"  Max correlation: {results.coupling_metrics.max_correlation:.3f}")
        print(f"  Lag: {results.coupling_metrics.lag_seconds:.2f}s")
        print(f"  EEG precedes fNIRS: {results.coupling_metrics.eeg_precedes_fnirs}")
        print(f"  Correlation adequate: {results.coupling_metrics.correlation_adequate}")
        print()
        
        print("Output Files:")
        print(f"  Directory: {output_dir}")
        output_files = list(output_dir.glob("sub-002_ses-001_task-fingertapping_*"))
        for file_path in sorted(output_files):
            print(f"  ✓ {file_path.name}")
        print()
        
        print("=" * 70)
        print("✓ INTEGRATION TEST PASSED")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ PIPELINE FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
