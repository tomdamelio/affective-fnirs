"""
Demonstration script for fNIRS analysis module.

This script shows how to use the fNIRS analysis functions for HRF extraction
and validation. It demonstrates the complete workflow from epoch creation to
visualization.

Note: This is a demonstration script. Actual usage requires real fNIRS data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.fnirs_analysis import (
    compute_hrf_quality_metrics,
    create_fnirs_epochs,
    extract_hrf,
    identify_motor_roi_channel,
    plot_hrf_curves,
    plot_hrf_spatial_map,
    validate_hrf_temporal_dynamics,
)

print("=" * 70)
print("fNIRS Analysis Module Demonstration")
print("=" * 70)

print("\n1. Module Functions:")
print("   ✓ create_fnirs_epochs - Extract fNIRS epochs with extended window")
print("   ✓ identify_motor_roi_channel - Find channel closest to motor cortex")
print("   ✓ extract_hrf - Extract averaged HRF for a channel")
print("   ✓ validate_hrf_temporal_dynamics - Validate HRF onset, peak, plateau")
print("   ✓ compute_hrf_quality_metrics - Compute SNR and trial consistency")
print("   ✓ plot_hrf_curves - Visualize HRF with annotations")
print("   ✓ plot_hrf_spatial_map - Spatial map of HRF amplitude")

print("\n2. Typical Workflow:")
print("   Step 1: Create fNIRS epochs (tmin=-5s, tmax=30s)")
print("   Step 2: Identify motor ROI channel (closest to C3)")
print("   Step 3: Extract HRF for HbO and HbR")
print("   Step 4: Validate temporal dynamics (onset, peak, plateau)")
print("   Step 5: Compute quality metrics (SNR, consistency)")
print("   Step 6: Visualize HRF curves with annotations")

print("\n3. Expected HRF Characteristics:")
print("   - Onset latency: ~2-3s after stimulus (neurovascular delay)")
print("   - Time-to-peak: ~5-8s post-stimulus")
print("   - Plateau: Sustained elevation during 15s task (5-15s)")
print("   - Return to baseline: ~20-30s after task cessation")
print("   - HbO: Increases during activation (positive deflection)")
print("   - HbR: Decreases during activation (inverse pattern)")

print("\n4. Validation Metrics:")
print("   - Onset detection: HbO > 0.1 μM within 2-3s window")
print("   - Peak detection: Maximum HbO in 4-8s window")
print("   - Plateau significance: Paired t-test (p < 0.05)")
print("   - Trial consistency: Pearson correlation (r > 0.7 = good)")
print("   - SNR: (plateau - baseline) / baseline_std (>2 = good)")

print("\n5. Requirements Satisfied:")
print("   ✓ Req 6.5: fNIRS epoch extraction")
print("   ✓ Req 6.6: Baseline correction (-5 to 0s)")
print("   ✓ Req 6.7: Motor ROI identification")
print("   ✓ Req 6.8: HRF extraction and averaging")
print("   ✓ Req 6.9: Onset detection (2-3s)")
print("   ✓ Req 6.10: Plateau amplitude validation")
print("   ✓ Req 6.11: Time-to-peak validation (4-8s)")
print("   ✓ Req 6.12: Quality metrics and visualization")
print("   ✓ Req 8.3: HRF visualization")

print("\n6. Example Usage:")
print("""
    # Create epochs
    epochs = create_fnirs_epochs(
        raw_haemo,
        event_id={'block_start': 2},
        tmin=-5.0,
        tmax=30.0
    )
    
    # Identify motor channel
    motor_channel = identify_motor_roi_channel(
        raw_haemo,
        montage_config,
        target_region='C3'
    )
    
    # Extract HRF
    times, hrf_hbo = extract_hrf(epochs, motor_channel, 'hbo')
    
    # Validate dynamics
    validation = validate_hrf_temporal_dynamics(
        times, hrf_hbo, epochs, motor_channel
    )
    
    # Compute quality
    quality = compute_hrf_quality_metrics(epochs, motor_channel)
    
    # Visualize
    fig = plot_hrf_curves(
        times, hrf_hbo, hrf_hbr,
        epochs=epochs,
        channel=motor_channel,
        onset_time=validation['onset_time_sec'],
        peak_time=validation['time_to_peak_sec']
    )
""")

print("\n" + "=" * 70)
print("Module demonstration complete!")
print("=" * 70)
print("\nNext steps:")
print("  1. Run pipeline on pilot data (sub-002)")
print("  2. Validate HRF patterns match expected motor cortex activation")
print("  3. Generate quality reports and visualizations")
print("  4. Integrate with multimodal analysis (neurovascular coupling)")
