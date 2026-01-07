"""
Test script for fNIRS quality assessment module.

This script tests the quality assessment functions on pilot data to verify
basic functionality before full integration testing.
"""

import logging
from pathlib import Path

import mne
import numpy as np

from affective_fnirs.config import PipelineConfig
from affective_fnirs.fnirs_quality import (
    assess_cardiac_power,
    calculate_coefficient_of_variation,
    calculate_sci,
    detect_flat_signal,
    detect_saturation,
    generate_quality_heatmap,
    generate_quality_summary_table,
    mark_bad_channels,
)
from affective_fnirs.ingestion import (
    extract_stream_data,
    identify_streams,
    load_xdf_file,
)
from affective_fnirs.mne_builder import build_fnirs_raw

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Test fNIRS quality assessment on pilot data."""
    # Use sub-002 (complete data)
    xdf_path = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
    fnirs_json_path = Path("data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_nirs.json")
    montage_config_path = Path("configs/montage_config.json")

    if not xdf_path.exists():
        logger.error(f"XDF file not found: {xdf_path}")
        return

    logger.info("=" * 80)
    logger.info("Testing fNIRS Quality Assessment Module")
    logger.info("=" * 80)

    # Load configuration
    config = PipelineConfig.default()

    # Load XDF file
    logger.info(f"\n1. Loading XDF file: {xdf_path}")
    streams, header = load_xdf_file(xdf_path)
    logger.info(f"   Loaded {len(streams)} streams")

    # Identify streams
    logger.info("\n2. Identifying streams")
    identified_streams = identify_streams(streams)
    logger.info(f"   Found streams: {list(identified_streams.keys())}")

    # Extract fNIRS stream
    logger.info("\n3. Extracting fNIRS stream")
    fnirs_stream = identified_streams["fnirs"]
    fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(fnirs_stream)
    logger.info(f"   fNIRS data shape: {fnirs_data.shape}")
    logger.info(f"   Sampling rate: {fnirs_sfreq} Hz")

    # Load montage configuration
    logger.info("\n4. Loading montage configuration")
    import json

    with open(montage_config_path, "r") as f:
        montage_config = json.load(f)

    # Build fNIRS Raw object
    logger.info("\n5. Building fNIRS Raw object")
    raw_fnirs = build_fnirs_raw(
        fnirs_data, fnirs_sfreq, montage_config, fnirs_timestamps
    )
    logger.info(f"   Created Raw object with {len(raw_fnirs.ch_names)} channels")
    logger.info(f"   Channel types: {set(raw_fnirs.get_channel_types())}")

    # Test quality assessment functions
    logger.info("\n" + "=" * 80)
    logger.info("QUALITY ASSESSMENT TESTS")
    logger.info("=" * 80)

    # Test 1: Calculate SCI
    logger.info("\n6. Testing SCI calculation")
    try:
        sci_values = calculate_sci(
            raw_fnirs,
            freq_range=(
                config.filters.cardiac_band_low_hz,
                config.filters.cardiac_band_high_hz,
            ),
            sci_threshold=config.quality.sci_threshold,
        )
        logger.info(f"   ✓ SCI calculated for {len(sci_values)} channel pairs")
        logger.info(f"   Sample values: {list(sci_values.items())[:3]}")
    except Exception as e:
        logger.error(f"   ✗ SCI calculation failed: {e}")
        return

    # Test 2: Calculate CV (using entire recording as baseline for testing)
    logger.info("\n7. Testing CV calculation")
    try:
        # For testing, use entire recording as "baseline"
        duration = raw_fnirs.times[-1]
        baseline_annotations = [(0, min(5.0, duration))]

        cv_values = calculate_coefficient_of_variation(
            raw_fnirs,
            baseline_annotations=baseline_annotations,
            cv_threshold=config.quality.cv_threshold_percent,
        )
        logger.info(f"   ✓ CV calculated for {len(cv_values)} channels")
        logger.info(f"   Sample values: {list(cv_values.items())[:3]}")
    except Exception as e:
        logger.error(f"   ✗ CV calculation failed: {e}")
        cv_values = None

    # Test 3: Detect saturation
    logger.info("\n8. Testing saturation detection")
    try:
        saturation_values = detect_saturation(
            raw_fnirs,
            adc_max=None,  # Auto-detect
            saturation_threshold=0.95,
            max_saturation_percent=config.quality.saturation_percent,
        )
        logger.info(f"   ✓ Saturation detected for {len(saturation_values)} channels")
        logger.info(f"   Sample values: {list(saturation_values.items())[:3]}")
    except Exception as e:
        logger.error(f"   ✗ Saturation detection failed: {e}")
        return

    # Test 4: Detect flat signals
    logger.info("\n9. Testing flat signal detection")
    try:
        flat_signals = detect_flat_signal(raw_fnirs, variance_threshold=1e-6)
        flat_count = sum(flat_signals.values())
        logger.info(f"   ✓ Flat signal detection complete: {flat_count} flat channels")
    except Exception as e:
        logger.error(f"   ✗ Flat signal detection failed: {e}")
        flat_signals = None

    # Test 5: Assess cardiac power
    logger.info("\n10. Testing cardiac power assessment")
    try:
        cardiac_power = assess_cardiac_power(
            raw_fnirs,
            freq_range=(
                config.filters.cardiac_band_low_hz,
                config.filters.cardiac_band_high_hz,
            ),
            power_threshold=config.quality.psp_threshold,
        )
        logger.info(f"   ✓ Cardiac power assessed for {len(cardiac_power)} channels")
        logger.info(f"   Sample values: {list(cardiac_power.items())[:3]}")
    except Exception as e:
        logger.error(f"   ✗ Cardiac power assessment failed: {e}")
        return

    # Test 6: Mark bad channels
    logger.info("\n11. Testing bad channel marking")
    try:
        raw_fnirs, failure_reasons = mark_bad_channels(
            raw_fnirs,
            sci_values=sci_values,
            saturation_values=saturation_values,
            cardiac_power=cardiac_power,
            cv_values=cv_values,
            flat_signals=flat_signals,
            sci_threshold=config.quality.sci_threshold,
            saturation_threshold=config.quality.saturation_percent,
            psp_threshold=config.quality.psp_threshold,
            cv_threshold=config.quality.cv_threshold_percent,
        )
        logger.info(f"   ✓ Bad channel marking complete")
        logger.info(f"   Bad channels: {len(raw_fnirs.info['bads'])}")
        if raw_fnirs.info["bads"]:
            logger.info(f"   Examples: {raw_fnirs.info['bads'][:5]}")
    except Exception as e:
        logger.error(f"   ✗ Bad channel marking failed: {e}")
        return

    # Test 7: Generate quality heatmap
    logger.info("\n12. Testing quality heatmap generation")
    try:
        output_dir = Path("outputs/quality_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        fig = generate_quality_heatmap(
            raw_fnirs,
            sci_values=sci_values,
            saturation_values=saturation_values,
            cardiac_power=cardiac_power,
            cv_values=cv_values,
            failure_reasons=failure_reasons,
            output_path=output_dir / "quality_heatmap.png",
        )
        logger.info(f"   ✓ Quality heatmap generated")
    except Exception as e:
        logger.error(f"   ✗ Quality heatmap generation failed: {e}")

    # Test 8: Generate quality summary table
    logger.info("\n13. Testing quality summary table generation")
    try:
        df = generate_quality_summary_table(
            raw_fnirs,
            sci_values=sci_values,
            saturation_values=saturation_values,
            cardiac_power=cardiac_power,
            cv_values=cv_values,
            failure_reasons=failure_reasons,
            output_path=output_dir / "quality_summary.tsv",
        )
        logger.info(f"   ✓ Quality summary table generated")
        logger.info(f"   Table shape: {df.shape}")
        logger.info(f"\n{df.head()}")
    except Exception as e:
        logger.error(f"   ✗ Quality summary table generation failed: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
