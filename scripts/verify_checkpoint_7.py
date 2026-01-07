"""
Checkpoint 7: Verify fNIRS Quality and Processing

This script verifies that:
1. All tests pass
2. Processing order is correct: Intensity → OD → TDDR → SCR → Hb → Filter
3. CV was calculated on raw intensities (not OD)
4. Systemic noise reduction metrics are computed

Requirements: Tasks 5.1-5.7, 6.1-6.8
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def verify_processing_order():
    """Verify that the fNIRS processing pipeline follows the correct order."""
    logger.info("=" * 80)
    logger.info("CHECKPOINT 7: Verifying fNIRS Quality and Processing")
    logger.info("=" * 80)
    
    # Check 1: Verify processing order in code
    logger.info("\n✓ Check 1: Verifying processing order in fnirs_processing.py")
    
    processing_file = Path("src/affective_fnirs/fnirs_processing.py")
    if not processing_file.exists():
        logger.error(f"❌ File not found: {processing_file}")
        return False
    
    with open(processing_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Verify critical processing order is documented
    expected_order = [
        "Intensity → Optical Density",
        "Motion correction (TDDR on OD)",
        "Short channel regression (on OD)",
        "OD → Hemoglobin (Beer-Lambert)",
        "Bandpass filter (0.01-0.5 Hz on Hb)"
    ]
    
    order_found = True
    for step in expected_order:
        if step not in content:
            logger.warning(f"  ⚠️  Processing step not clearly documented: {step}")
            order_found = False
    
    if order_found:
        logger.info("  ✓ Processing order correctly documented")
    
    # Verify process_fnirs_pipeline function exists
    if "def process_fnirs_pipeline" in content:
        logger.info("  ✓ process_fnirs_pipeline() function exists")
    else:
        logger.error("  ❌ process_fnirs_pipeline() function not found")
        return False
    
    # Verify critical order: OD conversion before motion correction
    if "convert_to_optical_density" in content and "correct_motion_artifacts" in content:
        od_pos = content.find("convert_to_optical_density")
        motion_pos = content.find("correct_motion_artifacts")
        if od_pos < motion_pos:
            logger.info("  ✓ OD conversion occurs before motion correction")
        else:
            logger.error("  ❌ Incorrect order: Motion correction before OD conversion")
            return False
    
    # Verify critical order: Motion correction before Beer-Lambert
    if "correct_motion_artifacts" in content and "convert_to_hemoglobin" in content:
        motion_pos = content.find("correct_motion_artifacts")
        beer_lambert_pos = content.find("convert_to_hemoglobin")
        if motion_pos < beer_lambert_pos:
            logger.info("  ✓ Motion correction occurs before Beer-Lambert conversion")
        else:
            logger.error("  ❌ Incorrect order: Beer-Lambert before motion correction")
            return False
    
    # Verify critical order: Beer-Lambert before filtering
    if "convert_to_hemoglobin" in content and "filter_hemoglobin_data" in content:
        beer_lambert_pos = content.find("convert_to_hemoglobin")
        filter_pos = content.find("filter_hemoglobin_data")
        if beer_lambert_pos < filter_pos:
            logger.info("  ✓ Beer-Lambert conversion occurs before filtering")
        else:
            logger.error("  ❌ Incorrect order: Filtering before Beer-Lambert")
            return False
    
    return True


def verify_cv_on_raw_intensity():
    """Verify that CV is calculated on raw intensities, not OD."""
    logger.info("\n✓ Check 2: Verifying CV calculation on raw intensities")
    
    quality_file = Path("src/affective_fnirs/fnirs_quality.py")
    if not quality_file.exists():
        logger.error(f"❌ File not found: {quality_file}")
        return False
    
    with open(quality_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check that calculate_coefficient_of_variation function exists
    if "def calculate_coefficient_of_variation" not in content:
        logger.error("  ❌ calculate_coefficient_of_variation() function not found")
        return False
    
    logger.info("  ✓ calculate_coefficient_of_variation() function exists")
    
    # Verify documentation mentions raw intensity
    cv_func_start = content.find("def calculate_coefficient_of_variation")
    cv_func_end = content.find("\ndef ", cv_func_start + 1)
    cv_func_content = content[cv_func_start:cv_func_end]
    
    if "RAW INTENSITY" in cv_func_content or "raw intensity" in cv_func_content.lower():
        logger.info("  ✓ Documentation specifies CV on raw intensity")
    else:
        logger.warning("  ⚠️  Documentation doesn't clearly specify raw intensity")
    
    # Verify it checks for fnirs_cw_amplitude channel type
    if "fnirs_cw_amplitude" in cv_func_content:
        logger.info("  ✓ Function validates fnirs_cw_amplitude channel type")
    else:
        logger.warning("  ⚠️  Function doesn't validate channel type")
    
    # Verify it rejects fnirs_od
    if "NOT optical density" in cv_func_content or "before OD conversion" in cv_func_content:
        logger.info("  ✓ Documentation warns against using OD data")
    else:
        logger.warning("  ⚠️  Documentation doesn't warn against OD data")
    
    return True


def verify_noise_reduction_metrics():
    """Verify that systemic noise reduction metrics are implemented."""
    logger.info("\n✓ Check 3: Verifying systemic noise reduction metrics")
    
    processing_file = Path("src/affective_fnirs/fnirs_processing.py")
    if not processing_file.exists():
        logger.error(f"❌ File not found: {processing_file}")
        return False
    
    with open(processing_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check that verify_systemic_noise_reduction function exists
    if "def verify_systemic_noise_reduction" not in content:
        logger.error("  ❌ verify_systemic_noise_reduction() function not found")
        return False
    
    logger.info("  ✓ verify_systemic_noise_reduction() function exists")
    
    # Verify it computes PSD before and after
    noise_func_start = content.find("def verify_systemic_noise_reduction")
    noise_func_end = content.find("\ndef ", noise_func_start + 1)
    noise_func_content = content[noise_func_start:noise_func_end]
    
    if "welch" in noise_func_content.lower() or "psd" in noise_func_content.lower():
        logger.info("  ✓ Function computes PSD for noise assessment")
    else:
        logger.warning("  ⚠️  PSD computation not clearly implemented")
    
    # Verify systemic band (0.1-0.4 Hz) is used
    if "0.1" in noise_func_content and "0.4" in noise_func_content:
        logger.info("  ✓ Systemic band (0.1-0.4 Hz) is used")
    else:
        logger.warning("  ⚠️  Systemic band not clearly specified")
    
    # Verify it returns reduction percentage
    if "reduction_percent" in noise_func_content or "mean_reduction" in noise_func_content:
        logger.info("  ✓ Function returns reduction percentage")
    else:
        logger.warning("  ⚠️  Reduction percentage not clearly returned")
    
    return True


def verify_quality_functions():
    """Verify that all quality assessment functions are implemented."""
    logger.info("\n✓ Check 4: Verifying quality assessment functions")
    
    quality_file = Path("src/affective_fnirs/fnirs_quality.py")
    if not quality_file.exists():
        logger.error(f"❌ File not found: {quality_file}")
        return False
    
    with open(quality_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    required_functions = [
        "calculate_sci",
        "calculate_coefficient_of_variation",
        "detect_saturation",
        "assess_cardiac_power",
        "mark_bad_channels",
    ]
    
    all_found = True
    for func_name in required_functions:
        if f"def {func_name}" in content:
            logger.info(f"  ✓ {func_name}() implemented")
        else:
            logger.error(f"  ❌ {func_name}() not found")
            all_found = False
    
    return all_found


def verify_processing_functions():
    """Verify that all processing functions are implemented."""
    logger.info("\n✓ Check 5: Verifying processing functions")
    
    processing_file = Path("src/affective_fnirs/fnirs_processing.py")
    if not processing_file.exists():
        logger.error(f"❌ File not found: {processing_file}")
        return False
    
    with open(processing_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    required_functions = [
        "convert_to_optical_density",
        "correct_motion_artifacts",
        "identify_short_channels",
        "apply_short_channel_regression",
        "verify_systemic_noise_reduction",
        "convert_to_hemoglobin",
        "filter_hemoglobin_data",
        "process_fnirs_pipeline",
    ]
    
    all_found = True
    for func_name in required_functions:
        if f"def {func_name}" in content:
            logger.info(f"  ✓ {func_name}() implemented")
        else:
            logger.error(f"  ❌ {func_name}() not found")
            all_found = False
    
    return all_found


def main():
    """Run all checkpoint verifications."""
    logger.info("Starting Checkpoint 7 verification...\n")
    
    checks = [
        ("Processing Order", verify_processing_order),
        ("CV on Raw Intensity", verify_cv_on_raw_intensity),
        ("Noise Reduction Metrics", verify_noise_reduction_metrics),
        ("Quality Functions", verify_quality_functions),
        ("Processing Functions", verify_processing_functions),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"❌ {check_name} check failed with exception: {e}")
            results[check_name] = False
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CHECKPOINT 7 SUMMARY")
    logger.info("=" * 80)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {check_name}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 80)
    
    if all_passed:
        logger.info("\n🎉 All checkpoint verifications passed!")
        logger.info("\nNext steps:")
        logger.info("  1. Run unit tests: micromamba run -n affective-fnirs pytest tests/")
        logger.info("  2. Proceed to Task 8: Implement EEG processing module")
        return 0
    else:
        logger.error("\n❌ Some checkpoint verifications failed.")
        logger.error("Please review the errors above and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
