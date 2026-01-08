"""
Quick verification script for BIDS utilities.

This script tests the basic functionality of the BIDS compliance utilities
to ensure they work correctly before integration into the main pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.bids_utils import (
    validate_bids_path,
    generate_derivative_path,
    validate_file_mode,
    BIDSValidationError,
    RawDataWriteError,
)


def test_validate_bids_path():
    """Test BIDS path validation."""
    print("Testing BIDS path validation...")

    # Test valid paths
    valid_paths = [
        "sub-001_ses-001_task-fingertapping_eeg.json",
        "sub-002_task-rest_bold.nii.gz",
        "sub-001_ses-001_task-fingertapping_run-01_eeg.xdf",
    ]

    for path in valid_paths:
        try:
            result = validate_bids_path(path)
            print(f"  ✓ Valid: {path}")
        except BIDSValidationError as e:
            print(f"  ✗ Unexpected error for {path}: {e}")

    # Test invalid paths
    invalid_paths = [
        "task-rest_sub-001_eeg.json",  # Wrong order
        "ses-001_sub-001_task-rest_eeg.json",  # Wrong order
        "eeg.json",  # Missing sub entity
    ]

    for path in invalid_paths:
        try:
            validate_bids_path(path)
            print(f"  ✗ Should have failed: {path}")
        except BIDSValidationError as e:
            print(f"  ✓ Correctly rejected: {path}")

    print()


def test_generate_derivative_path():
    """Test derivative path generation."""
    print("Testing derivative path generation...")

    # Test with all parameters
    path1 = generate_derivative_path(
        subject_id="001",
        session_id="001",
        task_id="fingertapping",
        suffix="desc-quality_channels",
        extension=".tsv",
    )
    expected1 = Path(
        "data/derivatives/validation-pipeline/sub-001/ses-001/"
        "sub-001_ses-001_task-fingertapping_desc-quality_channels.tsv"
    )
    print(f"  Generated: {path1}")
    print(f"  Expected:  {expected1}")
    print(f"  Match: {path1 == expected1}")

    # Test without session
    path2 = generate_derivative_path(
        subject_id="002",
        suffix="desc-validation_report",
        extension=".html",
    )
    expected2 = Path(
        "data/derivatives/validation-pipeline/sub-002/"
        "sub-002_desc-validation_report.html"
    )
    print(f"  Generated: {path2}")
    print(f"  Expected:  {expected2}")
    print(f"  Match: {path2 == expected2}")

    print()


def test_validate_file_mode():
    """Test file mode validation."""
    print("Testing file mode validation...")

    # Test read-only access to raw data (should pass)
    try:
        validate_file_mode("data/raw/sub-001/file.xdf", "r")
        print("  ✓ Read-only access to raw data allowed")
    except RawDataWriteError:
        print("  ✗ Read-only access incorrectly rejected")

    # Test write access to raw data (should fail)
    try:
        validate_file_mode("data/raw/sub-001/file.xdf", "w")
        print("  ✗ Write access to raw data incorrectly allowed")
    except RawDataWriteError:
        print("  ✓ Write access to raw data correctly rejected")

    # Test write access to derivatives (should pass)
    try:
        validate_file_mode("data/derivatives/output.tsv", "w")
        print("  ✓ Write access to derivatives allowed")
    except RawDataWriteError:
        print("  ✗ Write access to derivatives incorrectly rejected")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("BIDS Utilities Verification")
    print("=" * 60)
    print()

    test_validate_bids_path()
    test_generate_derivative_path()
    test_validate_file_mode()

    print("=" * 60)
    print("Verification complete!")
    print("=" * 60)
