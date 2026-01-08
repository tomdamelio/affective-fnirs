"""
Demonstration of BIDS Utilities

This script demonstrates the complete functionality of the BIDS compliance
utilities module, showing real-world usage scenarios.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs import (
    BIDSValidationError,
    RawDataWriteError,
    generate_derivative_path,
    validate_bids_path,
    validate_file_mode,
)


def demo_path_validation():
    """Demonstrate BIDS path validation."""
    print("=" * 70)
    print("DEMO 1: BIDS Path Validation")
    print("=" * 70)
    print()

    # Example 1: Valid path
    print("Example 1: Valid BIDS path")
    print("-" * 70)
    valid_path = "sub-002_ses-001_task-fingertapping_recording.xdf"
    try:
        validate_bids_path(valid_path)
        print(f"✓ Path is valid: {valid_path}")
    except BIDSValidationError as e:
        print(f"✗ Validation failed: {e}")
    print()

    # Example 2: Invalid path (wrong order)
    print("Example 2: Invalid BIDS path (wrong entity order)")
    print("-" * 70)
    invalid_path = "task-fingertapping_sub-002_ses-001_recording.xdf"
    try:
        validate_bids_path(invalid_path)
        print(f"✗ Path should have been rejected: {invalid_path}")
    except BIDSValidationError as e:
        print(f"✓ Correctly rejected with helpful error:")
        print(f"\n{e}\n")
    print()

    # Example 3: Missing mandatory entity
    print("Example 3: Missing mandatory 'sub' entity")
    print("-" * 70)
    missing_sub = "ses-001_task-fingertapping_recording.xdf"
    try:
        validate_bids_path(missing_sub)
        print(f"✗ Path should have been rejected: {missing_sub}")
    except BIDSValidationError as e:
        print(f"✓ Correctly rejected:")
        print(f"\n{e}\n")
    print()


def demo_derivative_path_generation():
    """Demonstrate derivative path generation."""
    print("=" * 70)
    print("DEMO 2: Derivative Path Generation")
    print("=" * 70)
    print()

    # Example 1: Full path with all entities
    print("Example 1: Quality report with all entities")
    print("-" * 70)
    path1 = generate_derivative_path(
        subject_id="002",
        session_id="001",
        task_id="fingertapping",
        suffix="desc-quality_channels",
        extension=".tsv",
    )
    print(f"Generated path:\n  {path1}")
    print()

    # Example 2: HTML report without session
    print("Example 2: Validation report without session")
    print("-" * 70)
    path2 = generate_derivative_path(
        subject_id="001",
        suffix="desc-validation_report",
        extension=".html",
    )
    print(f"Generated path:\n  {path2}")
    print()

    # Example 3: Metrics JSON with task
    print("Example 3: Metrics JSON with task")
    print("-" * 70)
    path3 = generate_derivative_path(
        subject_id="002",
        session_id="001",
        task_id="fingertapping",
        suffix="desc-validation_metrics",
        extension=".json",
    )
    print(f"Generated path:\n  {path3}")
    print()


def demo_file_mode_validation():
    """Demonstrate file mode validation."""
    print("=" * 70)
    print("DEMO 3: File Mode Validation (Raw Data Protection)")
    print("=" * 70)
    print()

    # Example 1: Read-only access to raw data (allowed)
    print("Example 1: Read-only access to raw data")
    print("-" * 70)
    raw_file = "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
    try:
        validate_file_mode(raw_file, "r")
        print(f"✓ Read-only access allowed: {raw_file}")
        print(f"  Mode: 'r' (read-only)")
    except RawDataWriteError as e:
        print(f"✗ Unexpected error: {e}")
    print()

    # Example 2: Write access to raw data (forbidden)
    print("Example 2: Write access to raw data (FORBIDDEN)")
    print("-" * 70)
    try:
        validate_file_mode(raw_file, "w")
        print(f"✗ Write access should have been blocked!")
    except RawDataWriteError as e:
        print(f"✓ Write access correctly blocked:")
        print(f"\n{e}\n")
    print()

    # Example 3: Write access to derivatives (allowed)
    print("Example 3: Write access to derivatives")
    print("-" * 70)
    derivative_file = "data/derivatives/validation-pipeline/sub-002/output.tsv"
    try:
        validate_file_mode(derivative_file, "w")
        print(f"✓ Write access allowed: {derivative_file}")
        print(f"  Mode: 'w' (write)")
    except RawDataWriteError as e:
        print(f"✗ Unexpected error: {e}")
    print()


def demo_real_world_workflow():
    """Demonstrate a real-world workflow."""
    print("=" * 70)
    print("DEMO 4: Real-World Workflow")
    print("=" * 70)
    print()

    print("Scenario: Processing subject 002, session 001, finger tapping task")
    print("-" * 70)
    print()

    # Step 1: Validate input path
    print("Step 1: Validate input XDF file path")
    input_path = "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
    try:
        validate_bids_path(input_path)
        print(f"  ✓ Input path is BIDS-compliant")
    except BIDSValidationError as e:
        print(f"  ✗ Input path validation failed: {e}")
    print()

    # Step 2: Ensure read-only access
    print("Step 2: Ensure read-only access to raw data")
    try:
        validate_file_mode(input_path, "r")
        print(f"  ✓ Read-only access verified")
    except RawDataWriteError as e:
        print(f"  ✗ File mode validation failed: {e}")
    print()

    # Step 3: Generate output paths
    print("Step 3: Generate derivative output paths")
    
    quality_path = generate_derivative_path(
        subject_id="002",
        session_id="001",
        task_id="fingertapping",
        suffix="desc-quality_channels",
        extension=".tsv",
    )
    print(f"  Quality report: {quality_path}")

    metrics_path = generate_derivative_path(
        subject_id="002",
        session_id="001",
        task_id="fingertapping",
        suffix="desc-validation_metrics",
        extension=".json",
    )
    print(f"  Metrics JSON:   {metrics_path}")

    report_path = generate_derivative_path(
        subject_id="002",
        session_id="001",
        task_id="fingertapping",
        suffix="desc-validation_report",
        extension=".html",
    )
    print(f"  HTML report:    {report_path}")
    print()

    # Step 4: Verify write access to derivatives
    print("Step 4: Verify write access to derivatives")
    try:
        validate_file_mode(str(quality_path), "w")
        print(f"  ✓ Write access to derivatives verified")
    except RawDataWriteError as e:
        print(f"  ✗ Unexpected error: {e}")
    print()

    print("✓ Workflow complete! All BIDS compliance checks passed.")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "BIDS UTILITIES DEMONSTRATION" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    demo_path_validation()
    demo_derivative_path_generation()
    demo_file_mode_validation()
    demo_real_world_workflow()

    print("=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)
    print()
