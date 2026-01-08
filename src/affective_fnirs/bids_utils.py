"""
BIDS Compliance Utilities

This module provides utilities for validating BIDS (Brain Imaging Data Structure)
compliance, generating derivative output paths, and enforcing read-only access
to raw data.

Scientific Context:
    BIDS is a standardized format for organizing neuroimaging data. It ensures
    that data is structured consistently across studies, enabling reproducibility
    and data sharing. Key principles:
    - Entity ordering: sub-XX_ses-XX_task-XX (fixed order)
    - Key-value pairs: separated by underscores
    - Derivatives: stored in data/derivatives/<pipeline_name>/
    - Raw data immutability: never write to data/raw/

References:
    - BIDS Specification: https://bids-specification.readthedocs.io/
    - Gorgolewski et al. (2016). The brain imaging data structure. Scientific Data.

Requirements:
    - 9.1: Read-only enforcement for raw data
    - 9.2: Derivative outputs to data/derivatives/validation-pipeline/
    - 9.3: BIDS naming conventions with key-value pairs
    - 9.5: Validate entity ordering (sub-XX_ses-XX_task-XX)
    - 9.6: Provide specific guidance for incorrect paths
"""

import re
from pathlib import Path
from typing import Optional


class BIDSValidationError(Exception):
    """Exception raised when BIDS validation fails."""

    pass


class RawDataWriteError(Exception):
    """Exception raised when attempting to write to raw data directory."""

    pass


def validate_bids_path(file_path: str | Path) -> bool:
    """
    Validate that a file path follows BIDS entity ordering conventions.

    BIDS requires entities to appear in a specific order:
    sub-<label>_ses-<label>_task-<label>_[other entities]_<suffix>.<extension>

    The mandatory entity order is:
    1. sub (subject)
    2. ses (session) - optional but must come after sub if present
    3. task - optional but must come after ses if present
    4. Other entities (run, acq, etc.)
    5. Suffix (e.g., eeg, nirs, bold)
    6. Extension (e.g., .xdf, .json, .nii.gz)

    Args:
        file_path: Path to validate (can be string or Path object)

    Returns:
        True if path follows BIDS conventions

    Raises:
        BIDSValidationError: If path violates BIDS entity ordering with
            specific guidance on correct format

    Examples:
        >>> validate_bids_path("sub-001_ses-001_task-fingertapping_eeg.json")
        True

        >>> validate_bids_path("task-rest_sub-001_eeg.json")
        BIDSValidationError: Incorrect entity ordering...

    References:
        - BIDS Specification Section 2.3: Entity ordering
    """
    # Convert to Path object and extract filename
    path = Path(file_path)
    filename = path.name

    # Remove extension(s) to focus on entity ordering
    # Handle double extensions like .nii.gz
    stem = filename
    while stem != Path(stem).stem:
        stem = Path(stem).stem

    # Split by underscores to get entities
    parts = stem.split("_")

    if len(parts) < 2:
        raise BIDSValidationError(
            f"Invalid BIDS filename: '{filename}'\n"
            f"BIDS filenames must contain at least a subject entity and suffix.\n"
            f"Expected format: sub-<label>_[ses-<label>_][task-<label>_]<suffix>\n"
            f"Example: sub-001_ses-001_task-fingertapping_eeg.json"
        )

    # Define expected entity order (entities that have ordering constraints)
    # The last part is the suffix (e.g., 'eeg', 'nirs', 'bold')
    ordered_entities = ["sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "echo"]

    # Extract entities from filename parts (exclude suffix which is last part)
    found_entities = []
    entity_positions = {}

    for i, part in enumerate(parts[:-1]):  # Exclude last part (suffix)
        if "-" in part:
            entity_key = part.split("-")[0]
            found_entities.append(entity_key)
            entity_positions[entity_key] = i

    # Check if 'sub' entity is present (mandatory)
    if "sub" not in found_entities:
        raise BIDSValidationError(
            f"Invalid BIDS filename: '{filename}'\n"
            f"Missing mandatory 'sub' (subject) entity.\n"
            f"Expected format: sub-<label>_[ses-<label>_][task-<label>_]<suffix>\n"
            f"Example: sub-001_ses-001_task-fingertapping_eeg.json"
        )

    # Check entity ordering for entities that have ordering constraints
    relevant_entities = [e for e in ordered_entities if e in found_entities]

    for i in range(len(relevant_entities) - 1):
        current_entity = relevant_entities[i]
        next_entity = relevant_entities[i + 1]

        current_pos = entity_positions[current_entity]
        next_pos = entity_positions[next_entity]

        if current_pos > next_pos:
            # Build correct ordering example
            correct_parts = []
            for entity in ordered_entities:
                if entity in entity_positions:
                    # Find the original part with this entity
                    for part in parts[:-1]:
                        if part.startswith(f"{entity}-"):
                            correct_parts.append(part)
                            break
            correct_parts.append(parts[-1])  # Add suffix
            correct_filename = "_".join(correct_parts) + "".join(path.suffixes)

            raise BIDSValidationError(
                f"Invalid BIDS filename: '{filename}'\n"
                f"Incorrect entity ordering: '{current_entity}' must come before '{next_entity}'.\n"
                f"\n"
                f"Current order: {' → '.join(found_entities)}\n"
                f"Required order: {' → '.join(ordered_entities[:len(relevant_entities)])}\n"
                f"\n"
                f"Correct filename: {correct_filename}\n"
                f"\n"
                f"BIDS entity ordering rules:\n"
                f"  1. sub (subject) - mandatory, always first\n"
                f"  2. ses (session) - optional, must follow sub\n"
                f"  3. task - optional, must follow ses\n"
                f"  4. Other entities (acq, run, etc.) - follow task\n"
                f"  5. Suffix (e.g., eeg, nirs) - always last before extension"
            )

    return True


def generate_derivative_path(
    subject_id: str,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    suffix: str = "",
    extension: str = "",
    pipeline_name: str = "validation-pipeline",
    base_dir: Path = Path("data/derivatives"),
) -> Path:
    """
    Generate BIDS-compliant derivative output path.

    Creates paths following BIDS derivative structure:
    data/derivatives/<pipeline_name>/sub-<label>/[ses-<label>/]<filename>

    Args:
        subject_id: Subject identifier (without 'sub-' prefix)
        session_id: Session identifier (without 'ses-' prefix), optional
        task_id: Task identifier (without 'task-' prefix), optional
        suffix: BIDS suffix (e.g., 'desc-quality_channels', 'bold')
        extension: File extension (e.g., '.tsv', '.json', '.html')
        pipeline_name: Name of the processing pipeline (default: 'validation-pipeline')
        base_dir: Base derivatives directory (default: 'data/derivatives')

    Returns:
        Path object with BIDS-compliant derivative path

    Examples:
        >>> generate_derivative_path(
        ...     subject_id="001",
        ...     session_id="001",
        ...     task_id="fingertapping",
        ...     suffix="desc-quality_channels",
        ...     extension=".tsv"
        ... )
        Path('data/derivatives/validation-pipeline/sub-001/ses-001/
             sub-001_ses-001_task-fingertapping_desc-quality_channels.tsv')

        >>> generate_derivative_path(
        ...     subject_id="002",
        ...     suffix="desc-validation_report",
        ...     extension=".html"
        ... )
        Path('data/derivatives/validation-pipeline/sub-002/
             sub-002_desc-validation_report.html')

    Requirements:
        - 9.2: Output to data/derivatives/validation-pipeline/
        - 9.3: BIDS naming with key-value pairs

    References:
        - BIDS Specification Section 5: Derivatives
    """
    # Build directory structure
    pipeline_dir = base_dir / pipeline_name
    subject_dir = pipeline_dir / f"sub-{subject_id}"

    if session_id:
        output_dir = subject_dir / f"ses-{session_id}"
    else:
        output_dir = subject_dir

    # Build filename with BIDS entity ordering
    filename_parts = [f"sub-{subject_id}"]

    if session_id:
        filename_parts.append(f"ses-{session_id}")

    if task_id:
        filename_parts.append(f"task-{task_id}")

    if suffix:
        filename_parts.append(suffix)

    filename = "_".join(filename_parts) + extension

    return output_dir / filename


def validate_file_mode(file_path: str | Path, mode: str) -> None:
    """
    Validate that file operations on raw data use read-only mode.

    Enforces data immutability by preventing write operations on raw data.
    This is a critical BIDS principle: raw data must never be modified.

    Args:
        file_path: Path to file being accessed
        mode: File access mode (e.g., 'r', 'w', 'a', 'r+')

    Raises:
        RawDataWriteError: If attempting to open raw data with write permissions

    Examples:
        >>> validate_file_mode("data/raw/sub-001/file.xdf", "r")  # OK
        >>> validate_file_mode("data/raw/sub-001/file.xdf", "w")  # Raises error
        >>> validate_file_mode("data/derivatives/output.tsv", "w")  # OK

    Requirements:
        - 9.1: Never open files in data/raw with write permissions

    References:
        - BIDS Specification: Raw data immutability principle
    """
    path = Path(file_path)

    # Check if path is in raw data directory
    # Handle both absolute and relative paths
    path_parts = path.parts

    # Look for 'raw' directory in path
    is_raw_data = False
    if "raw" in path_parts:
        # Check if it's in a data/raw structure
        for i, part in enumerate(path_parts):
            if part == "raw" and i > 0 and path_parts[i - 1] == "data":
                is_raw_data = True
                break

    # Check if mode includes write permissions
    write_modes = ["w", "a", "r+", "w+", "a+", "x"]
    has_write_permission = any(write_mode in mode for write_mode in write_modes)

    if is_raw_data and has_write_permission:
        raise RawDataWriteError(
            f"Attempted to open raw data file with write permissions.\n"
            f"File: {file_path}\n"
            f"Mode: {mode}\n"
            f"\n"
            f"BIDS Principle: Raw data must remain immutable.\n"
            f"Raw data files in 'data/raw/' can only be opened in read-only mode ('r', 'rb').\n"
            f"\n"
            f"If you need to save processed data:\n"
            f"  1. Use data/derivatives/<pipeline_name>/ directory\n"
            f"  2. Use generate_derivative_path() to create output paths\n"
            f"  3. Never modify files in data/raw/\n"
            f"\n"
            f"Allowed modes for raw data: 'r', 'rb'\n"
            f"Forbidden modes for raw data: 'w', 'a', 'r+', 'w+', 'a+', 'x'"
        )
