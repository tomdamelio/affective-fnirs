"""Unit tests for src/affective_fnirs/validation.py.

Validates Requirements 5.1, 5.2, 5.3, 5.4, 5.5:
- Epoch count parity within 10% tolerance
- Epoch duration within 6.0s ± 0.1s
- EEG/fNIRS presence checks
- Warning logging on failures, summary on success
"""

import sys
from pathlib import Path

import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.validation import (
    NothingValidationResult,
    _check_condition_presence,
    _check_epoch_count_parity,
    _check_epoch_durations,
    _count_condition_epochs,
    validate_nothing_condition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eeg_epochs(
    condition_counts: dict[str, int],
    n_channels: int = 3,
    sfreq: float = 256.0,
    epoch_duration_sec: float = 6.0,
) -> mne.Epochs:
    """Build a minimal EpochsArray with the requested conditions."""
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    n_times = int(epoch_duration_sec * sfreq)
    rng = np.random.default_rng(99)

    all_data: list[np.ndarray] = []
    all_events: list[list[int]] = []
    event_id: dict[str, int] = {}
    code = 1
    sample_offset = 0

    for condition_name, n_trials in condition_counts.items():
        if n_trials == 0:
            code += 1
            continue
        event_id[condition_name] = code
        for _ in range(n_trials):
            all_data.append(rng.standard_normal((n_channels, n_times)) * 1e-6)
            all_events.append([sample_offset, 0, code])
            sample_offset += n_times
        code += 1

    data_array = np.array(all_data) if all_data else np.empty((0, n_channels, n_times))
    events_array = np.array(all_events, dtype=int) if all_events else np.empty((0, 3), dtype=int)

    return mne.EpochsArray(
        data_array, info, events=events_array, tmin=0.0, event_id=event_id,
    )


def _make_nothing_annotations(
    n_annotations: int,
    duration: float = 6.0,
    onset_start: float = 10.0,
    onset_spacing: float = 15.0,
) -> mne.Annotations:
    """Build NOTHING annotations with uniform duration."""
    onsets = [onset_start + idx * onset_spacing for idx in range(n_annotations)]
    durations = [duration] * n_annotations
    descriptions = ["NOTHING"] * n_annotations
    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


# ---------------------------------------------------------------------------
# NothingValidationResult dataclass
# ---------------------------------------------------------------------------

class TestNothingValidationResult:
    """Basic dataclass behaviour."""

    def test_all_passed_true(self) -> None:
        result = NothingValidationResult(
            epoch_count_ok=True,
            epoch_duration_ok=True,
            eeg_presence_ok=True,
            fnirs_presence_ok=True,
            n_nothing_epochs=10,
            n_left_epochs=10,
            n_right_epochs=10,
        )
        assert result.all_passed is True
        assert result.warnings == []

    def test_all_passed_false_when_count_fails(self) -> None:
        result = NothingValidationResult(
            epoch_count_ok=False,
            epoch_duration_ok=True,
            eeg_presence_ok=True,
            fnirs_presence_ok=True,
            n_nothing_epochs=5,
            n_left_epochs=10,
            n_right_epochs=10,
            warnings=["count mismatch"],
        )
        assert result.all_passed is False


# ---------------------------------------------------------------------------
# _check_epoch_count_parity
# ---------------------------------------------------------------------------

class TestCheckEpochCountParity:

    def test_exact_match_passes(self) -> None:
        passed, warning = _check_epoch_count_parity(10, 10, "LEFT", 0.1)
        assert passed is True
        assert warning is None

    def test_within_tolerance_passes(self) -> None:
        # 9 vs 10 → 10% deviation, exactly at boundary
        passed, warning = _check_epoch_count_parity(9, 10, "LEFT", 0.1)
        assert passed is True

    def test_exceeds_tolerance_fails(self) -> None:
        # 8 vs 10 → 20% deviation
        passed, warning = _check_epoch_count_parity(8, 10, "LEFT", 0.1)
        assert passed is False
        assert "LEFT" in warning
        assert "20" in warning  # 20% deviation

    def test_zero_reference_fails(self) -> None:
        passed, warning = _check_epoch_count_parity(5, 0, "RIGHT", 0.1)
        assert passed is False
        assert "reference count is 0" in warning


# ---------------------------------------------------------------------------
# _check_epoch_durations
# ---------------------------------------------------------------------------

class TestCheckEpochDurations:

    def test_all_within_tolerance(self) -> None:
        annotations = _make_nothing_annotations(5, duration=6.0)
        passed, warning = _check_epoch_durations(annotations, 6.0, 0.1)
        assert passed is True
        assert warning is None

    def test_edge_of_tolerance_passes(self) -> None:
        annotations = _make_nothing_annotations(3, duration=6.1)
        passed, warning = _check_epoch_durations(annotations, 6.0, 0.1)
        assert passed is True

    def test_out_of_tolerance_fails(self) -> None:
        annotations = _make_nothing_annotations(3, duration=6.5)
        passed, warning = _check_epoch_durations(annotations, 6.0, 0.1)
        assert passed is False
        assert "out of range" in warning

    def test_empty_annotations_passes(self) -> None:
        annotations = mne.Annotations(onset=[], duration=[], description=[])
        passed, warning = _check_epoch_durations(annotations, 6.0, 0.1)
        assert passed is True


# ---------------------------------------------------------------------------
# _check_condition_presence
# ---------------------------------------------------------------------------

class TestCheckConditionPresence:

    def test_nothing_present(self) -> None:
        epochs = _make_eeg_epochs({"LEFT": 3, "RIGHT": 3, "NOTHING": 3})
        passed, warning = _check_condition_presence(epochs, "EEG")
        assert passed is True

    def test_nothing_absent(self) -> None:
        epochs = _make_eeg_epochs({"LEFT": 3, "RIGHT": 3})
        passed, warning = _check_condition_presence(epochs, "EEG")
        assert passed is False
        assert "NOTHING" in warning

    def test_none_epochs_skipped(self) -> None:
        passed, warning = _check_condition_presence(None, "fNIRS")
        assert passed is True
        assert warning is None


# ---------------------------------------------------------------------------
# validate_nothing_condition (integration)
# ---------------------------------------------------------------------------

class TestValidateNothingCondition:

    def test_all_checks_pass(self) -> None:
        """Validates Req 5.5: summary logged when all pass."""
        eeg_epochs = _make_eeg_epochs({"LEFT": 10, "RIGHT": 10, "NOTHING": 10})
        annotations = _make_nothing_annotations(10, duration=6.0)
        result = validate_nothing_condition(
            eeg_epochs=eeg_epochs,
            fnirs_epochs=None,
            nothing_annotations=annotations,
        )
        assert result.all_passed is True
        assert result.warnings == []
        assert result.n_nothing_epochs == 10
        assert result.n_left_epochs == 10
        assert result.n_right_epochs == 10

    def test_count_parity_failure(self) -> None:
        """Validates Req 5.1, 5.4: warning on count mismatch."""
        eeg_epochs = _make_eeg_epochs({"LEFT": 10, "RIGHT": 10, "NOTHING": 5})
        result = validate_nothing_condition(
            eeg_epochs=eeg_epochs, fnirs_epochs=None,
        )
        assert result.epoch_count_ok is False
        assert result.all_passed is False
        assert any("parity" in w for w in result.warnings)

    def test_duration_failure(self) -> None:
        """Validates Req 5.2, 5.4: warning on bad duration."""
        eeg_epochs = _make_eeg_epochs({"LEFT": 5, "RIGHT": 5, "NOTHING": 5})
        bad_annotations = _make_nothing_annotations(5, duration=7.0)
        result = validate_nothing_condition(
            eeg_epochs=eeg_epochs,
            fnirs_epochs=None,
            nothing_annotations=bad_annotations,
        )
        assert result.epoch_duration_ok is False
        assert any("duration" in w.lower() for w in result.warnings)

    def test_eeg_presence_failure(self) -> None:
        """Validates Req 5.3: NOTHING missing from EEG epochs."""
        eeg_epochs = _make_eeg_epochs({"LEFT": 5, "RIGHT": 5})
        result = validate_nothing_condition(
            eeg_epochs=eeg_epochs, fnirs_epochs=None,
        )
        assert result.eeg_presence_ok is False

    def test_fnirs_none_skips_check(self) -> None:
        """Validates error handling: fNIRS absent → fnirs_presence_ok=True."""
        eeg_epochs = _make_eeg_epochs({"LEFT": 5, "RIGHT": 5, "NOTHING": 5})
        result = validate_nothing_condition(
            eeg_epochs=eeg_epochs, fnirs_epochs=None,
        )
        assert result.fnirs_presence_ok is True

    def test_both_eeg_and_fnirs_none(self) -> None:
        """Edge case: both modalities absent."""
        result = validate_nothing_condition(
            eeg_epochs=None, fnirs_epochs=None,
        )
        assert result.eeg_presence_ok is True
        assert result.fnirs_presence_ok is True
        assert result.n_nothing_epochs == 0

    def test_no_annotations_skips_duration_check(self) -> None:
        """When nothing_annotations is None, duration check passes."""
        eeg_epochs = _make_eeg_epochs({"LEFT": 5, "RIGHT": 5, "NOTHING": 5})
        result = validate_nothing_condition(
            eeg_epochs=eeg_epochs,
            fnirs_epochs=None,
            nothing_annotations=None,
        )
        assert result.epoch_duration_ok is True
