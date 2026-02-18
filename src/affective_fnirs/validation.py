"""Validation checks for the NOTHING (REST) condition.

Ensures NOTHING epochs are correctly synthesized and present across
all EEG and fNIRS analysis outputs. Checks epoch counts, durations,
and condition-key presence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mne

logger = logging.getLogger(__name__)


@dataclass
class NothingValidationResult:
    """Results from NOTHING condition validation checks.

    Attributes:
        epoch_count_ok: True when NOTHING count is within tolerance of
            both LEFT and RIGHT counts.
        epoch_duration_ok: True when every NOTHING annotation duration
            falls within the expected window.
        eeg_presence_ok: True when ``'NOTHING'`` exists in
            ``eeg_epochs.event_id``.
        fnirs_presence_ok: True when ``'NOTHING'`` exists in
            ``fnirs_epochs.event_id`` (defaults to True when fNIRS
            data is unavailable).
        n_nothing_epochs: Number of NOTHING epochs found.
        n_left_epochs: Number of LEFT epochs found.
        n_right_epochs: Number of RIGHT epochs found.
        warnings: Human-readable messages for every failed check.
    """

    epoch_count_ok: bool
    epoch_duration_ok: bool
    eeg_presence_ok: bool
    fnirs_presence_ok: bool
    n_nothing_epochs: int
    n_left_epochs: int
    n_right_epochs: int
    warnings: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Return True only when every validation check succeeded."""
        return (
            self.epoch_count_ok
            and self.epoch_duration_ok
            and self.eeg_presence_ok
            and self.fnirs_presence_ok
        )


def _check_epoch_count_parity(
    n_nothing: int,
    n_reference: int,
    reference_label: str,
    tolerance_fraction: float,
) -> tuple[bool, str | None]:
    """Check that NOTHING count is within *tolerance_fraction* of a reference.

    Args:
        n_nothing: Number of NOTHING epochs.
        n_reference: Number of reference-condition epochs (LEFT or RIGHT).
        reference_label: Human-readable name of the reference condition.
        tolerance_fraction: Maximum allowed relative deviation (e.g. 0.1
            for 10 %).

    Returns:
        A ``(passed, warning_message | None)`` tuple.
    """
    if n_reference == 0:
        warning_msg = (
            f"Epoch count parity ({reference_label}): "
            f"reference count is 0 — cannot evaluate parity"
        )
        return False, warning_msg

    deviation = abs(n_nothing - n_reference) / n_reference
    if deviation > tolerance_fraction:
        warning_msg = (
            f"Epoch count parity ({reference_label}): "
            f"expected NOTHING within {tolerance_fraction:.0%} of "
            f"{n_reference}, got {n_nothing} "
            f"(deviation {deviation:.1%})"
        )
        return False, warning_msg

    return True, None


def _check_epoch_durations(
    nothing_annotations: mne.Annotations,
    expected_duration_sec: float,
    duration_tolerance_sec: float,
) -> tuple[bool, str | None]:
    """Verify every NOTHING annotation duration is within tolerance.

    Args:
        nothing_annotations: Annotations filtered to NOTHING events.
        expected_duration_sec: Target duration in seconds.
        duration_tolerance_sec: Allowed absolute deviation in seconds.

    Returns:
        A ``(passed, warning_message | None)`` tuple.
    """
    nothing_mask = nothing_annotations.description == "NOTHING"
    nothing_durations = nothing_annotations.duration[nothing_mask]

    if len(nothing_durations) == 0:
        return True, None

    lower_bound = expected_duration_sec - duration_tolerance_sec
    upper_bound = expected_duration_sec + duration_tolerance_sec

    out_of_range = [
        dur for dur in nothing_durations
        if dur < lower_bound or dur > upper_bound
    ]

    if out_of_range:
        warning_msg = (
            f"Epoch duration check: expected {expected_duration_sec}s "
            f"± {duration_tolerance_sec}s, but {len(out_of_range)} of "
            f"{len(nothing_durations)} NOTHING annotations are out of "
            f"range (e.g. {out_of_range[0]:.3f}s)"
        )
        return False, warning_msg

    return True, None


def _check_condition_presence(
    epochs: mne.Epochs | None,
    modality_label: str,
) -> tuple[bool, str | None]:
    """Check that ``'NOTHING'`` is a key in *epochs.event_id*.

    Args:
        epochs: MNE Epochs object (may be ``None`` for optional modalities).
        modality_label: Human-readable modality name (e.g. ``"EEG"``).

    Returns:
        A ``(passed, warning_message | None)`` tuple.  When *epochs* is
        ``None``, returns ``(True, None)`` — the check is skipped.
    """
    if epochs is None:
        return True, None

    if "NOTHING" in epochs.event_id:
        return True, None

    available_conditions = list(epochs.event_id.keys())
    warning_msg = (
        f"{modality_label} presence check: 'NOTHING' not found in "
        f"event_id; available conditions: {available_conditions}"
    )
    return False, warning_msg


def _count_condition_epochs(
    epochs: mne.Epochs | None,
    condition: str,
) -> int:
    """Return the number of epochs for *condition*, or 0 if unavailable.

    Args:
        epochs: MNE Epochs object (may be ``None``).
        condition: Event-id key to count (e.g. ``"LEFT"``).

    Returns:
        Epoch count for the requested condition.
    """
    if epochs is None:
        return 0
    if condition not in epochs.event_id:
        return 0
    return len(epochs[condition])


def validate_nothing_condition(
    eeg_epochs: mne.Epochs | None,
    fnirs_epochs: mne.Epochs | None,
    nothing_annotations: mne.Annotations | None = None,
    expected_duration_sec: float = 6.0,
    count_tolerance_fraction: float = 0.1,
    duration_tolerance_sec: float = 0.1,
) -> NothingValidationResult:
    """Run all NOTHING-condition validation checks.

    Checks performed:
        1. **Count parity** — NOTHING epoch count is within
           *count_tolerance_fraction* of both LEFT and RIGHT counts.
        2. **Duration** — Every NOTHING annotation duration equals
           *expected_duration_sec* ± *duration_tolerance_sec*.
        3. **EEG presence** — ``'NOTHING'`` key exists in
           ``eeg_epochs.event_id``.
        4. **fNIRS presence** — ``'NOTHING'`` key exists in
           ``fnirs_epochs.event_id`` (skipped when *fnirs_epochs* is
           ``None``).

    Args:
        eeg_epochs: EEG Epochs object (may be ``None``).
        fnirs_epochs: fNIRS Epochs object (may be ``None``; fNIRS
            checks are skipped when absent).
        nothing_annotations: Annotations containing NOTHING events,
            used for the duration check.  When ``None`` the duration
            check is skipped (passes by default).
        expected_duration_sec: Target NOTHING epoch duration in seconds.
        count_tolerance_fraction: Maximum allowed relative deviation
            between NOTHING and LEFT/RIGHT counts (0.1 = 10 %).
        duration_tolerance_sec: Allowed absolute deviation for
            annotation durations in seconds.

    Returns:
        A :class:`NothingValidationResult` summarising all checks.
    """
    collected_warnings: list[str] = []

    # --- Epoch counts ---
    n_nothing = _count_condition_epochs(eeg_epochs, "NOTHING")
    n_left = _count_condition_epochs(eeg_epochs, "LEFT")
    n_right = _count_condition_epochs(eeg_epochs, "RIGHT")

    left_ok, left_warn = _check_epoch_count_parity(
        n_nothing, n_left, "LEFT", count_tolerance_fraction,
    )
    right_ok, right_warn = _check_epoch_count_parity(
        n_nothing, n_right, "RIGHT", count_tolerance_fraction,
    )
    epoch_count_ok = left_ok and right_ok
    if left_warn:
        collected_warnings.append(left_warn)
    if right_warn:
        collected_warnings.append(right_warn)

    # --- Duration check ---
    if nothing_annotations is not None:
        epoch_duration_ok, dur_warn = _check_epoch_durations(
            nothing_annotations, expected_duration_sec, duration_tolerance_sec,
        )
        if dur_warn:
            collected_warnings.append(dur_warn)
    else:
        epoch_duration_ok = True

    # --- EEG presence ---
    eeg_presence_ok, eeg_warn = _check_condition_presence(
        eeg_epochs, "EEG",
    )
    if eeg_warn:
        collected_warnings.append(eeg_warn)

    # --- fNIRS presence (skipped when data unavailable) ---
    fnirs_presence_ok, fnirs_warn = _check_condition_presence(
        fnirs_epochs, "fNIRS",
    )
    if fnirs_warn:
        collected_warnings.append(fnirs_warn)

    # --- Logging ---
    for warning_text in collected_warnings:
        logger.warning("NOTHING validation: %s", warning_text)

    result = NothingValidationResult(
        epoch_count_ok=epoch_count_ok,
        epoch_duration_ok=epoch_duration_ok,
        eeg_presence_ok=eeg_presence_ok,
        fnirs_presence_ok=fnirs_presence_ok,
        n_nothing_epochs=n_nothing,
        n_left_epochs=n_left,
        n_right_epochs=n_right,
        warnings=collected_warnings,
    )

    if result.all_passed:
        logger.info(
            "NOTHING validation passed: %d NOTHING, %d LEFT, %d RIGHT "
            "epochs — condition integrity confirmed",
            n_nothing,
            n_left,
            n_right,
        )
    else:
        logger.warning(
            "NOTHING validation completed with %d warning(s)",
            len(collected_warnings),
        )

    return result
