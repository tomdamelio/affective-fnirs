"""
Test script for EEG analysis module.

This script verifies that the EEG analysis functions can be imported
and have the correct signatures.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from affective_fnirs.eeg_analysis import (
    compute_tfr,
    create_epochs,
    detect_erd_ers,
    plot_erd_timecourse,
    plot_eeg_spectrogram,
    select_motor_channel,
)


def test_imports():
    """Test that all functions can be imported."""
    print("✓ All EEG analysis functions imported successfully")


def test_function_signatures():
    """Test that functions have expected signatures."""
    import inspect

    # Check create_epochs
    sig = inspect.signature(create_epochs)
    assert "raw" in sig.parameters
    assert "event_id" in sig.parameters
    assert "tmin" in sig.parameters
    assert "tmax" in sig.parameters
    print("✓ create_epochs has correct signature")

    # Check compute_tfr
    sig = inspect.signature(compute_tfr)
    assert "epochs" in sig.parameters
    assert "freqs" in sig.parameters
    assert "n_cycles" in sig.parameters
    print("✓ compute_tfr has correct signature")

    # Check select_motor_channel
    sig = inspect.signature(select_motor_channel)
    assert "raw_or_epochs" in sig.parameters
    assert "primary_channel" in sig.parameters
    assert "fallback_channels" in sig.parameters
    print("✓ select_motor_channel has correct signature")

    # Check detect_erd_ers
    sig = inspect.signature(detect_erd_ers)
    assert "tfr" in sig.parameters
    assert "channel" in sig.parameters
    assert "alpha_band" in sig.parameters
    assert "beta_band" in sig.parameters
    print("✓ detect_erd_ers has correct signature")

    # Check plot_eeg_spectrogram
    sig = inspect.signature(plot_eeg_spectrogram)
    assert "tfr" in sig.parameters
    assert "channel" in sig.parameters
    assert "vmin" in sig.parameters
    assert "vmax" in sig.parameters
    print("✓ plot_eeg_spectrogram has correct signature")

    # Check plot_erd_timecourse
    sig = inspect.signature(plot_erd_timecourse)
    assert "tfr" in sig.parameters
    assert "channel" in sig.parameters
    assert "alpha_band" in sig.parameters
    assert "beta_band" in sig.parameters
    print("✓ plot_erd_timecourse has correct signature")


def test_docstrings():
    """Test that all functions have docstrings."""
    functions = [
        create_epochs,
        compute_tfr,
        select_motor_channel,
        detect_erd_ers,
        plot_eeg_spectrogram,
        plot_erd_timecourse,
    ]

    for func in functions:
        assert func.__doc__ is not None, f"{func.__name__} missing docstring"
        assert len(func.__doc__) > 100, f"{func.__name__} docstring too short"
        print(f"✓ {func.__name__} has comprehensive docstring")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing EEG Analysis Module")
    print("=" * 60)

    test_imports()
    print()

    test_function_signatures()
    print()

    test_docstrings()
    print()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

