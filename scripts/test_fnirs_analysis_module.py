"""
Quick test script to verify fnirs_analysis module functions are importable.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from affective_fnirs.fnirs_analysis import (
        create_fnirs_epochs,
        identify_motor_roi_channel,
        extract_hrf,
        validate_hrf_temporal_dynamics,
        compute_hrf_quality_metrics,
        plot_hrf_curves,
        plot_hrf_spatial_map,
    )

    print("✓ All functions imported successfully")
    print("\nAvailable functions:")
    print("  - create_fnirs_epochs")
    print("  - identify_motor_roi_channel")
    print("  - extract_hrf")
    print("  - validate_hrf_temporal_dynamics")
    print("  - compute_hrf_quality_metrics")
    print("  - plot_hrf_curves")
    print("  - plot_hrf_spatial_map")
    print("\nModule verification complete!")

except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)
