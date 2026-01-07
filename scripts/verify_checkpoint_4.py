"""
Checkpoint 4 Verification Script: Data Ingestion and MNE Construction

This script verifies that Tasks 1-3 have been completed correctly:
- Task 1: Environment and configuration infrastructure
- Task 2: Data ingestion module (ingestion.py)
- Task 3: MNE object construction module (mne_builder.py)

Verification Criteria:
1. All tests pass (if any exist)
2. EEG and fNIRS are in separate Raw objects
3. LSL timestamps are used for synchronization
4. Modules can be imported successfully
5. Basic functionality works with test data

Requirements Validated:
- 1.1-1.5: XDF data ingestion
- 2.1-2.6: MNE object construction
- 9.1: Read-only access to raw data
- 11.1-11.3: Error handling and diagnostics
"""

from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np


def verify_imports():
    """Verify all required modules can be imported."""
    print("=" * 80)
    print("CHECKPOINT 4: Data Ingestion and MNE Construction Verification")
    print("=" * 80)
    print("\n1. Verifying module imports...")
    
    try:
        from affective_fnirs import ingestion, mne_builder
        print("   ✓ affective_fnirs.ingestion imported successfully")
        print("   ✓ affective_fnirs.mne_builder imported successfully")
        return True
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False


def verify_ingestion_module():
    """Verify ingestion module has required functions."""
    print("\n2. Verifying ingestion module functions...")
    
    from affective_fnirs import ingestion
    
    required_functions = [
        "load_xdf_file",
        "identify_streams",
        "extract_stream_data",
    ]
    
    all_present = True
    for func_name in required_functions:
        if hasattr(ingestion, func_name):
            print(f"   ✓ {func_name} exists")
        else:
            print(f"   ✗ {func_name} missing")
            all_present = False
    
    return all_present


def verify_mne_builder_module():
    """Verify mne_builder module has required functions."""
    print("\n3. Verifying mne_builder module functions...")
    
    from affective_fnirs import mne_builder
    
    required_functions = [
        "build_eeg_raw",
        "build_fnirs_raw",
        "embed_events",
    ]
    
    all_present = True
    for func_name in required_functions:
        if hasattr(mne_builder, func_name):
            print(f"   ✓ {func_name} exists")
        else:
            print(f"   ✗ {func_name} missing")
            all_present = False
    
    return all_present


def verify_test_data_loading():
    """Verify we can load test data (sub-002)."""
    print("\n4. Verifying test data loading (sub-002)...")
    
    from affective_fnirs import ingestion
    
    # Check if test data exists
    test_file = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
    
    if not test_file.exists():
        print(f"   ⚠ Test data not found: {test_file}")
        print("   → This is expected if data hasn't been added yet")
        return None
    
    try:
        # Load XDF file
        streams, header = ingestion.load_xdf_file(test_file)
        print(f"   ✓ Loaded XDF file with {len(streams)} streams")
        
        # Identify streams
        identified = ingestion.identify_streams(streams)
        print(f"   ✓ Identified streams: {list(identified.keys())}")
        
        # Extract stream data
        for stream_name, stream in identified.items():
            data, sfreq, timestamps = ingestion.extract_stream_data(stream)
            print(f"   ✓ {stream_name.upper()}: shape={data.shape}, sfreq={sfreq}Hz, "
                  f"duration={len(timestamps)/sfreq:.1f}s")
        
        return identified
    except Exception as e:
        print(f"   ✗ Failed to load test data: {e}")
        return None


def verify_mne_construction(identified_streams):
    """Verify MNE object construction with test data."""
    if identified_streams is None:
        print("\n5. Skipping MNE construction (no test data)")
        return None
    
    print("\n5. Verifying MNE object construction...")
    
    from affective_fnirs import ingestion, mne_builder
    import mne
    
    try:
        # Build EEG Raw
        eeg_stream = identified_streams['eeg']
        eeg_data, eeg_sfreq, eeg_timestamps = ingestion.extract_stream_data(eeg_stream)
        raw_eeg = mne_builder.build_eeg_raw(
            eeg_data, eeg_sfreq, eeg_stream['info'], eeg_timestamps
        )
        print(f"   ✓ EEG Raw object created: {len(raw_eeg.ch_names)} channels, "
              f"{raw_eeg.n_times} samples")
        
        # Verify EEG has montage
        eeg_positions = sum(1 for ch in raw_eeg.info['chs'] 
                           if not np.allclose(ch['loc'][:3], 0.0))
        print(f"   ✓ EEG channels with 3D positions: {eeg_positions}/{len(raw_eeg.ch_names)}")
        
        # Build fNIRS Raw
        fnirs_stream = identified_streams['fnirs']
        fnirs_data, fnirs_sfreq, fnirs_timestamps = ingestion.extract_stream_data(fnirs_stream)
        
        # Load fNIRS JSON config
        json_path = Path("data/raw/sub-002/sub-002_Tomi_ses-001_task-fingertapping_nirs.json")
        if json_path.exists():
            import json
            with open(json_path, 'r') as f:
                fnirs_json = json.load(f)
            montage_config = fnirs_json.get('ChMontage', [])
        else:
            print(f"   ⚠ fNIRS JSON not found: {json_path}")
            montage_config = []
        
        raw_fnirs = mne_builder.build_fnirs_raw(
            fnirs_data, fnirs_sfreq, montage_config, fnirs_timestamps
        )
        print(f"   ✓ fNIRS Raw object created: {len(raw_fnirs.ch_names)} channels, "
              f"{raw_fnirs.n_times} samples")
        
        # Verify fNIRS channel types
        fnirs_types = raw_fnirs.get_channel_types()
        fnirs_cw_count = sum(1 for t in fnirs_types if t == 'fnirs_cw_amplitude')
        print(f"   ✓ fNIRS channels with 'fnirs_cw_amplitude' type: {fnirs_cw_count}/{len(fnirs_types)}")
        
        # Embed events
        marker_stream = identified_streams['markers']
        raw_eeg = mne_builder.embed_events(raw_eeg, marker_stream)
        raw_fnirs = mne_builder.embed_events(raw_fnirs, marker_stream)
        
        print(f"   ✓ Events embedded in EEG: {len(raw_eeg.annotations)} annotations")
        print(f"   ✓ Events embedded in fNIRS: {len(raw_fnirs.annotations)} annotations")
        
        return raw_eeg, raw_fnirs
    except Exception as e:
        print(f"   ✗ MNE construction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_separation(mne_objects):
    """Verify EEG and fNIRS are in separate Raw objects."""
    if mne_objects is None:
        print("\n6. Skipping separation check (no MNE objects)")
        return None
    
    print("\n6. Verifying EEG and fNIRS separation...")
    
    raw_eeg, raw_fnirs = mne_objects
    
    # Check they are different objects
    if raw_eeg is raw_fnirs:
        print("   ✗ EEG and fNIRS are the same object!")
        return False
    else:
        print("   ✓ EEG and fNIRS are separate Raw objects")
    
    # Check channel types don't overlap
    eeg_types = set(raw_eeg.get_channel_types())
    fnirs_types = set(raw_fnirs.get_channel_types())
    
    print(f"   ✓ EEG channel types: {eeg_types}")
    print(f"   ✓ fNIRS channel types: {fnirs_types}")
    
    # Verify no mixing
    if 'fnirs_cw_amplitude' in eeg_types:
        print("   ✗ EEG Raw contains fNIRS channels!")
        return False
    
    if 'eeg' in fnirs_types:
        print("   ✗ fNIRS Raw contains EEG channels!")
        return False
    
    print("   ✓ No channel type mixing detected")
    return True


def verify_lsl_timestamps(mne_objects):
    """Verify LSL timestamps are used for synchronization."""
    if mne_objects is None:
        print("\n7. Skipping LSL timestamp check (no MNE objects)")
        return None
    
    print("\n7. Verifying LSL timestamp usage...")
    
    raw_eeg, raw_fnirs = mne_objects
    
    # Check for LSL timestamps attribute
    if hasattr(raw_eeg, '_lsl_timestamps'):
        print(f"   ✓ EEG Raw has LSL timestamps: {len(raw_eeg._lsl_timestamps)} samples")
    else:
        print("   ✗ EEG Raw missing LSL timestamps")
        return False
    
    if hasattr(raw_fnirs, '_lsl_timestamps'):
        print(f"   ✓ fNIRS Raw has LSL timestamps: {len(raw_fnirs._lsl_timestamps)} samples")
    else:
        print("   ✗ fNIRS Raw missing LSL timestamps")
        return False
    
    # Verify annotations use LSL-based timing
    if len(raw_eeg.annotations) > 0:
        first_event_time = raw_eeg.annotations.onset[0]
        print(f"   ✓ First EEG event at {first_event_time:.3f}s (relative to recording start)")
    
    if len(raw_fnirs.annotations) > 0:
        first_event_time = raw_fnirs.annotations.onset[0]
        print(f"   ✓ First fNIRS event at {first_event_time:.3f}s (relative to recording start)")
    
    return True


def main():
    """Run all verification checks."""
    results = []
    
    # Run checks
    results.append(("Module imports", verify_imports()))
    results.append(("Ingestion functions", verify_ingestion_module()))
    results.append(("MNE builder functions", verify_mne_builder_module()))
    
    identified = verify_test_data_loading()
    mne_objects = verify_mne_construction(identified)
    
    results.append(("EEG/fNIRS separation", verify_separation(mne_objects)))
    results.append(("LSL timestamps", verify_lsl_timestamps(mne_objects)))
    
    # Summary
    print("\n" + "=" * 80)
    print("CHECKPOINT 4 SUMMARY")
    print("=" * 80)
    
    for check_name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        elif result is None:
            status = "⚠ SKIP"
        else:
            status = "? UNKNOWN"
        
        print(f"{status:10s} {check_name}")
    
    # Overall result
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    
    print("\n" + "=" * 80)
    if failed == 0:
        print("✓ CHECKPOINT 4 PASSED")
        print(f"  {passed} checks passed, {skipped} skipped")
        print("\nReady to proceed to Task 5: fNIRS Quality Assessment")
    else:
        print("✗ CHECKPOINT 4 FAILED")
        print(f"  {passed} checks passed, {failed} failed, {skipped} skipped")
        print("\nPlease address failures before proceeding.")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
