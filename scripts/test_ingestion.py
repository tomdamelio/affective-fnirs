"""
Quick verification script for data ingestion module.

This script tests the ingestion module with actual pilot data to ensure
all functions work correctly before proceeding to MNE object construction.
"""

from pathlib import Path

from affective_fnirs.ingestion import (
    DataIngestionError,
    extract_stream_data,
    identify_streams,
    load_xdf_file,
)


def main():
    """Test ingestion module with pilot data."""
    print("=" * 70)
    print("Testing Data Ingestion Module")
    print("=" * 70)

    # Test file path - using sub-002 which should have complete data
    xdf_path = Path(
        "data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf"
    )

    # Test 1: Load XDF file
    print("\n[1/3] Testing load_xdf_file()...")
    try:
        streams, header = load_xdf_file(xdf_path)
        print(f"✓ Loaded {len(streams)} streams from XDF file")
        for i, stream in enumerate(streams):
            stream_name = stream["info"]["name"][0]
            stream_type = stream["info"].get("type", ["Unknown"])[0]
            ts = stream["time_series"]
            if hasattr(ts, 'shape'):
                data_shape = ts.shape if ts.size > 0 else "empty"
            else:
                data_shape = f"list({len(ts)})" if ts else "empty"
            print(f"  Stream {i}: {stream_name} (type: {stream_type}, data: {data_shape})")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return

    # Test 2: Identify streams
    print("\n[2/3] Testing identify_streams()...")
    try:
        identified = identify_streams(streams)
        print(f"✓ Identified {len(identified)} required streams:")
        for stream_type, stream in identified.items():
            stream_name = stream["info"]["name"][0]
            ts = stream["time_series"]
            if hasattr(ts, 'shape'):
                data_shape = ts.shape if ts.size > 0 else "empty"
            else:
                data_shape = f"list({len(ts)})" if ts else "empty"
            print(f"  {stream_type.upper()}: {stream_name} (data: {data_shape})")
    except DataIngestionError as e:
        print(f"✗ Failed: {e}")
        return

    # Test 3: Extract stream data
    print("\n[3/3] Testing extract_stream_data()...")
    for stream_type, stream in identified.items():
        try:
            data, sfreq, timestamps = extract_stream_data(stream)
            stream_name = stream["info"]["name"][0]
            print(
                f"✓ {stream_type.upper()} ({stream_name}):"
            )
            print(f"    Shape: {data.shape}")
            print(f"    Sampling rate: {sfreq} Hz")
            print(f"    Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
            print(f"    Timestamp range: [{timestamps[0]:.3f}, {timestamps[-1]:.3f}]")
        except Exception as e:
            print(f"✗ Failed for {stream_type}: {e}")
            return

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
