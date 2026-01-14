"""
Check EEG hardware reference from XDF and JSON metadata.

This script inspects the original hardware reference used during recording.
"""

import sys
from pathlib import Path
import pyxdf
import json

from affective_fnirs.ingestion import identify_streams


def check_reference(subject: str) -> None:
    """Check EEG reference for a subject."""
    xdf_file = Path(f"data/raw/{subject}/{subject}_ses-001_task-fingertapping_recording.xdf")
    eeg_json = Path(f"data/raw/{subject}/{subject}_ses-001_task-fingertapping_eeg.json")
    
    if not xdf_file.exists():
        print(f"ERROR: XDF file not found: {xdf_file}")
        return
    if not eeg_json.exists():
        print(f"ERROR: EEG JSON not found: {eeg_json}")
        return
    
    print("=" * 70)
    print(f"EEG REFERENCE CHECK - {subject.upper()}")
    print("=" * 70)
    
    # Load XDF and get EEG stream
    print("\n[1] Loading XDF file...")
    streams, _ = pyxdf.load_xdf(str(xdf_file))
    stream_info = identify_streams(streams)
    eeg_stream = stream_info["eeg"]
    
    # Check XDF stream info for reference
    print("\n[2] Checking XDF stream metadata...")
    stream_name = eeg_stream["info"]["name"][0]
    print(f"Stream name: {stream_name}")
    
    # Look for reference information in stream info
    if "desc" in eeg_stream["info"]:
        desc = eeg_stream["info"]["desc"]
        if desc and len(desc) > 0:
            desc_dict = desc[0]
            print(f"\nStream description keys: {list(desc_dict.keys())}")
            
            # Check for reference field
            if "reference" in desc_dict:
                ref_info = desc_dict["reference"]
                print(f"Reference field found: {ref_info}")
            else:
                print("No 'reference' field in stream description")
            
            # Check channels for reference info
            if "channels" in desc_dict:
                channels = desc_dict["channels"]
                if channels and len(channels) > 0:
                    channel_list = channels[0].get("channel", [])
                    print(f"\nTotal channels in stream: {len(channel_list)}")
                    
                    # Look for reference channel
                    ref_channels = []
                    for ch in channel_list:
                        ch_label = ch.get("label", [""])[0]
                        ch_type = ch.get("type", [""])[0]
                        if "ref" in ch_label.lower() or ch_type.lower() == "ref":
                            ref_channels.append(ch_label)
                    
                    if ref_channels:
                        print(f"Reference channels found: {ref_channels}")
                    else:
                        print("No explicit reference channels found in channel list")
    
    # Load JSON metadata
    print("\n[3] Checking JSON sidecar metadata...")
    with open(eeg_json) as f:
        eeg_meta = json.load(f)
    
    print(f"JSON keys: {list(eeg_meta.keys())}")
    
    # Check for reference information
    if "EEGReference" in eeg_meta:
        print(f"\nEEGReference: {eeg_meta['EEGReference']}")
    else:
        print("\nNo 'EEGReference' field in JSON")
    
    if "EEGGround" in eeg_meta:
        print(f"EEGGround: {eeg_meta['EEGGround']}")
    
    # Check channel names
    if "channels" in eeg_meta:
        channels = eeg_meta["channels"]
        print(f"\nChannels in JSON: {len(channels)}")
        print("Channel names:")
        for ch in channels:
            ch_name = ch.get("name", "")
            ch_type = ch.get("type", "")
            print(f"  - {ch_name} ({ch_type})")
    
    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Hardware reference is typically set during recording and depends on:
1. Amplifier configuration (e.g., actiCHamp settings)
2. Physical electrode placement
3. Recording software settings

Common configurations:
- Cz reference: One electrode at Cz position used as reference
- Linked mastoids: Average of electrodes behind ears (M1, M2)
- Average reference: Computed average of all electrodes (software)

If no explicit reference is documented, check:
1. Amplifier manual/settings during recording
2. Lab recording protocol
3. Physical electrode montage photos

MNE-Python assumes data is already referenced by hardware.
Our pipeline applies additional software re-referencing to Cz.
""")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python check_eeg_reference.py <subject>")
        print("Example: python check_eeg_reference.py sub-006")
        return 1
    
    subject = sys.argv[1]
    check_reference(subject)
    return 0


if __name__ == "__main__":
    sys.exit(main())
