import pyxdf
import numpy as np
import argparse
from termcolor import colored

def inspect_xdf(file_path):
    print(colored(f"Inspecting file: {file_path}", "cyan"))
    try:
        streams, header = pyxdf.load_xdf(file_path)
        print(f"Loaded {len(streams)} streams.")
    except Exception as e:
        print(colored(f"Error loading XDF: {e}", "red"))
        return

    for i, stream in enumerate(streams):
        info = stream['info']
        name = info['name'][0]
        stype = info['type'][0]
        channel_count = info['channel_count'][0]
        print(f"\nStream {i}: {colored(name, 'yellow')} (Type: {stype}, Channels: {channel_count})")
        
        # Check for markers
        data = stream['time_series']
        if len(data) == 0:
            print("  (Empty stream)")
            continue
            
        if isinstance(data[0], (str, np.str_, list)) or stype == 'Markers':
            unique_markers = np.unique(data)
            print(f"  Sample markers ({len(data)} total): {data[:5]}")
            print(f"  Unique markers: {unique_markers}")
            
            # Check for keywords
            for expected in ['LEFT', 'RIGHT', 'NOTHING']:
                found = any(expected in str(m) for m in unique_markers)
                if found:
                    print(colored(f"  -> Found '{expected}' in markers!", "green"))
        else:
             print(f"  Data shape: {np.array(data).shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect XDF streams for markers.")
    parser.add_argument("file_path", help="Path to XDF file")
    args = parser.parse_args()
    inspect_xdf(args.file_path)
