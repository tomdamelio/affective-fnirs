
import pyxdf
import numpy as np
import logging

def check_file(path, subj):
    print(f"\n--- Checking {subj} ---")
    print(f"File: {path}")
    try:
        streams, header = pyxdf.load_xdf(path)
        
        # Collect marker streams
        candidates = []
        for s in streams:
            try:
                name = s['info']['name'][0]
                stype = s['info']['type'][0]
                if stype == 'Markers' or 'marker' in name.lower() or 'event' in name.lower():
                    candidates.append(s)
            except: pass
            
        print(f"Found {len(candidates)} marker candidates.")
        
        for c in candidates:
            name = c['info']['name'][0]
            data = c.get('time_series', [])
            
            # Check content
            events_found = []
            if len(data) > 0:
                flat = np.array(data).flatten()
                 # check first 5000
                sample = [str(x) for x in flat[:5000]]
                
                if any('LEFT' in s for s in sample): events_found.append('LEFT')
                if any('RIGHT' in s for s in sample): events_found.append('RIGHT')
                if any('NOTHING' in s for s in sample): events_found.append('NOTHING')
            
            if events_found:
                print(f"  [MATCH] Stream: '{name}' -> Events: {events_found}")
            else:
                print(f"  [EMPTY] Stream: '{name}' -> No events or no match")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    files = [
        ("sub-009", "data/raw/sub-009/ses-001/sub-009_ses-001_task-fingertapping_recording.xdf"),
        ("sub-010", "data/raw/sub-010/ses-001/sub-010_ses-001_task-fingertapping_recording.xdf"),
        ("sub-011", "data/raw/sub-011/ses-001/sub-011_ses-001_task-fingertapping_recording.xdf"),
    ]
    for subj, path in files:
        check_file(path, subj)
