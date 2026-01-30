
import pyxdf
import numpy as np
import json
import os

def count_markers(path):
    try:
        if not os.path.exists(path): return {"error": "File not found"}
        streams, header = pyxdf.load_xdf(path)
        
        counts = {"LEFT": 0, "RIGHT": 0, "NOTHING": 0}
        target_stream_name = "eeg_markers"
        found_stream = None
        
        # specific search for eeg_markers
        for s in streams:
            try:
                name = s['info']['name'][0]
                if name == target_stream_name:
                    found_stream = s
                    break
            except: pass
        
        if not found_stream:
            # Fallback check if simple search failed (sometimes names have whitespace?)
            for s in streams:
                 try: 
                    if target_stream_name in s['info']['name'][0]:
                        found_stream = s
                        break
                 except: pass
        
        if not found_stream:
            return {"error": f"Stream '{target_stream_name}' not found"}

        data = found_stream.get('time_series', [])
        if len(data) > 0:
            flat = np.array(data).flatten()
            # Convert all to string once
            for x in flat:
                s = str(x)
                if 'LEFT' in s: counts["LEFT"] += 1
                elif 'RIGHT' in s: counts["RIGHT"] += 1
                elif 'NOTHING' in s: counts["NOTHING"] += 1
        
        return counts
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    files = {
        "sub-009": "data/raw/sub-009/ses-001/sub-009_ses-001_task-fingertapping_recording.xdf",
        "sub-010": "data/raw/sub-010/ses-001/sub-010_ses-001_task-fingertapping_recording.xdf",
        "sub-011": "data/raw/sub-011/ses-001/sub-011_ses-001_task-fingertapping_recording.xdf",
    }
    
    results = {}
    for subj, path in files.items():
        results[subj] = count_markers(path)
        
    print(json.dumps(results, indent=2))
