
import pyxdf
import numpy as np
try:
    streams, header = pyxdf.load_xdf('data/raw/sub-011/ses-001/sub-011_ses-001_task-fingertapping_recording.xdf')
    for s in streams:
        if 'time_series' in s:
            data = s['time_series']
            if len(data) == 0: continue
            flat = np.array(data).flatten()
            # check a subset
            strs = [str(x) for x in flat[:1000]] 
            if any('LEFT' in x for x in strs):
                print(f"FOUND_STREAM_NAME: {s['info']['name'][0]}")
except Exception as e:
    print(e)
