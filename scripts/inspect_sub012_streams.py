
import sys
from pathlib import Path
import pyxdf

# Add src to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent / "src"))

from affective_fnirs.config import SubjectConfig

def inspect_streams():
    # Construct path based on run_analysis_sub012.py logic
    # Assuming config is standard
    try:
        config = SubjectConfig.from_yaml(Path("configs/sub-012.yml"))
    except Exception as e:
        print(f"Could not load config: {e}")
        return

    xdf_filename = f"sub-12_ses-{config.subject.session}_task-{config.subject.task}_recording.xdf"
    xdf_path = config.data_root / f"sub-{config.subject.id}" / f"ses-{config.subject.session}" / xdf_filename
    
    if not xdf_path.exists():
        xdf_path = config.data_root / f"sub-{config.subject.id}" / xdf_filename

    if not xdf_path.exists():
        print(f"XDF file not found at {xdf_path}")
        # Try finding ANY xdf in the folder
        xdf_dir = config.data_root / f"sub-{config.subject.id}"
        print(f"Searching in {xdf_dir}...")
        xdfs = list(xdf_dir.glob("**/*.xdf"))
        if xdfs:
            xdf_path = xdfs[0]
            print(f"Found alternative: {xdf_path}")
        else:
            return

    print(f"Loading {xdf_path}...")
    streams, header = pyxdf.load_xdf(str(xdf_path))
    
    print(f"\nFound {len(streams)} streams:")
    for i, stream in enumerate(streams):
        name = stream['info']['name'][0]
        type_ = stream['info']['type'][0]
        channel_count = int(stream['info']['channel_count'][0])
        nominal_srate = float(stream['info']['nominal_srate'][0])
        print(f"  {i+1}. Name: '{name}', Type: '{type_}', Channels: {channel_count}, Rate: {nominal_srate}")

if __name__ == "__main__":
    inspect_streams()
