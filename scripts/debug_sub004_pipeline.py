"""Debug pipeline execution for sub-004."""

from pathlib import Path
import traceback

from affective_fnirs.config import PipelineConfig
from affective_fnirs.pipeline import run_validation_pipeline

xdf_file = Path("data/raw/sub-004/sub-004_ses-001_task-fingertapping_recording.xdf")
eeg_json = Path("data/raw/sub-004/sub-004_ses-001_task-fingertapping_eeg.json")
fnirs_json = Path("data/raw/sub-004/sub-004_ses-001_task-fingertapping_nirs.json")
output_dir = Path("data/derivatives/validation-pipeline/sub-004/ses-001")

config_path = Path("configs/sub004_optimized.yml")
print(f"Loading config from: {config_path}")
config = PipelineConfig.from_yaml(config_path)

print(f"XDF: {xdf_file} (exists: {xdf_file.exists()})")
print(f"EEG JSON: {eeg_json} (exists: {eeg_json.exists()})")
print(f"fNIRS JSON: {fnirs_json} (exists: {fnirs_json.exists()})")
print(f"Output dir: {output_dir}")
print()

try:
    print("Running pipeline...")
    results = run_validation_pipeline(xdf_file, eeg_json, fnirs_json, config, output_dir)
    print("Pipeline completed successfully!")
    print(f"Subject: {results.subject_id}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
