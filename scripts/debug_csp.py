
import sys
import logging
import traceback
from pathlib import Path
import mne
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('debug_csp')

# Add src to path
# Use absolute path to avoid ambiguity
src_path = Path("c:/Users/tdamelio/Desktop/fnirs/affective-fnirs/src").resolve()
sys.path.append(str(src_path))
logger.info(f"Added to sys.path: {src_path}")

# Import from run_analysis
import run_analysis
from affective_fnirs.config import SubjectConfig

def main():
    try:
        config_path = "configs/sub-011.yml"
        logger.info(f"Loading config from {config_path}")
        config = SubjectConfig.from_yaml(config_path)

        output_path = Path("data/derivatives/validation-pipeline/sub-011/ses-001")
        epochs_path = output_path / "sub-011_ses-001_task-fingertapping_desc-cleaned_epo.fif"
        
        logger.info(f"Loading epochs from {epochs_path}")
        epochs = mne.read_epochs(epochs_path, preload=True)
        
        logger.info("Calling generate_csp_analysis...")
        # We need to make sure we are calling the one in run_analysis module
        path, results = run_analysis.generate_csp_analysis(epochs, output_path, config)
        
        if path:
            logger.info(f"Success! Saved to {path}")
            logger.info(f"Results: {results}")
        else:
            logger.error("Failed! Returned None.")

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
