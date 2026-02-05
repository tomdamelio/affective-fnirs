
import sys
import logging
from pathlib import Path
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('verify_fix')

def main():
    logger.info("Starting verification of ERP fix...")
    
    # Path to run_analysis.py
    script_path = Path("scripts/run_analysis.py")
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return 1
        
    # Command to run analysis with --load-epochs
    # This skips the heavy preprocessing and uses the existing cleaned epochs
    cmd = [
        "micromamba", "run", "-n", "affective-fnirs", 
        "python", str(script_path),
        "--config", "configs/sub-011.yml",
        "--load-epochs" 
    ]
    
    logger.info(f"Executing command: {' '.join(cmd)}")
    
    try:
        # Run the validation pipeline in load-epochs mode
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check standard output for the "Generating ERP Analysis..." log
        if "Generating ERP Analysis..." in result.stderr:
             logger.info("✓ Log confirmation: 'Generating ERP Analysis...' found in stderr.")
        else:
             logger.warning("✗ Log confirmation: 'Generating ERP Analysis...' NOT found in stderr (check logs below).")
             
        if result.returncode != 0:
            logger.error(f"Analysis script failed with return code {result.returncode}")
            logger.error("StdOut stub:\n" + result.stdout[-500:])
            logger.error("StdErr stub:\n" + result.stderr[-500:])
            return 1
            
        logger.info("✓ Analysis script executed successfully.")
        
        # Check if the ERP file exists in the expected location
        output_dir = Path("data/derivatives/validation-pipeline/sub-011/ses-001")
        erp_file = output_dir / "sub-011_ses-001_task-fingertapping_desc-erp_analysis.png"
        
        if erp_file.exists():
            logger.info(f"✓ ERP Plot created found: {erp_file}")
            logger.info(f"  Size: {erp_file.stat().st_size} bytes")
        else:
            logger.error(f"✗ ERP Plot NOT found at: {erp_file}")
            return 1

        # Check HTML report for the ERP section specific string
        report_file = output_dir / "sub-011_ses-001_task-fingertapping_desc-validation_report.html"
        if report_file.exists():
            content = report_file.read_text(encoding='utf-8')
            if "1.0.3.2 ERP Analysis (Evoked Potentials)" in content:
                logger.info("✓ HTML Report confirmation: ERP Analysis section found in report.")
            else:
                logger.error("✗ HTML Report confirmation: ERP Analysis section NOT found in report.")
                return 1
        else:
             logger.error(f"✗ Report file not found: {report_file}")
             return 1
             
        logger.info("\nSUCCESS: The ERP Analysis fix is verified!")
        return 0
        
    except Exception as e:
        logger.error(f"Verification failed with exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
