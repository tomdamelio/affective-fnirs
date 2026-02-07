
import sys
import logging
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('verify_mov_csp')

def main():
    logger.info("Starting verification of MOV vs NO MOV CSP implementation...")
    
    script_path = Path("scripts/run_analysis.py")
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return 1
        
    cmd = [
        "micromamba", "run", "-n", "affective-fnirs", 
        "python", str(script_path),
        "--config", "configs/sub-011.yml",
        "--load-epochs" 
    ]
    
    logger.info(f"Executing command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check logs for confirmation
        if "Generating CSP Analysis (MOV vs NO MOV)..." in result.stderr:
             logger.info("✓ Log confirmation: Function call found in stderr.")
        else:
             logger.warning("✗ Log confirmation: Function call NOT found in stderr.")
             
        if result.returncode != 0:
            logger.error(f"Analysis script failed with return code {result.returncode}")
            logger.error("StdErr tail:\n" + result.stderr[-500:])
            return 1
            
        logger.info("✓ Analysis script executed successfully.")
        
        output_dir = Path("data/derivatives/validation-pipeline/sub-011/ses-001")
        csp_file = output_dir / "sub-011_ses-001_task-fingertapping_desc-csp_mov_vs_rest.png"
        
        if csp_file.exists():
            logger.info(f"✓ CSP Plot found: {csp_file}")
            logger.info(f"  Size: {csp_file.stat().st_size} bytes")
        else:
            logger.error(f"✗ CSP Plot NOT found at: {csp_file}")
            return 1

        report_file = output_dir / "sub-011_ses-001_task-fingertapping_desc-validation_report.html"
        if report_file.exists():
            content = report_file.read_text(encoding='utf-8')
            if "1.0.3.5 Common Spatial Patterns (CSP): MOV vs NO MOV Discrimination" in content:
                logger.info("✓ HTML Report confirmation: Section 1.0.3.5 found.")
            else:
                logger.error("✗ HTML Report confirmation: Section 1.0.3.5 NOT found.")
                return 1
        else:
             logger.error(f"✗ Report file not found: {report_file}")
             return 1
             
        logger.info("\nSUCCESS: MOV vs NO MOV CSP analysis verified!")
        return 0
        
    except Exception as e:
        logger.error(f"Verification failed with exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
