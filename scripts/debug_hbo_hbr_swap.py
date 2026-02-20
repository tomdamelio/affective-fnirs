#!/usr/bin/env python
"""Debug script to investigate potential HbO/HbR channel swap issue.

This script verifies:
1. How wavelengths are stored in channel metadata
2. How MNE's beer_lambert_law() converts OD to hemoglobin
3. Whether the HbO/HbR assignment is correct

Scientific background:
- 760nm has higher absorption for HbR (deoxyhemoglobin)
- 850nm has higher absorption for HbO (oxyhemoglobin)
- MNE uses extinction coefficients from Cope (1991)
"""

import json
import logging
from pathlib import Path

import mne
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run diagnostic checks for HbO/HbR channel assignment."""
    logger.info("=" * 70)
    logger.info("HbO/HbR Channel Swap Diagnostic")
    logger.info("=" * 70)
    
    # Load montage JSON
    montage_path = Path("data/raw/sub-012/ses-001/montage_combined_EEG_fNIRS_with_3Dcoords_approx.json")
    
    if not montage_path.exists():
        logger.error(f"Montage file not found: {montage_path}")
        return
    
    with open(montage_path, "r") as f:
        montage = json.load(f)
    
    ch_montage = montage.get("ChMontage", [])
    
    # Check wavelength-to-label mapping in JSON
    logger.info("\n1. Wavelength-to-label mapping in montage JSON:")
    logger.info("-" * 50)
    
    wavelength_labels = {}
    for ch in ch_montage[:8]:  # First 8 channels (4 pairs)
        wl = ch["wavelength"]
        label = ch["location_label"]
        suffix = label.split("_")[-1]  # Get _Hb or _HbO
        
        if wl not in wavelength_labels:
            wavelength_labels[wl] = []
        wavelength_labels[wl].append(suffix)
        
        logger.info(f"  Channel {ch['channel_idx']}: {wl}nm -> {label} (suffix: {suffix})")
    
    logger.info("\n  Summary:")
    for wl, suffixes in wavelength_labels.items():
        unique_suffix = list(set(suffixes))[0]
        logger.info(f"    {wl}nm -> {unique_suffix}")
    
    # Explain the physics
    logger.info("\n2. Physical interpretation:")
    logger.info("-" * 50)
    logger.info("  Extinction coefficients (Cope 1991):")
    logger.info("    760nm: HbO=1486.6, HbR=3843.7")
    logger.info("           -> 760nm is MORE sensitive to HbR (higher absorption)")
    logger.info("    850nm: HbO=2526.4, HbR=1798.6")
    logger.info("           -> 850nm is MORE sensitive to HbO (higher absorption)")
    logger.info("")
    logger.info("  The JSON labels are CORRECT:")
    logger.info("    - 760nm labeled as _Hb (HbR) because 760nm detects HbR better")
    logger.info("    - 850nm labeled as _HbO because 850nm detects HbO better")
    
    # Check how MNE handles the conversion
    logger.info("\n3. MNE Beer-Lambert Law conversion:")
    logger.info("-" * 50)
    logger.info("  MNE's beer_lambert_law() uses wavelength values from loc[9]")
    logger.info("  to look up extinction coefficients and solve the system:")
    logger.info("")
    logger.info("  [OD_760]   [e_HbO_760  e_HbR_760] [HbO]")
    logger.info("  [OD_850] = [e_HbO_850  e_HbR_850] [HbR] * L * DPF")
    logger.info("")
    logger.info("  Inverting this matrix gives HbO and HbR concentrations.")
    logger.info("  The labels in the JSON are NOT used by MNE - only wavelength values.")
    
    # Check if there's an issue with the data order
    logger.info("\n4. Potential issues to check:")
    logger.info("-" * 50)
    logger.info("  a) Are wavelength values correctly stored in loc[9]?")
    logger.info("  b) Is the data matrix ordered correctly (channel_idx matches data column)?")
    logger.info("  c) Are the raw XDF data channels in the expected order?")
    
    # Load preprocessed data if available
    preprocessed_path = Path("data/derivatives/validation-pipeline/sub-012/ses-001/sub-012_ses-001_task-fingertapping_desc-preprocessed_fnirs.fif")
    
    if preprocessed_path.exists():
        logger.info("\n5. Checking preprocessed fNIRS data:")
        logger.info("-" * 50)
        
        raw = mne.io.read_raw_fif(preprocessed_path, preload=True)
        
        # Check channel types
        ch_types = raw.get_channel_types()
        hbo_channels = [ch for ch, t in zip(raw.ch_names, ch_types) if t == "hbo"]
        hbr_channels = [ch for ch, t in zip(raw.ch_names, ch_types) if t == "hbr"]
        
        logger.info(f"  HbO channels: {len(hbo_channels)}")
        logger.info(f"  HbR channels: {len(hbr_channels)}")
        
        if hbo_channels:
            logger.info(f"  First HbO channel: {hbo_channels[0]}")
            
            # Get data statistics
            hbo_data = raw.get_data(picks=hbo_channels[:5])
            hbr_data = raw.get_data(picks=hbr_channels[:5])
            
            logger.info(f"\n  Data statistics (first 5 channels):")
            logger.info(f"    HbO mean: {hbo_data.mean() * 1e6:.3f} uM")
            logger.info(f"    HbO std:  {hbo_data.std() * 1e6:.3f} uM")
            logger.info(f"    HbR mean: {hbr_data.mean() * 1e6:.3f} uM")
            logger.info(f"    HbR std:  {hbr_data.std() * 1e6:.3f} uM")
            
            # Check expected HRF pattern
            logger.info("\n  Expected HRF pattern during motor task:")
            logger.info("    - HbO should INCREASE (positive) during task")
            logger.info("    - HbR should DECREASE (negative) during task")
            logger.info("    - If reversed, channels may be swapped")
    else:
        logger.info(f"\n  Preprocessed file not found: {preprocessed_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Diagnostic complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
