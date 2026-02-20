#!/usr/bin/env python
"""Debug script v3 - Check loc array and distance calculation.

The issue: MNE's beer_lambert_law() produces all zeros.
This happens when source-detector distances are not properly set.
"""

import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mne
import numpy as np
import pyxdf

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Check loc array values and distance calculation."""
    logger.info("=" * 70)
    logger.info("HbO/HbR Debug v3 - Checking loc array and distances")
    logger.info("=" * 70)
    
    # Paths
    xdf_path = Path("data/raw/sub-012/ses-001/sub-12_ses-001_task-fingertapping_recording.xdf")
    montage_path = Path("data/raw/sub-012/ses-001/montage_combined_EEG_fNIRS_with_3Dcoords_approx.json")
    
    # Load montage
    with open(montage_path, "r") as f:
        montage = json.load(f)
    
    ch_montage = montage.get("ChMontage", [])
    
    # Load XDF
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    
    fnirs_stream = None
    for stream in streams:
        stream_type = stream["info"]["type"][0].lower()
        if "nirs" in stream_type or "fnirs" in stream_type:
            fnirs_stream = stream
            break
    
    fnirs_data = fnirs_stream["time_series"]
    fnirs_info = fnirs_stream["info"]
    sfreq = float(fnirs_info["nominal_srate"][0])
    timestamps = fnirs_stream["time_stamps"]
    
    # Build MNE Raw
    from affective_fnirs.mne_builder import build_fnirs_raw
    
    raw_fnirs = build_fnirs_raw(
        data=fnirs_data,
        sfreq=sfreq,
        montage_config=ch_montage,
        timestamps=timestamps,
    )
    
    logger.info("\n1. Checking loc array BEFORE applying montage:")
    logger.info("-" * 50)
    
    for i in range(4):
        ch_name = raw_fnirs.ch_names[i]
        loc = raw_fnirs.info["chs"][i]["loc"]
        logger.info(f"  {ch_name}:")
        logger.info(f"    loc[0:3] (channel pos): {loc[0:3]}")
        logger.info(f"    loc[3:6] (source pos):  {loc[3:6]}")
        logger.info(f"    loc[6:9] (detector pos): {loc[6:9]}")
        logger.info(f"    loc[9] (wavelength):    {loc[9]}")
        logger.info(f"    loc[10] (distance):     {loc[10]}")
    
    # Apply montage from JSON
    logger.info("\n2. Applying 3D montage from JSON...")
    logger.info("-" * 50)
    
    # Import the function from run_analysis_sub012
    sys.path.insert(0, str(Path(__file__).parent))
    from run_analysis_sub012 import apply_fnirs_montage_from_json
    
    raw_fnirs = apply_fnirs_montage_from_json(raw_fnirs, montage)
    
    logger.info("\n3. Checking loc array AFTER applying montage:")
    logger.info("-" * 50)
    
    for i in range(4):
        ch_name = raw_fnirs.ch_names[i]
        loc = raw_fnirs.info["chs"][i]["loc"]
        logger.info(f"  {ch_name}:")
        logger.info(f"    loc[0:3] (channel pos): {loc[0:3]}")
        logger.info(f"    loc[3:6] (source pos):  {loc[3:6]}")
        logger.info(f"    loc[6:9] (detector pos): {loc[6:9]}")
        logger.info(f"    loc[9] (wavelength):    {loc[9]}")
        logger.info(f"    loc[10] (distance):     {loc[10]}")
    
    # Check MNE's source_detector_distances
    logger.info("\n4. MNE source_detector_distances():")
    logger.info("-" * 50)
    
    distances = mne.preprocessing.nirs.source_detector_distances(raw_fnirs.info)
    logger.info(f"  Distances (first 8): {distances[:8]}")
    logger.info(f"  Min: {distances.min():.4f} m")
    logger.info(f"  Max: {distances.max():.4f} m")
    
    # Convert to OD
    logger.info("\n5. Converting to OD...")
    raw_od = mne.preprocessing.nirs.optical_density(raw_fnirs)
    
    # Check loc array after OD conversion
    logger.info("\n6. Checking loc array AFTER OD conversion:")
    logger.info("-" * 50)
    
    for i in range(4):
        ch_name = raw_od.ch_names[i]
        loc = raw_od.info["chs"][i]["loc"]
        logger.info(f"  {ch_name}:")
        logger.info(f"    loc[3:6] (source pos):  {loc[3:6]}")
        logger.info(f"    loc[6:9] (detector pos): {loc[6:9]}")
        logger.info(f"    loc[9] (wavelength):    {loc[9]}")
    
    # Check distances after OD
    distances_od = mne.preprocessing.nirs.source_detector_distances(raw_od.info)
    logger.info(f"\n  Distances after OD (first 8): {distances_od[:8]}")
    
    # Convert to hemoglobin
    logger.info("\n7. Converting to hemoglobin...")
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
    
    # Check data
    hbo_data = raw_haemo.get_data(picks="hbo")
    hbr_data = raw_haemo.get_data(picks="hbr")
    
    logger.info(f"\n8. Hemoglobin data statistics:")
    logger.info("-" * 50)
    logger.info(f"  HbO mean: {hbo_data.mean() * 1e6:.6f} uM")
    logger.info(f"  HbO std:  {hbo_data.std() * 1e6:.6f} uM")
    logger.info(f"  HbO max:  {hbo_data.max() * 1e6:.6f} uM")
    logger.info(f"  HbR mean: {hbr_data.mean() * 1e6:.6f} uM")
    logger.info(f"  HbR std:  {hbr_data.std() * 1e6:.6f} uM")
    
    if hbo_data.max() == 0:
        logger.warning("\n  WARNING: All hemoglobin values are zero!")
        logger.warning("  This indicates beer_lambert_law() failed.")
        logger.warning("  Possible causes:")
        logger.warning("    1. Source-detector distances are zero or invalid")
        logger.warning("    2. Wavelength values not properly set")
        logger.warning("    3. Channel pairing failed")
    
    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
