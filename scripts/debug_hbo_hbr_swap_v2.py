#!/usr/bin/env python
"""Debug script v2 - Check raw XDF data and wavelength assignment.

This script loads the raw XDF data and verifies:
1. Channel order in XDF matches montage JSON
2. Wavelength values are correctly assigned
3. Data polarity after Beer-Lambert conversion
"""

import json
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mne
import numpy as np
import pyxdf

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run diagnostic checks for HbO/HbR channel assignment."""
    logger.info("=" * 70)
    logger.info("HbO/HbR Channel Swap Diagnostic v2 - Raw XDF Analysis")
    logger.info("=" * 70)
    
    # Paths
    xdf_path = Path("data/raw/sub-012/ses-001/sub-12_ses-001_task-fingertapping_recording.xdf")
    montage_path = Path("data/raw/sub-012/ses-001/montage_combined_EEG_fNIRS_with_3Dcoords_approx.json")
    
    if not xdf_path.exists():
        logger.error(f"XDF file not found: {xdf_path}")
        return
    
    # Load montage
    with open(montage_path, "r") as f:
        montage = json.load(f)
    
    ch_montage = montage.get("ChMontage", [])
    
    # Load XDF
    logger.info("\n1. Loading XDF file...")
    logger.info("-" * 50)
    
    streams, header = pyxdf.load_xdf(str(xdf_path))
    
    # Find fNIRS stream
    fnirs_stream = None
    for stream in streams:
        stream_type = stream["info"]["type"][0].lower()
        if "nirs" in stream_type or "fnirs" in stream_type:
            fnirs_stream = stream
            break
    
    if fnirs_stream is None:
        logger.error("No fNIRS stream found in XDF")
        return
    
    fnirs_data = fnirs_stream["time_series"]
    fnirs_info = fnirs_stream["info"]
    
    logger.info(f"  fNIRS stream found: {fnirs_info['name'][0]}")
    logger.info(f"  Data shape: {fnirs_data.shape}")
    logger.info(f"  Sampling rate: {fnirs_info['nominal_srate'][0]} Hz")
    
    # Get channel names from XDF
    logger.info("\n2. Channel names in XDF stream:")
    logger.info("-" * 50)
    
    xdf_channels = []
    try:
        desc = fnirs_info["desc"][0]
        channels_elem = desc["channels"][0]
        stream_channels = channels_elem["channel"]
        
        for ch in stream_channels:
            if "label" in ch:
                xdf_channels.append(ch["label"][0])
    except (KeyError, IndexError, TypeError):
        logger.warning("Could not extract channel names from XDF")
    
    if xdf_channels:
        logger.info(f"  Total channels in XDF: {len(xdf_channels)}")
        logger.info(f"  First 10 channels: {xdf_channels[:10]}")
    
    # Compare with montage JSON
    logger.info("\n3. Comparing XDF channels with montage JSON:")
    logger.info("-" * 50)
    
    # Check if XDF channel names match montage location_label
    for i, ch_info in enumerate(ch_montage[:8]):
        json_label = ch_info["location_label"]
        json_wl = ch_info["wavelength"]
        json_idx = ch_info["channel_idx"]
        
        xdf_label = xdf_channels[json_idx] if json_idx < len(xdf_channels) else "N/A"
        
        match = "OK" if json_label == xdf_label else "MISMATCH"
        logger.info(f"  idx={json_idx}: JSON={json_label} ({json_wl}nm), XDF={xdf_label} [{match}]")
    
    # Check data statistics per wavelength
    logger.info("\n4. Data statistics by wavelength:")
    logger.info("-" * 50)
    
    # Group channels by wavelength
    wl_760_indices = [ch["channel_idx"] for ch in ch_montage if ch["wavelength"] == 760]
    wl_850_indices = [ch["channel_idx"] for ch in ch_montage if ch["wavelength"] == 850]
    
    if wl_760_indices and wl_850_indices:
        data_760 = fnirs_data[:, wl_760_indices]
        data_850 = fnirs_data[:, wl_850_indices]
        
        logger.info(f"  760nm channels ({len(wl_760_indices)}):")
        logger.info(f"    Mean: {data_760.mean():.4f}")
        logger.info(f"    Std:  {data_760.std():.4f}")
        logger.info(f"    Min:  {data_760.min():.4f}")
        logger.info(f"    Max:  {data_760.max():.4f}")
        
        logger.info(f"  850nm channels ({len(wl_850_indices)}):")
        logger.info(f"    Mean: {data_850.mean():.4f}")
        logger.info(f"    Std:  {data_850.std():.4f}")
        logger.info(f"    Min:  {data_850.min():.4f}")
        logger.info(f"    Max:  {data_850.max():.4f}")
    
    # Build MNE Raw and check wavelength storage
    logger.info("\n5. Building MNE Raw and checking wavelength storage:")
    logger.info("-" * 50)
    
    from affective_fnirs.mne_builder import build_fnirs_raw
    
    sfreq = float(fnirs_info["nominal_srate"][0])
    timestamps = fnirs_stream["time_stamps"]
    
    raw_fnirs = build_fnirs_raw(
        data=fnirs_data,
        sfreq=sfreq,
        montage_config=ch_montage,
        timestamps=timestamps,
    )
    
    logger.info(f"  MNE Raw created with {len(raw_fnirs.ch_names)} channels")
    
    # Check wavelength values in loc[9]
    logger.info("\n  Wavelength values stored in loc[9]:")
    for i in range(min(8, len(raw_fnirs.ch_names))):
        ch_name = raw_fnirs.ch_names[i]
        wl_stored = raw_fnirs.info["chs"][i]["loc"][9]
        logger.info(f"    {ch_name}: loc[9] = {wl_stored}")
    
    # Convert to OD and then to hemoglobin
    logger.info("\n6. Converting to hemoglobin and checking polarity:")
    logger.info("-" * 50)
    
    # Convert to OD
    raw_od = mne.preprocessing.nirs.optical_density(raw_fnirs)
    logger.info(f"  Converted to OD: {len(raw_od.ch_names)} channels")
    
    # Convert to hemoglobin
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
    logger.info(f"  Converted to hemoglobin: {len(raw_haemo.ch_names)} channels")
    
    # Check channel types
    ch_types = raw_haemo.get_channel_types()
    hbo_channels = [ch for ch, t in zip(raw_haemo.ch_names, ch_types) if t == "hbo"]
    hbr_channels = [ch for ch, t in zip(raw_haemo.ch_names, ch_types) if t == "hbr"]
    
    logger.info(f"  HbO channels: {len(hbo_channels)}")
    logger.info(f"  HbR channels: {len(hbr_channels)}")
    
    # Get data and check statistics
    if hbo_channels and hbr_channels:
        hbo_data = raw_haemo.get_data(picks=hbo_channels)
        hbr_data = raw_haemo.get_data(picks=hbr_channels)
        
        logger.info(f"\n  Data statistics (all channels):")
        logger.info(f"    HbO mean: {hbo_data.mean() * 1e6:.3f} uM")
        logger.info(f"    HbO std:  {hbo_data.std() * 1e6:.3f} uM")
        logger.info(f"    HbR mean: {hbr_data.mean() * 1e6:.3f} uM")
        logger.info(f"    HbR std:  {hbr_data.std() * 1e6:.3f} uM")
        
        # Check correlation between HbO and HbR
        # In typical fNIRS, HbO and HbR should be anti-correlated during task
        hbo_mean_ts = hbo_data.mean(axis=0)
        hbr_mean_ts = hbr_data.mean(axis=0)
        
        correlation = np.corrcoef(hbo_mean_ts, hbr_mean_ts)[0, 1]
        logger.info(f"\n  HbO-HbR correlation: {correlation:.3f}")
        logger.info(f"    Expected: negative (anti-correlated during task)")
        logger.info(f"    If positive: may indicate swap or systemic noise")
    
    logger.info("\n" + "=" * 70)
    logger.info("Diagnostic complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
