#!/usr/bin/env python
"""Debug script - Check Short Channel Regression effect.

The issue: HbO and HbR are moving in the same direction (both positive or both negative).
This indicates systemic noise contamination.

SCR should remove superficial/systemic signals, making HbO and HbR anti-correlated.
"""

import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import mne
import mne_nirs
import numpy as np
import pyxdf
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Compare HRF with and without SCR."""
    logger.info("=" * 70)
    logger.info("Short Channel Regression Effect Analysis")
    logger.info("=" * 70)
    
    # Paths
    xdf_path = Path("data/raw/sub-012/ses-001/sub-12_ses-001_task-fingertapping_recording.xdf")
    montage_path = Path("data/raw/sub-012/ses-001/montage_combined_EEG_fNIRS_with_3Dcoords_approx.json")
    
    # Load montage
    with open(montage_path, "r") as f:
        montage = json.load(f)
    
    ch_montage = montage.get("ChMontage", [])
    
    # Load XDF
    logger.info("\nLoading XDF data...")
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    
    fnirs_stream = None
    marker_stream = None
    
    for stream in streams:
        stream_type = stream["info"]["type"][0].lower()
        stream_name = stream["info"]["name"][0].lower()
        
        if "nirs" in stream_type or "fnirs" in stream_type:
            fnirs_stream = stream
        elif "marker" in stream_type or "marker" in stream_name:
            marker_stream = stream
    
    fnirs_data = fnirs_stream["time_series"]
    fnirs_info = fnirs_stream["info"]
    sfreq = float(fnirs_info["nominal_srate"][0])
    timestamps = fnirs_stream["time_stamps"]
    
    # Build MNE Raw
    from affective_fnirs.mne_builder import build_fnirs_raw, embed_events
    from run_analysis_sub012 import apply_fnirs_montage_from_json
    
    raw_fnirs = build_fnirs_raw(
        data=fnirs_data,
        sfreq=sfreq,
        montage_config=ch_montage,
        timestamps=timestamps,
    )
    
    # Apply montage
    raw_fnirs = apply_fnirs_montage_from_json(raw_fnirs, montage)
    
    # Embed events
    raw_fnirs = embed_events(raw_fnirs, marker_stream)
    
    # Identify short and long channels
    logger.info("\nIdentifying short and long channels...")
    
    # Short channels have source numbers >= 13 (S13, S14, S15, S16)
    short_channels = [ch for ch in raw_fnirs.ch_names if any(f"S{i}_" in ch for i in [13, 14, 15, 16])]
    long_channels = [ch for ch in raw_fnirs.ch_names if ch not in short_channels and raw_fnirs.get_channel_types([ch])[0] == "fnirs_cw_amplitude"]
    
    logger.info(f"  Short channels: {len(short_channels)}")
    logger.info(f"  Long channels: {len(long_channels)}")
    logger.info(f"  Short channel names: {short_channels}")
    
    # Check distances - only for fNIRS channels
    fnirs_ch_names = [ch for ch in raw_fnirs.ch_names if raw_fnirs.get_channel_types([ch])[0] == "fnirs_cw_amplitude"]
    raw_fnirs_only = raw_fnirs.copy().pick(fnirs_ch_names)
    
    distances = mne.preprocessing.nirs.source_detector_distances(raw_fnirs_only.info)
    short_mask = np.array([ch in short_channels for ch in fnirs_ch_names])
    
    if short_mask.any():
        short_distances = distances[short_mask]
        long_distances = distances[~short_mask & (distances > 0)]
        
        logger.info(f"\n  Short channel distances: {short_distances * 1000} mm")
        if len(long_distances) > 0:
            logger.info(f"  Long channel distances (mean): {long_distances.mean() * 1000:.1f} mm")
    
    # Convert to OD
    raw_od = mne.preprocessing.nirs.optical_density(raw_fnirs)
    
    # Apply TDDR
    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
    
    # Re-apply montage after TDDR (positions may be lost)
    raw_od = apply_fnirs_montage_from_json(raw_od, montage)
    
    # Process WITHOUT SCR
    logger.info("\n" + "-" * 50)
    logger.info("Processing WITHOUT Short Channel Regression...")
    logger.info("-" * 50)
    
    raw_haemo_no_scr = mne.preprocessing.nirs.beer_lambert_law(raw_od.copy(), ppf=6.0)
    raw_haemo_no_scr.filter(l_freq=0.01, h_freq=0.5, picks=["hbo", "hbr"])
    
    # Process WITH SCR
    logger.info("\n" + "-" * 50)
    logger.info("Processing WITH Short Channel Regression...")
    logger.info("-" * 50)
    
    # Check if MNE-NIRS can detect short channels
    is_short = mne.preprocessing.nirs.short_channels(raw_od.info, threshold=0.015)
    logger.info(f"  MNE detected {is_short.sum()} short channels (threshold=15mm)")
    
    if is_short.sum() > 0:
        try:
            raw_od_scr = mne_nirs.signal_enhancement.short_channel_regression(raw_od.copy())
            logger.info("  SCR applied successfully")
        except Exception as e:
            logger.warning(f"  SCR failed: {e}")
            raw_od_scr = raw_od.copy()
    else:
        logger.warning("  No short channels detected by MNE, skipping SCR")
        raw_od_scr = raw_od.copy()
    
    raw_haemo_scr = mne.preprocessing.nirs.beer_lambert_law(raw_od_scr, ppf=6.0)
    raw_haemo_scr.filter(l_freq=0.01, h_freq=0.5, picks=["hbo", "hbr"])
    
    # Compare HbO-HbR correlation
    logger.info("\n" + "=" * 70)
    logger.info("HbO-HbR Correlation Analysis")
    logger.info("=" * 70)
    
    for label, raw_haemo in [("WITHOUT SCR", raw_haemo_no_scr), ("WITH SCR", raw_haemo_scr)]:
        ch_types = raw_haemo.get_channel_types()
        hbo_channels = [ch for ch, t in zip(raw_haemo.ch_names, ch_types) if t == "hbo"]
        hbr_channels = [ch for ch, t in zip(raw_haemo.ch_names, ch_types) if t == "hbr"]
        
        # Get mean time series
        hbo_data = raw_haemo.get_data(picks=hbo_channels)
        hbr_data = raw_haemo.get_data(picks=hbr_channels)
        
        hbo_mean = hbo_data.mean(axis=0)
        hbr_mean = hbr_data.mean(axis=0)
        
        correlation = np.corrcoef(hbo_mean, hbr_mean)[0, 1]
        
        logger.info(f"\n{label}:")
        logger.info(f"  HbO-HbR correlation: {correlation:.3f}")
        logger.info(f"  Expected: negative (anti-correlated)")
        
        if correlation > 0:
            logger.warning(f"  WARNING: Positive correlation indicates systemic noise!")
    
    # Create epochs and compare HRF
    logger.info("\n" + "=" * 70)
    logger.info("HRF Comparison")
    logger.info("=" * 70)
    
    events, event_id = mne.events_from_annotations(raw_haemo_no_scr)
    
    tmin, tmax = -5, 25
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for col, (label, raw_haemo) in enumerate([("WITHOUT SCR", raw_haemo_no_scr), ("WITH SCR", raw_haemo_scr)]):
        # Create epochs for RIGHT hand (left hemisphere contralateral)
        right_events = [k for k in event_id.keys() if "RIGHT" in k.upper()]
        if not right_events:
            continue
        
        cond_event_id = {k: event_id[k] for k in right_events}
        
        epochs = mne.Epochs(
            raw_haemo,
            events,
            event_id=cond_event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=(-5, 0),
            preload=True,
        )
        
        times = epochs.times
        
        # Left hemisphere channels (contralateral to RIGHT hand)
        ch_types = epochs.info.get_channel_types()
        left_hbo = [ch for ch in epochs.ch_names if any(f"S{i}_" in ch for i in [1, 3, 5, 7, 9]) and "hbo" in ch]
        left_hbr = [ch for ch in epochs.ch_names if any(f"S{i}_" in ch for i in [1, 3, 5, 7, 9]) and "hbr" in ch]
        
        # Plot HbO
        ax = axes[0, col]
        if left_hbo:
            data = epochs.get_data(picks=left_hbo) * 1e6
            mean_data = data.mean(axis=(0, 1))
            std_data = data.mean(axis=1).std(axis=0)
            
            ax.plot(times, mean_data, 'r-', linewidth=2, label='HbO')
            ax.fill_between(times, mean_data - std_data, mean_data + std_data,
                           color='red', alpha=0.2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='green', linestyle='-', alpha=0.5)
        ax.axvspan(0, 15, color='yellow', alpha=0.1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration change (uM)')
        ax.set_title(f'HbO - {label}')
        ax.legend(loc='upper right')
        ax.set_xlim(tmin, tmax)
        
        # Plot HbR
        ax = axes[1, col]
        if left_hbr:
            data = epochs.get_data(picks=left_hbr) * 1e6
            mean_data = data.mean(axis=(0, 1))
            std_data = data.mean(axis=1).std(axis=0)
            
            ax.plot(times, mean_data, 'b-', linewidth=2, label='HbR')
            ax.fill_between(times, mean_data - std_data, mean_data + std_data,
                           color='blue', alpha=0.2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='green', linestyle='-', alpha=0.5)
        ax.axvspan(0, 15, color='yellow', alpha=0.1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration change (uM)')
        ax.set_title(f'HbR - {label}')
        ax.legend(loc='upper right')
        ax.set_xlim(tmin, tmax)
    
    plt.suptitle('RIGHT hand task - Left hemisphere (contralateral)\nExpected: HbO increases, HbR decreases',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path("data/derivatives/validation-pipeline/sub-012/ses-001")
    fig_path = output_path / "debug_scr_effect.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"\nSaved figure to: {fig_path}")
    
    plt.close()
    
    logger.info("\n" + "=" * 70)
    logger.info("Diagnostic complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
