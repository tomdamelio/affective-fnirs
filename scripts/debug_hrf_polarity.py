#!/usr/bin/env python
"""Debug script - Check HRF polarity during motor task.

Expected physiological response during finger tapping:
- HbO should INCREASE (positive change) in contralateral motor cortex
- HbR should DECREASE (negative change) in contralateral motor cortex

If we see the opposite pattern, channels may be swapped.
"""

import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import mne
import numpy as np
import pyxdf
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Analyze HRF polarity during motor task."""
    logger.info("=" * 70)
    logger.info("HRF Polarity Analysis - Motor Task")
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
    
    # Find streams
    fnirs_stream = None
    marker_stream = None
    
    for stream in streams:
        stream_type = stream["info"]["type"][0].lower()
        stream_name = stream["info"]["name"][0].lower()
        
        if "nirs" in stream_type or "fnirs" in stream_type:
            fnirs_stream = stream
        elif "marker" in stream_type or "marker" in stream_name:
            marker_stream = stream
    
    if fnirs_stream is None:
        logger.error("No fNIRS stream found")
        return
    
    if marker_stream is None:
        logger.error("No marker stream found")
        return
    
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
    
    logger.info(f"\nEvents found: {set(raw_fnirs.annotations.description)}")
    
    # Convert to OD
    raw_od = mne.preprocessing.nirs.optical_density(raw_fnirs)
    
    # Apply TDDR
    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
    
    # Convert to hemoglobin
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
    
    # Filter
    raw_haemo.filter(l_freq=0.01, h_freq=0.5, picks=["hbo", "hbr"])
    
    logger.info(f"\nHemoglobin channels: {len(raw_haemo.ch_names)}")
    
    # Get channel types
    ch_types = raw_haemo.get_channel_types()
    hbo_channels = [ch for ch, t in zip(raw_haemo.ch_names, ch_types) if t == "hbo"]
    hbr_channels = [ch for ch, t in zip(raw_haemo.ch_names, ch_types) if t == "hbr"]
    
    # Define hemispheres based on source numbers
    # Left hemisphere: S1, S3, S5, S7, S9 (odd numbers, left prefrontal/motor)
    # Right hemisphere: S2, S4, S6, S8, S10 (even numbers, right prefrontal/motor)
    left_hbo = [ch for ch in hbo_channels if any(f"S{i}_" in ch for i in [1, 3, 5, 7, 9])]
    right_hbo = [ch for ch in hbo_channels if any(f"S{i}_" in ch for i in [2, 4, 6, 8, 10])]
    left_hbr = [ch for ch in hbr_channels if any(f"S{i}_" in ch for i in [1, 3, 5, 7, 9])]
    right_hbr = [ch for ch in hbr_channels if any(f"S{i}_" in ch for i in [2, 4, 6, 8, 10])]
    
    logger.info(f"\nLeft hemisphere HbO channels: {len(left_hbo)}")
    logger.info(f"Right hemisphere HbO channels: {len(right_hbo)}")
    
    # Create epochs for each condition
    events, event_id = mne.events_from_annotations(raw_haemo)
    
    logger.info(f"\nEvent IDs: {event_id}")
    
    # Find LEFT and RIGHT conditions
    left_events = [k for k in event_id.keys() if "LEFT" in k.upper()]
    right_events = [k for k in event_id.keys() if "RIGHT" in k.upper()]
    
    logger.info(f"LEFT conditions: {left_events}")
    logger.info(f"RIGHT conditions: {right_events}")
    
    if not left_events and not right_events:
        logger.warning("No LEFT or RIGHT conditions found")
        return
    
    # Create epochs
    tmin, tmax = -5, 25  # 5s baseline, 25s post-stimulus
    
    results = {}
    
    for cond_name, cond_events in [("LEFT", left_events), ("RIGHT", right_events)]:
        if not cond_events:
            continue
        
        cond_event_id = {k: event_id[k] for k in cond_events}
        
        epochs = mne.Epochs(
            raw_haemo,
            events,
            event_id=cond_event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
        )
        
        if len(epochs) == 0:
            continue
        
        logger.info(f"\n{cond_name} condition: {len(epochs)} epochs")
        
        # Get data for each hemisphere
        times = epochs.times
        task_mask = (times >= 2) & (times <= 15)  # Task window
        baseline_mask = (times >= -5) & (times <= 0)  # Baseline
        
        # Contralateral hemisphere (LEFT hand -> RIGHT hemisphere, RIGHT hand -> LEFT hemisphere)
        if cond_name == "LEFT":
            contra_hbo = right_hbo
            contra_hbr = right_hbr
            ipsi_hbo = left_hbo
            ipsi_hbr = left_hbr
        else:
            contra_hbo = left_hbo
            contra_hbr = left_hbr
            ipsi_hbo = right_hbo
            ipsi_hbr = right_hbr
        
        # Get mean response
        if contra_hbo:
            data_hbo = epochs.get_data(picks=contra_hbo)
            mean_hbo = data_hbo.mean(axis=(0, 1))  # Average across epochs and channels
            
            baseline_hbo = mean_hbo[baseline_mask].mean()
            task_hbo = mean_hbo[task_mask].mean()
            delta_hbo = (task_hbo - baseline_hbo) * 1e6  # Convert to uM
            
            results[f"{cond_name}_contra_HbO"] = delta_hbo
            logger.info(f"  Contralateral HbO change: {delta_hbo:.3f} uM")
        
        if contra_hbr:
            data_hbr = epochs.get_data(picks=contra_hbr)
            mean_hbr = data_hbr.mean(axis=(0, 1))
            
            baseline_hbr = mean_hbr[baseline_mask].mean()
            task_hbr = mean_hbr[task_mask].mean()
            delta_hbr = (task_hbr - baseline_hbr) * 1e6
            
            results[f"{cond_name}_contra_HbR"] = delta_hbr
            logger.info(f"  Contralateral HbR change: {delta_hbr:.3f} uM")
    
    # Analyze results
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    logger.info("\nExpected physiological response during motor task:")
    logger.info("  - HbO should INCREASE (positive) in contralateral cortex")
    logger.info("  - HbR should DECREASE (negative) in contralateral cortex")
    
    logger.info("\nObserved responses:")
    for key, value in results.items():
        direction = "INCREASE" if value > 0 else "DECREASE"
        logger.info(f"  {key}: {value:.3f} uM ({direction})")
    
    # Check for swap
    swap_indicators = 0
    
    for cond in ["LEFT", "RIGHT"]:
        hbo_key = f"{cond}_contra_HbO"
        hbr_key = f"{cond}_contra_HbR"
        
        if hbo_key in results and hbr_key in results:
            hbo_val = results[hbo_key]
            hbr_val = results[hbr_key]
            
            # Expected: HbO > 0, HbR < 0
            if hbo_val < 0 and hbr_val > 0:
                swap_indicators += 1
                logger.warning(f"\n  WARNING: {cond} condition shows REVERSED pattern!")
                logger.warning(f"    HbO is negative ({hbo_val:.3f}) but should be positive")
                logger.warning(f"    HbR is positive ({hbr_val:.3f}) but should be negative")
    
    if swap_indicators > 0:
        logger.warning("\n" + "!" * 70)
        logger.warning("POTENTIAL HbO/HbR SWAP DETECTED!")
        logger.warning("The hemodynamic response shows reversed polarity.")
        logger.warning("This could indicate:")
        logger.warning("  1. Channels are swapped in the data")
        logger.warning("  2. Wavelength assignment is incorrect")
        logger.warning("  3. Unusual physiological response (less likely)")
        logger.warning("!" * 70)
    else:
        logger.info("\n  Response polarity appears CORRECT")
    
    # Create visualization
    logger.info("\nGenerating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (cond_name, cond_events) in enumerate([("LEFT", left_events), ("RIGHT", right_events)]):
        if not cond_events:
            continue
        
        cond_event_id = {k: event_id[k] for k in cond_events}
        
        epochs = mne.Epochs(
            raw_haemo,
            events,
            event_id=cond_event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=(-5, 0),
            preload=True,
        )
        
        if len(epochs) == 0:
            continue
        
        times = epochs.times
        
        # Contralateral channels
        if cond_name == "LEFT":
            contra_hbo = right_hbo
            contra_hbr = right_hbr
            title_suffix = "(Right hemisphere - contralateral)"
        else:
            contra_hbo = left_hbo
            contra_hbr = left_hbr
            title_suffix = "(Left hemisphere - contralateral)"
        
        # Plot HbO
        ax = axes[idx, 0]
        if contra_hbo:
            data = epochs.get_data(picks=contra_hbo) * 1e6
            mean_data = data.mean(axis=(0, 1))
            std_data = data.mean(axis=1).std(axis=0)
            
            ax.plot(times, mean_data, 'r-', linewidth=2, label='HbO')
            ax.fill_between(times, mean_data - std_data, mean_data + std_data, 
                           color='red', alpha=0.2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='green', linestyle='-', alpha=0.5, label='Stimulus onset')
        ax.axvspan(0, 15, color='yellow', alpha=0.1, label='Task period')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration change (uM)')
        ax.set_title(f'{cond_name} hand - HbO {title_suffix}')
        ax.legend(loc='upper right')
        ax.set_xlim(tmin, tmax)
        
        # Plot HbR
        ax = axes[idx, 1]
        if contra_hbr:
            data = epochs.get_data(picks=contra_hbr) * 1e6
            mean_data = data.mean(axis=(0, 1))
            std_data = data.mean(axis=1).std(axis=0)
            
            ax.plot(times, mean_data, 'b-', linewidth=2, label='HbR')
            ax.fill_between(times, mean_data - std_data, mean_data + std_data,
                           color='blue', alpha=0.2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='green', linestyle='-', alpha=0.5, label='Stimulus onset')
        ax.axvspan(0, 15, color='yellow', alpha=0.1, label='Task period')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration change (uM)')
        ax.set_title(f'{cond_name} hand - HbR {title_suffix}')
        ax.legend(loc='upper right')
        ax.set_xlim(tmin, tmax)
    
    plt.suptitle('HRF Polarity Check - Expected: HbO increases, HbR decreases during task',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path("data/derivatives/validation-pipeline/sub-012/ses-001")
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "debug_hrf_polarity.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"\nSaved figure to: {fig_path}")
    
    plt.close()
    
    logger.info("\n" + "=" * 70)
    logger.info("Diagnostic complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
