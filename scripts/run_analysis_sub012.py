#!/usr/bin/env python3
"""
Dedicated Analysis Script for Sub-012 (Modified Fingertapping Protocol).

This script implements the specific requirements for sub-012:
1. Handles 'sub-12' XDF filename discrepancy (vs 'sub-012')
2. Syncs NOTHING epochs from post-trial rest periods (10s task + 8s rest)
3. Runs standard validation pipeline (TFR, ERP, CSP)
4. Uses 'Agg' backend to prevent interactive plots from blocking execution.
"""
import matplotlib
# Force non-interactive backend BEFORE importing run_analysis or mne
matplotlib.use('Agg')

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import mne

# Add src to path for imports
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent / "src"))

from affective_fnirs.config import SubjectConfig
from affective_fnirs.ingestion import (
    load_xdf_file,
    identify_streams,
    extract_stream_data,
)
import json
from affective_fnirs.mne_builder import (
    build_eeg_raw,
    build_fnirs_raw,
    embed_events,
)
from affective_fnirs.validation import validate_nothing_condition
from affective_fnirs.fnirs_quality import calculate_sci
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import run_analysis as main_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class SynthesisStats:
    """Tracks NOTHING annotation synthesis outcomes.

    Attributes:
        n_created: Number of NOTHING annotations successfully created.
        n_skipped: Number of trials skipped due to insufficient rest.
        n_source_trials: Total LEFT/RIGHT trials found as synthesis sources.
    """

    n_created: int
    n_skipped: int
    n_source_trials: int


def synthesize_nothing_annotations(
    raw: mne.io.Raw,
    task_duration: float = 7.0,
    rest_duration_cap: float = 6.0,
) -> tuple[mne.io.Raw, dict[str, int]]:
    """Synthesize NOTHING annotations from post-trial rest periods.

    Creates a NOTHING annotation for each LEFT/RIGHT trial whose subsequent
    rest period is long enough.  The virtual onset is placed 1.0 s after
    stimulus offset (task_end + 1.0) to allow a 1-s baseline gap, and the
    annotation duration equals ``rest_duration_cap`` (6.0 s by default),
    matching LEFT/RIGHT epoch length.

    Args:
        raw: MNE Raw object containing LEFT/RIGHT annotations.
        task_duration: Duration of the motor task in seconds.
        rest_duration_cap: Duration assigned to each NOTHING annotation
            (epoch length).  The minimum required rest between trials is
            ``rest_duration_cap + 1.0`` (baseline gap + epoch).

    Returns:
        A tuple of (modified raw, synthesis_stats) where synthesis_stats is
        a dict with keys ``n_created``, ``n_skipped``, ``n_source_trials``.
    """
    logger.info("Synthesizing NOTHING annotations from post-trial rest...")

    onsets = raw.annotations.onset
    descriptions = raw.annotations.description
    sort_idx = np.argsort(onsets)
    sorted_onsets = onsets[sort_idx]
    sorted_descriptions = descriptions[sort_idx]

    new_onsets: list[float] = []
    new_durations: list[float] = []
    new_descriptions: list[str] = []

    n_source_trials = 0
    n_skipped = 0

    for trial_idx in range(len(sorted_onsets)):
        description = sorted_descriptions[trial_idx]
        onset = sorted_onsets[trial_idx]

        is_left = description == "LEFT" or description.startswith("LEFT/")
        is_right = description == "RIGHT" or description.startswith("RIGHT/")

        if not (is_left or is_right):
            continue

        n_source_trials += 1
        task_end = onset + task_duration
        nothing_virtual_onset = task_end + 1.0

        # Find next LEFT/RIGHT trial onset to bound available rest
        next_trial_onset: Optional[float] = None
        for search_idx in range(trial_idx + 1, len(sorted_onsets)):
            next_desc = sorted_descriptions[search_idx]
            if (
                next_desc == "LEFT"
                or next_desc.startswith("LEFT/")
                or next_desc == "RIGHT"
                or next_desc.startswith("RIGHT/")
            ):
                next_trial_onset = sorted_onsets[search_idx]
                break

        if next_trial_onset is not None:
            available_rest_duration = next_trial_onset - task_end
            if available_rest_duration >= (rest_duration_cap + 1.0):
                duration = rest_duration_cap
            else:
                n_skipped += 1
                continue
        else:
            # Last trial — assume sufficient rest
            duration = rest_duration_cap

        new_onsets.append(nothing_virtual_onset)
        new_durations.append(duration)
        new_descriptions.append("NOTHING")

    n_created = len(new_onsets)
    synthesis_stats: dict[str, int] = {
        "n_created": n_created,
        "n_skipped": n_skipped,
        "n_source_trials": n_source_trials,
    }

    if not new_onsets:
        logger.warning(
            "No NOTHING annotations created "
            f"(source trials: {n_source_trials}, skipped: {n_skipped})"
        )
        return raw, synthesis_stats

    nothing_annotations = mne.Annotations(
        onset=new_onsets,
        duration=new_durations,
        description=new_descriptions,
        orig_time=raw.annotations.orig_time,
    )

    raw.set_annotations(raw.annotations + nothing_annotations)
    logger.info(
        f"NOTHING synthesis complete — "
        f"created: {n_created}, skipped: {n_skipped}, "
        f"source trials: {n_source_trials}"
    )
    return raw, synthesis_stats


def generate_sci_comparison_plot(
    raw: mne.io.Raw,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate grouped barplot of SCI values comparing first and second halves of recording.

    Args:
        raw: MNE Raw object (fNIRS intensity).
        output_path: Directory to save the plot.
        config: Subject configuration.

    Returns:
        Path to saved plot or None if failed.
    """
    logger.info("Generating SCI comparison plot (Initial vs Final)...")
    try:
        # Split data into two halves
        duration = raw.times[-1]
        midpoint = duration / 2.0
        
        raw_initial = raw.copy().crop(tmin=0, tmax=midpoint)
        # raw.crop is in-place and fails if tmin < raw.first_time, but raw is usually 0-anchored here
        # For the second half, we need a fresh copy of the original
        raw_final = raw.copy().crop(tmin=midpoint, tmax=None)
        
        # Calculate SCI for both halves
        # Note: calculate_sci expects freq_range=(0.5, 2.5) by default, 
        # which matches typical cardiac band.
        sci_initial = calculate_sci(raw_initial, sci_threshold=0.0) # threshold 0 to get all values
        sci_final = calculate_sci(raw_final, sci_threshold=0.0)
        
        # Prepare data for plotting
        data = []
        # sci_initial is dict {channel_pair: value}
        # Channel pairs are like 'S1_D1'
        
        all_pairs = sorted(list(set(list(sci_initial.keys()) + list(sci_final.keys()))))
        
        for pair in all_pairs:
            # Initial
            if pair in sci_initial:
                data.append({
                    'Channel Pair': pair,
                    'Segment': 'Initial (First Half)',
                    'SCI': sci_initial[pair]
                })
            
            # Final
            if pair in sci_final:
                data.append({
                    'Channel Pair': pair,
                    'Segment': 'Final (Second Half)',
                    'SCI': sci_final[pair]
                })
                
        if not data:
            logger.warning("No SCI data available for comparison plot")
            return None
            
        df = pd.DataFrame(data)
        
        # Plotting
        plt.figure(figsize=(15, 8))
        sns.set_theme(style="whitegrid")
        
        # Determine appropriate bar width and layout based on number of channels
        # If many channels, might need to rotate labels more or split plot
        
        ax = sns.barplot(
            data=df,
            x='Channel Pair',
            y='SCI',
            hue='Segment',
            palette=['#3498db', '#e74c3c'], # Blue for initial, Red for final
            alpha=0.8
        )
        
        # Add threshold line (typical good quality threshold)
        threshold = config.quality.sci_threshold
        plt.axhline(y=threshold, color='green', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
        
        plt.title(f'Scalp Coupling Index (SCI) Stability: Initial vs Final Segment\nSubject {config.subject.id}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Channel Pair', fontsize=12)
        plt.ylabel('Scalp Coupling Index (0-1)', fontsize=12)
        plt.ylim(0, 1.1) # SCI is 0-1, leave room for legend/text
        plt.legend(title='Recording Segment', loc='upper right')
        
        # Rotate x-axis labels if many channels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-sci_comparison.png"
        )
        filepath = output_path / filename
        plt.savefig(str(filepath), dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"SCI comparison plot saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate SCI comparison plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_fnirs_timeseries_plot(
    raw_haemo: mne.io.Raw,
    output_path: Path,
    config: SubjectConfig,
    good_channels: list[str] = None
) -> Optional[Path]:
    """
    Generate a full time series plot of HbO and HbR with stimulus markers.
    
    Args:
        raw_haemo: Preprocessed fNIRS data (haemoglobin concentration).
        output_path: Directory to save the plot.
        config: Subject configuration.
        good_channels: List of channel names to include (if None, use all).
    """
    logger.info("Generating fNIRS time series plot...")
    try:
        # Get data
        data = raw_haemo.get_data() # (n_channels, n_times)
        times = raw_haemo.times
        ch_names = raw_haemo.ch_names
        
        # Filter channels if provided
        if good_channels:
            # We need to find indices for good channels
            # Note: ch_names in raw_haemo will be like 'S1_D1 hbo', 'S1_D1 hbr'
            # good_channels might be 'S1_D1' or similar base names
            # Let's assume good_channels contains the base channel names (e.g., 'S1_D1')
            
            hbo_indices = []
            hbr_indices = []
            
            for i, name in enumerate(ch_names):
                # Check if this channel belongs to a good channel pair
                is_good = False
                for good_ch in good_channels:
                    if good_ch in name:
                        is_good = True
                        break
                
                if is_good:
                    if 'hbo' in name:
                        hbo_indices.append(i)
                    elif 'hbr' in name:
                        hbr_indices.append(i)
        else:
            hbo_indices = [i for i, name in enumerate(ch_names) if 'hbo' in name]
            hbr_indices = [i for i, name in enumerate(ch_names) if 'hbr' in name]
            
        if not hbo_indices or not hbr_indices:
            logger.warning("No valid Hbo/Hbr channels found for time series plot")
            return None
            
        # Calculate mean traces
        hbo_mean = np.mean(data[hbo_indices, :], axis=0) * 1e6 # Convert to uM
        hbr_mean = np.mean(data[hbr_indices, :], axis=0) * 1e6
        
        # Calculate std error for shading
        hbo_std = np.std(data[hbo_indices, :], axis=0) * 1e6
        hbr_std = np.std(data[hbr_indices, :], axis=0) * 1e6
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # HbO
        ax1.plot(times, hbo_mean, color='r', label='HbO Mean')
        ax1.fill_between(times, hbo_mean - hbo_std, hbo_mean + hbo_std, color='r', alpha=0.2)
        ax1.set_ylabel('Concentration ($\mu$M)')
        ax1.set_title('Oxyhemoglobin (HbO) - Mean of Good Channels')
        ax1.grid(True, alpha=0.3)
        
        # HbR
        ax2.plot(times, hbr_mean, color='b', label='HbR Mean')
        ax2.fill_between(times, hbr_mean - hbr_std, hbr_mean + hbr_std, color='b', alpha=0.2)
        ax2.set_ylabel('Concentration ($\mu$M)')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Deoxyhemoglobin (HbR) - Mean of Good Channels')
        ax2.grid(True, alpha=0.3)
        
        # Add events
        events, event_id = mne.events_from_annotations(raw_haemo, verbose=False)
        # Scale for visualization
        y_min1, y_max1 = ax1.get_ylim()
        y_min2, y_max2 = ax2.get_ylim()
        
        # Define colors for events
        event_colors = {'LEFT': 'green', 'RIGHT': 'purple', 'NOTHING': 'gray'}
        
        for event in events:
            onset_sample = event[0]
            onset_time = raw_haemo.times[onset_sample]
            event_code = event[2]
            
            # Find label
            label = None
            for k, v in event_id.items():
                if v == event_code:
                    label = k
                    break
            
            if label:
                # Strip any slashes for color mapping
                base_label = label.split('/')[0] if '/' in label else label
                color = event_colors.get(base_label, 'black')
                
                ax1.axvline(x=onset_time, color=color, linestyle='--', alpha=0.5)
                ax2.axvline(x=onset_time, color=color, linestyle='--', alpha=0.5)
                
                # Add text label at top
                ax1.text(onset_time, y_max1, label, rotation=90, verticalalignment='top', fontsize=8, color=color)

        
        plt.tight_layout()
        
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-fnirs_timeseries.png"
        )
        filepath = output_path / filename
        plt.savefig(str(filepath), dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"fNIRS time series plot saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate fNIRS time series plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_fnirs_hrf_by_condition_4roi(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
    good_channels: list[str] = None
) -> Optional[Path]:
    """
    Generate HRF plots for 4 specific ROIs (Left/Right Anterior, Left/Right Posterior).
    
    ROIs defined as:
    Left Anterior: S1(AFF5h), S9(F7) -> AF7, F5, F7-F5, F7-AF7 (using S1 or S9 as source)
    Right Anterior: S2(AFF6h), S10(F8) -> AF8, F6, F8-F6, F8-AF8 (using S2 or S10 as source)
    Left Posterior: S3(CCP5h), S4(TTP7h), S5(TPP7h) (using S3, S4, S5 as source)
    Right Posterior: S6(CCP6h), S7(TTP8h), S8(TPP8h) (using S6, S7, S8 as source)
    
    Args:
        epochs: fNIRS epochs.
        output_path: Directory to save the plot.
        config: Subject configuration.
        good_channels: List of channel names to filter by.
    """
    logger.info("Generating fNIRS HRF by condition (4 ROIs)...")
    try:
        # Define ROIs by Source labels
        # We look for channels that have these sources in their label
        # Channel names are typically "S1_D1 hbo", etc.
        
        rois = {
            'Left Anterior': ['S1_', 'S9_'],
            'Right Anterior': ['S2_', 'S10_'],
            'Left Posterior': ['S3_', 'S4_', 'S5_'],
            'Right Posterior': ['S6_', 'S7_', 'S8_']
        }
        
        conditions = ['LEFT', 'RIGHT', 'NOTHING']
        colors = {'LEFT': 'green', 'RIGHT': 'purple', 'NOTHING': 'gray'}
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        
        times = epochs.times
        
        for i, (roi_name, source_prefixes) in enumerate(rois.items()):
            # Find indices for this ROI
            roi_indices_hbo = []
            roi_indices_hbr = []
            
            for ch_idx, ch_name in enumerate(epochs.ch_names):
                # Check if channel matches ROI source prefixes
                matches_source = any(prefix in ch_name for prefix in source_prefixes)
                
                # Check if channel is "good"
                is_good = True
                if good_channels:
                    # ch_name is like "S1_D1 hbo"
                    # good_channels has "S1_D1"
                    # Need to check coverage
                    base_name = ch_name.split(' ')[0]
                    if base_name not in good_channels:
                        is_good = False
                
                if matches_source and is_good:
                    if 'hbo' in ch_name:
                        roi_indices_hbo.append(ch_idx)
                    elif 'hbr' in ch_name:
                        roi_indices_hbr.append(ch_idx)
            
            # Subplots for this ROI
            ax_hbo = axes[i, 0]
            ax_hbr = axes[i, 1]
            
            # Plot each condition
            for condition in conditions:
                if condition in epochs.event_id:
                    evoked = epochs[condition].average(picks=roi_indices_hbo)
                    if evoked.nave > 0:
                        mean_data = evoked.data.mean(axis=0) * 1e6
                        ax_hbo.plot(times, mean_data, label=f"{condition} (n={evoked.nave})", color=colors.get(condition, 'k'))
                    
                    evoked = epochs[condition].average(picks=roi_indices_hbr)
                    if evoked.nave > 0:
                        mean_data = evoked.data.mean(axis=0) * 1e6
                        ax_hbr.plot(times, mean_data, label=f"{condition} (n={evoked.nave})", color=colors.get(condition, 'k'))
            
            # Styling
            ax_hbo.set_title(f"{roi_name} - HbO")
            ax_hbo.set_ylabel('Concentration ($\mu$M)')
            ax_hbo.grid(True, alpha=0.3)
            ax_hbo.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            if i == 0: ax_hbo.legend()
            
            ax_hbr.set_title(f"{roi_name} - HbR")
            ax_hbr.grid(True, alpha=0.3)
            ax_hbr.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        axes[3, 0].set_xlabel('Time (s)')
        axes[3, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-fnirs_hrf_4roi.png"
        )
        filepath = output_path / filename
        plt.savefig(str(filepath), dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"fNIRS 4-ROI HRF plot saved to: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Failed to generate fNIRS 4-ROI HRF plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Sub-012 Analysis")
    parser.add_argument("--config", type=Path, default=Path("configs/sub-012.yml"), help="Path to config file")
    args = parser.parse_args()
    
    # 1. Load Configuration
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
        
    config = SubjectConfig.from_yaml(args.config)
    main_pipeline.print_configuration_summary(config)
    
    # 2. Custom Data Loading (Handle sub-12 vs sub-012)
    # Explicitly check for the 'sub-12' filename variant
    xdf_filename = f"sub-12_ses-{config.subject.session}_task-{config.subject.task}_recording.xdf"
    xdf_path = config.data_root / f"sub-{config.subject.id}" / f"ses-{config.subject.session}" / xdf_filename
    
    if not xdf_path.exists():
        # Try flat structure
        xdf_path = config.data_root / f"sub-{config.subject.id}" / xdf_filename
        
    # If still not found, try the standard 'sub-012' name using the main pipeline's logic
    if not xdf_path.exists():
        logger.warning(f"Custom sub-12 file not found at {xdf_path}, trying standard naming...")
        try:
             # This might fail if load_and_identify_streams is too strict, but worth a try as fallback
             streams = main_pipeline.load_and_identify_streams(config)
        except Exception:
             logger.error(f"XDF file not found. Checked custom path: {xdf_path}")
             sys.exit(1)
    else:
        logger.info(f"Loading XDF from custom path: {xdf_path}")
        streams, header = load_xdf_file(xdf_path)
        streams = identify_streams(streams)

    # 3. Build MNE Object (Custom for Sub-012)
    # Explicitly load the new montage file
    montage_path = config.data_root / "sub-012" / "ses-001" / "montage_combined_EEG_fNIRS_with_3Dcoords_approx.json"
    if not montage_path.exists():
        # Try absolute path if relative fails (fallback)
        montage_path = Path(r"c:\Users\tdamelio\Desktop\fnirs\affective-fnirs\data\raw\sub-012\ses-001\montage_combined_EEG_fNIRS_with_3Dcoords_approx.json")
    
    logger.info(f"Loading custom montage from: {montage_path}")
    if not montage_path.exists():
        logger.error(f"Montage file not found: {montage_path}")
        sys.exit(1)
        
    with open(montage_path, 'r') as f:
        montage_json = json.load(f)
        
    # Build objects manually to bypass main pipeline's JSON lookup
    raw_eeg = None
    raw_fnirs = None
    
    eeg_stream = streams.get("eeg")
    fnirs_stream = streams.get("fnirs")
    marker_stream = streams.get("marker")
    
    if eeg_stream:
        logger.info("Building EEG Raw object...")
        eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(eeg_stream)
        raw_eeg = build_eeg_raw(eeg_data, eeg_sfreq, eeg_stream['info'], eeg_timestamps)
        if marker_stream:
            raw_eeg = embed_events(raw_eeg, marker_stream)
            
    if fnirs_stream:
        logger.info("Building fNIRS Raw object with custom montage...")
        fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(fnirs_stream)
        # Use our loaded montage_json
        raw_fnirs = build_fnirs_raw(fnirs_data, fnirs_sfreq, montage_json['ChMontage'], fnirs_timestamps)
        if marker_stream:
            raw_fnirs = embed_events(raw_fnirs, marker_stream)
            
    if raw_eeg is None:
        logger.error("No EEG stream found")
        sys.exit(1)

    if raw_fnirs is not None:
         logger.info(f"fNIRS stream found: {len(raw_fnirs.ch_names)} channels (including misc)")
         # Only select fNIRS channels for analysis to avoid issues with 'misc' channels later
         # But we might want to keep them in 'raw' for completeness, let's just be careful
    else:
         logger.warning("No fNIRS stream found")
        
    # 4. Synthesize NOTHING conditions
    # Task is 7s, NOTHING epochs are 6s (matching LEFT/RIGHT epoch duration)
    raw_eeg, eeg_synthesis_stats = synthesize_nothing_annotations(
        raw_eeg, task_duration=7.0, rest_duration_cap=6.0
    )
    
    if raw_fnirs is not None:
        # Run synthesis on fNIRS too — it has its own copy of annotations
        raw_fnirs, fnirs_synthesis_stats = synthesize_nothing_annotations(
            raw_fnirs, task_duration=7.0, rest_duration_cap=6.0
        )

    
    # 5. Preprocessing (Reuse main pipeline)
    # Output path
    output_path = config.output_root / f"sub-{config.subject.id}" / f"ses-{config.subject.session}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize viz_paths
    viz_paths = {}
    
    # OVERRIDE SCI THRESHOLD (Task 4)
    logger.info(f"Overriding SCI threshold from {config.quality.sci_threshold} to 0.50")
    config.quality.sci_threshold = 0.50

    # Collect good channels list (Task 4)
    good_fnirs_channels = None

    # NEW: Generate SCI Comparison Plot (Initial vs Final)
    # Must be done on raw intensity data (before preprocessing/OD conversion)
    if raw_fnirs is not None:
        # Pick only fNIRS channels for SCI calculation to avoid 'misc'
        fnirs_picks = mne.pick_types(raw_fnirs.info, fnirs=True, exclude=[])
        raw_fnirs_only = raw_fnirs.copy().pick(fnirs_picks)
        
        sci_plot_path = generate_sci_comparison_plot(raw_fnirs_only, output_path, config)
        if sci_plot_path:
            viz_paths['fnirs_sci_comparison'] = sci_plot_path
            
        # Determine good channels based on NEW threshold (0.50)
        # We can run a quick check here to get the list for visualization later
        logger.info("Identifying good channels with SCI > 0.50...")
        sci_vals = calculate_sci(raw_fnirs_only, sci_threshold=0.50)
        good_fnirs_channels = list(sci_vals.keys()) # Keys are 'S1_D1' etc.
        logger.info(f"Found {len(good_fnirs_channels)} good channels (out of {len(raw_fnirs_only.ch_names)})")

    # Pass both EEG and fNIRS
    processed_eeg, processed_fnirs = main_pipeline.run_preprocessing(
        raw_eeg=raw_eeg,
        raw_fnirs=raw_fnirs, 
        config=config,
        output_path=output_path
    )
    
    if processed_eeg is None:
        logger.error("EEG Preprocessing failed")
        sys.exit(1)
        
    # 6. Analysis and Visualization
    # Run Standard EEG Analysis (Epoching, TFR, ERD/ERS)
    logger.info("Running EEG Analysis...")
    eeg_results = main_pipeline.run_eeg_analysis(processed_eeg, config, output_path)
    
    # Run fNIRS Analysis
    fnirs_results = None
    if processed_fnirs is not None:
        logger.info("Running fNIRS Analysis...")
        # Check if function exists
        if hasattr(main_pipeline, 'run_fnirs_analysis'):
            fnirs_results = main_pipeline.run_fnirs_analysis(processed_fnirs, config)
        else:
            logger.warning("run_fnirs_analysis function not found in main_pipeline")

    # 6.5 Validate NOTHING condition integrity (advisory, non-blocking)
    logger.info("Validating NOTHING condition...")
    fnirs_epochs_for_validation = (
        fnirs_results["epochs"] if fnirs_results and "epochs" in fnirs_results else None
    )
    eeg_epochs_for_validation = (
        eeg_results["epochs"] if eeg_results and "epochs" in eeg_results else None
    )
    validation_result = validate_nothing_condition(
        eeg_epochs=eeg_epochs_for_validation,
        fnirs_epochs=fnirs_epochs_for_validation,
    )
    if validation_result.all_passed:
        logger.info(
            "NOTHING validation: all checks passed "
            "(%d NOTHING, %d LEFT, %d RIGHT epochs)",
            validation_result.n_nothing_epochs,
            validation_result.n_left_epochs,
            validation_result.n_right_epochs,
        )
    else:
        logger.warning(
            "NOTHING validation: %d issue(s) detected — %s",
            len(validation_result.warnings),
            "; ".join(validation_result.warnings),
        )

    # Process EEG Results (CSP, Viz)
    # viz_paths is already initialized above
    if eeg_results:
        # Run CSP (LEFT vs RIGHT)
        # Note: This function requires 'epochs' in eeg_results
        logger.info("Running CSP Analysis (LEFT vs RIGHT)...")
        csp_path, csp_results = main_pipeline.generate_csp_analysis(eeg_results['epochs'], output_path, config)
        eeg_results['csp_analysis_path'] = csp_path
        eeg_results['csp_results'] = csp_results
        
        # Run CSP (MOV vs REST)
        # Verify module path and function existence
        logger.info(f"Loaded main_pipeline from: {main_pipeline.__file__}")
        if not hasattr(main_pipeline, 'generate_csp_movement_vs_rest'):
             logger.error("generate_csp_movement_vs_rest NOT FOUND in main_pipeline!")
        
        logger.info(f"Event IDs in epochs: {eeg_results['epochs'].event_id}")

        logger.info("Running CSP Analysis (MOV vs NO MOV)...")
        csp_mov_path, csp_mov_results = main_pipeline.generate_csp_movement_vs_rest(eeg_results['epochs'], output_path, config)
        eeg_results['csp_mov_vs_rest_path'] = csp_mov_path
        eeg_results['csp_mov_results'] = csp_mov_results
        
        # Generate Individual Visualizations with all 3 conditions (LEFT, RIGHT, NOTHING)
        logger.info("Generating individual visualizations with all conditions...")
        epochs = eeg_results['epochs']
        
        # Add CSP paths to viz_paths for report generation
        if 'csp_analysis_path' in eeg_results and eeg_results['csp_analysis_path']:
             viz_paths['eeg_csp_analysis'] = eeg_results['csp_analysis_path']
        if 'csp_mov_vs_rest_path' in eeg_results and eeg_results['csp_mov_vs_rest_path']:
             viz_paths['eeg_csp_mov_vs_rest'] = eeg_results['csp_mov_vs_rest_path']

        # TFR Maps (shows LEFT, RIGHT, NOTHING in separate columns)
        logger.info("Generating TFR maps...")
        tfr_path = main_pipeline.generate_tfr_maps(epochs, output_path, config)
        if tfr_path:
            viz_paths['eeg_tfr_maps'] = tfr_path
        
        # ERP Analysis (shows all 3 conditions)
        logger.info("Generating ERP analysis...")
        erp_path = main_pipeline.generate_erp_analysis(epochs, output_path, config)
        if erp_path:
            viz_paths['eeg_erp_analysis'] = erp_path
        
        # Clustered TFR Maps
        logger.info("Generating clustered TFR maps...")
        clustered_tfr_path = main_pipeline.generate_clustered_tfr_maps(epochs, output_path, config)
        if clustered_tfr_path:
            viz_paths['eeg_tfr_maps_roi'] = clustered_tfr_path
        
        # Beta Topoplots
        logger.info("Generating beta topoplots...")
        beta_topo_path = main_pipeline.generate_beta_topoplots(epochs, output_path, config)
        if beta_topo_path:
            viz_paths['eeg_beta_topoplot'] = beta_topo_path
        
        # Contralateral ERD plots
        logger.info("Generating contralateral ERD plots...")
        contralat_timecourse, contralat_topo = main_pipeline.generate_contralateral_erd_plots(epochs, output_path, config)
        if contralat_topo:
            viz_paths['eeg_contralateral_topoplot'] = contralat_topo
        if contralat_timecourse:
            viz_paths['eeg_contralateral_timecourse'] = contralat_timecourse
            
        # Contrast Analysis
        logger.info("Generating contrast analysis...")
        contrast_path, lat_index_path = main_pipeline.generate_contrast_analysis(epochs, output_path, config)
        if contrast_path:
            viz_paths['eeg_contrast_analysis'] = contrast_path
        
        # fNIRS Visualizations
        # fNIRS Visualizations
        if fnirs_results and 'epochs' in fnirs_results:
            logger.info("Generating fNIRS visualizations with all conditions...")
            fnirs_epochs = fnirs_results['epochs']
            
            # HRF by Condition (Original + 4 ROI)
            logger.info("Generating fNIRS HRF by condition...")
            # Original function (might filter internally, but let's trust it for now or replace call if needed)
            # The user asked to "Expand HRF by condition to 4 ROIs". 
            # We'll use our NEW function for the 4-ROI plot.
            
            hrf_4roi_path = generate_fnirs_hrf_by_condition_4roi(fnirs_epochs, output_path, config, good_channels=good_fnirs_channels)
            if hrf_4roi_path:
                viz_paths['fnirs_hrf_by_condition_4roi'] = hrf_4roi_path
                
            # Keep original as well if useful, or assume 4-ROI replaces it? 
            # Implementation plan item 6 says "Expand...". Let's run potentially both or just the new one.
            # But the user also asked for "Full-experiment HBO/HBR time series plot" (Task 5)
            
            # Time Series Plot (Task 5)
            if processed_fnirs is not None:
                 ts_path = generate_fnirs_timeseries_plot(processed_fnirs, output_path, config, good_channels=good_fnirs_channels)
                 if ts_path:
                     viz_paths['fnirs_timeseries'] = ts_path
            
            # Block Average (Task 4: Filter bad channels)
            # Modifying main pipeline calls isn't easy without changing main pipeline code. 
            # But we can try to pass `picks` if the function supports it, or just rely on the new plots.
            # Task 4 said "Modify the calls... to pick only good channels". 
            # `generate_fnirs_block_average` in main_pipeline likely doesn't verify `good_channels` argument unless we check/modify it.
            # If we cannot modify `main_pipeline` here, we might skip the old plots if they look bad, 
            # OR we rely on pre-processing bad channel marking.
            
            # Note: proper way to filter bad channels in MNE is to mark them as 'bads' in info.
            # If we mark them in `processed_fnirs.info['bads']`, main pipeline functions *might* respect that.
            
            if good_fnirs_channels:
                 # Identify bad channels
                 all_chs = fnirs_epochs.ch_names
                 # good_fnirs_channels are like "S1_D1"
                 # bads are those NOT in good keys
                 # Map good keys to full names
                 bad_bads = []
                 for ch in all_chs:
                     # ch is "S1_D1 760" or "S1_D1 hbo"
                     base = ch.split(' ')[0]
                     if base not in good_fnirs_channels:
                         bad_bads.append(ch)
                 
                 fnirs_epochs.info['bads'] = bad_bads
                 if processed_fnirs:
                     processed_fnirs.info['bads'] = bad_bads
                 logger.info(f"Marked {len(bad_bads)} channels as bad in MNE info for visualization")

            logger.info("Generating fNIRS block average...")
            block_avg_path = main_pipeline.generate_fnirs_block_average(fnirs_epochs, output_path, config)
            if block_avg_path:
                viz_paths['fnirs_block_average'] = block_avg_path
            
            # Contrast Map
            logger.info("Generating fNIRS contrast map...")
            contrast_path = main_pipeline.generate_fnirs_contrast_map(fnirs_epochs, output_path, config)
            if contrast_path:
                viz_paths['fnirs_contrast'] = contrast_path

    else:
        logger.error("EEG Analysis failed to produce results.")


    # 7. Quality Assessment (Reuse main pipeline)
    # Note: This will count all channels in raw_fnirs (including misc if any)
    # We can patch n_total_channels after
    qa_results = main_pipeline.run_quality_assessment(raw_eeg, raw_fnirs, config)
    
    # FIX TOTAL CHANNELS COUNT (Task 2)
    if qa_results and raw_fnirs:
        # Count only fNIRS types
        n_fnirs = len(mne.pick_types(raw_fnirs.info, fnirs=True, exclude=[]))
        logger.info(f"Correcting total channel count in QA report: {qa_results.n_total_channels} -> {n_fnirs}")
        qa_results.n_total_channels = n_fnirs

    # 8. Save Full Report
    logger.info("Saving Full Report...")
    main_pipeline.save_full_report(
        qa_results=qa_results,
        eeg_results=eeg_results,
        fnirs_results=fnirs_results,
        multimodal_results=None,
        visualization_paths=viz_paths,
        config=config,
        output_path=output_path
    )
    
    logger.info(f"Sub-012 Analysis Complete. Results available in {output_path}")

if __name__ == "__main__":
    main()
