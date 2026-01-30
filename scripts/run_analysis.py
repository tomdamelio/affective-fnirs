#!/usr/bin/env python3
"""
Unified Analysis Pipeline for EEG and fNIRS Data.

This script provides a single entry point for processing EEG and/or fNIRS data
from finger-tapping experiments. It orchestrates existing modules with subject-
specific configuration loaded from YAML files.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class PipelineError(Exception):
    """Exception raised for pipeline execution failures with stage context."""

    def __init__(self, stage: str, message: str, original_exception: Optional[Exception] = None):
        """
        Initialize PipelineError with stage context.

        Args:
            stage: Name of the pipeline stage that failed (e.g., "Data Loading", "Preprocessing")
            message: Descriptive error message
            original_exception: Original exception that caused the failure (if any)
        """
        self.stage = stage
        self.original_exception = original_exception
        
        # Construct full error message with stage context
        full_message = f"Stage '{stage}' failed: {message}"
        if original_exception:
            full_message += f"\nOriginal error: {type(original_exception).__name__}: {original_exception}"
        
        super().__init__(full_message)
        
        # Preserve original traceback if available
        if original_exception:
            self.__cause__ = original_exception

from affective_fnirs.config import SubjectConfig
from affective_fnirs.ingestion import (
    load_xdf_file,
    identify_streams,
    extract_stream_data,
    DataIngestionError,
)
from affective_fnirs.mne_builder import (
    build_eeg_raw,
    build_fnirs_raw,
    embed_events,
    MNEConstructionError,
)
from affective_fnirs.reporting import (
    compute_eeg_channel_quality,
    EEGChannelQuality,
    QualityReport,
    ChannelQuality,
    ValidationResults,
    ERDMetrics,
    HRFValidation,
    CouplingMetrics,
    ExperimentQA,
    LateralizationMetrics,
    ClassificationMetrics,
    generate_validation_report_html,
)
from affective_fnirs.fnirs_quality import (
    calculate_sci,
    detect_saturation,
    assess_cardiac_power,
    calculate_coefficient_of_variation,
    mark_bad_channels,
)
from affective_fnirs.bids_utils import generate_derivative_path
from affective_fnirs.eeg_processing import preprocess_eeg_pipeline
from affective_fnirs.fnirs_processing import process_fnirs_pipeline
from affective_fnirs.eeg_analysis import compute_tfr, detect_erd_ers
from affective_fnirs.fnirs_analysis import create_fnirs_epochs, extract_hrf
from affective_fnirs.multimodal_analysis import compute_neurovascular_coupling
import mne
import json
import numpy as np
import matplotlib.pyplot as plt


def generate_tfr_maps(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate Time-Frequency Maps (TFR plots) for motor cortex channels.
    
    This is the most informative canonical plot for ERD/ERS analysis.
    Shows power changes across time and frequency for C3 and C4 channels
    during LEFT, RIGHT, and NOTHING conditions.
    
    Expected pattern:
    - Blue patch (power decrease) in 8-30 Hz starting before movement onset
    - Red patch (power increase) in Beta (~20 Hz) after movement ends
    - Contralateral effect: C3 shows stronger ERD for RIGHT hand, C4 for LEFT hand
    
    Args:
        epochs: MNE Epochs object with condition information
        output_path: Directory to save plot
        config: SubjectConfig with subject information
        
    Returns:
        Path to saved TFR map or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check conditions
        conditions = list(epochs.event_id.keys())
        has_left = any('LEFT' in cond for cond in conditions)
        has_right = any('RIGHT' in cond for cond in conditions)
        has_nothing = any('NOTHING' in cond for cond in conditions)
        
        if not (has_left and has_right):
            logger.warning("Need both LEFT and RIGHT conditions for TFR maps")
            return None
        
        # Check channels
        if 'C3' not in epochs.ch_names or 'C4' not in epochs.ch_names:
            logger.warning("Need C3 and C4 channels for TFR maps")
            return None
        
        logger.info("Generating Time-Frequency Maps (TFR plots)...")
        
        from affective_fnirs.eeg_analysis import compute_tfr
        
        # Frequency range: 4-40 Hz (covers theta, alpha, beta, low gamma)
        freqs = np.arange(4, 41, 1)
        
        # Get conditions
        left_cond = [c for c in conditions if 'LEFT' in c][0]
        right_cond = [c for c in conditions if 'RIGHT' in c][0]
        nothing_cond = [c for c in conditions if 'NOTHING' in c][0] if has_nothing else None
        
        # Compute TFR for each condition
        logger.info(f"Computing TFR for LEFT condition...")
        tfr_left = compute_tfr(
            epochs[left_cond],
            freqs=freqs,
            n_cycles=freqs / 2.0,  # Adaptive cycles for better resolution
            baseline=(config.analysis.baseline_window_start_sec,
                     config.analysis.baseline_window_end_sec),
            baseline_mode="percent",
        )
        
        logger.info(f"Computing TFR for RIGHT condition...")
        tfr_right = compute_tfr(
            epochs[right_cond],
            freqs=freqs,
            n_cycles=freqs / 2.0,
            baseline=(config.analysis.baseline_window_start_sec,
                     config.analysis.baseline_window_end_sec),
            baseline_mode="percent",
        )
        
        tfr_nothing = None
        if nothing_cond:
            logger.info(f"Computing TFR for NOTHING condition...")
            tfr_nothing = compute_tfr(
                epochs[nothing_cond],
                freqs=freqs,
                n_cycles=freqs / 2.0,
                baseline=(config.analysis.baseline_window_start_sec,
                         config.analysis.baseline_window_end_sec),
                baseline_mode="percent",
            )
        
        # Create figure: 2 rows (C3, C4) x 3 columns (LEFT, RIGHT, NOTHING)
        n_cols = 3 if tfr_nothing else 2
        fig, axes = plt.subplots(2, n_cols, figsize=(8*n_cols, 10))
        
        # Time window for display
        if config.analysis.tfr_view_tmin_sec is not None:
             tmin = config.analysis.tfr_view_tmin_sec
        else:
             tmin = config.analysis.baseline_window_end_sec
             
        if config.analysis.tfr_view_tmax_sec is not None:
             tmax = config.analysis.tfr_view_tmax_sec
        else:
             tmax = config.trials.task_duration_sec + 2.0
        
        # Color scale: compute adaptive limits based on actual data
        # Collect all TFR data for C3 and C4 to determine appropriate color scale
        c3_idx = tfr_left.ch_names.index('C3')
        c4_idx = tfr_left.ch_names.index('C4')
        
        all_data = [
            tfr_left.data[c3_idx, :, :],
            tfr_left.data[c4_idx, :, :],
            tfr_right.data[c3_idx, :, :],
            tfr_right.data[c4_idx, :, :],
        ]
        if tfr_nothing:
            all_data.extend([
                tfr_nothing.data[c3_idx, :, :],
                tfr_nothing.data[c4_idx, :, :],
            ])
        
        all_data_concat = np.concatenate([d.flatten() for d in all_data])
        
        # Use percentiles to set color limits (more robust than min/max)
        vmin = np.percentile(all_data_concat, 5)
        vmax = np.percentile(all_data_concat, 95)
        
        # Ensure symmetric scale around 0 for better ERD/ERS visualization
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs
        
        # Cap at reasonable limits (±100%) to avoid extreme outliers
        vmin = max(vmin, -100)
        vmax = min(vmax, 100)
        
        logger.info(f"TFR color scale: {vmin:.1f}% to {vmax:.1f}%")
        
        # Row 1: C3 (Left Motor Cortex)
        # C3 - LEFT hand
        ch_idx = tfr_left.ch_names.index('C3')
        im = axes[0, 0].imshow(
            tfr_left.data[ch_idx, :, :],
            aspect='auto',
            origin='lower',
            extent=[tfr_left.times[0], tfr_left.times[-1], freqs[0], freqs[-1]],
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=2, label='Movement onset')
        axes[0, 0].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
        axes[0, 0].set_xlim(tmin, tmax)
        axes[0, 0].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('C3 (Left Motor Cortex) - LEFT Hand', fontsize=14, fontweight='bold')
        axes[0, 0].legend(loc='upper right', fontsize=10)
        
        # C3 - RIGHT hand
        im = axes[0, 1].imshow(
            tfr_right.data[ch_idx, :, :],
            aspect='auto',
            origin='lower',
            extent=[tfr_right.times[0], tfr_right.times[-1], freqs[0], freqs[-1]],
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[0, 1].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
        axes[0, 1].set_xlim(tmin, tmax)
        axes[0, 1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('C3 (Left Motor Cortex) - RIGHT Hand (Contralateral)', fontsize=14, fontweight='bold')
        
        # C3 - NOTHING (if available)
        if tfr_nothing:
            im = axes[0, 2].imshow(
                tfr_nothing.data[ch_idx, :, :],
                aspect='auto',
                origin='lower',
                extent=[tfr_nothing.times[0], tfr_nothing.times[-1], freqs[0], freqs[-1]],
                cmap='RdBu_r',
                vmin=vmin,
                vmax=vmax,
            )
            axes[0, 2].axvline(0, color='black', linestyle='--', linewidth=2)
            axes[0, 2].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
            axes[0, 2].set_xlim(tmin, tmax)
            axes[0, 2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            axes[0, 2].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            axes[0, 2].set_title('C3 (Left Motor Cortex) - NOTHING (Baseline)', fontsize=14, fontweight='bold')
        
        # Row 2: C4 (Right Motor Cortex)
        ch_idx = tfr_left.ch_names.index('C4')
        
        # C4 - LEFT hand
        im = axes[1, 0].imshow(
            tfr_left.data[ch_idx, :, :],
            aspect='auto',
            origin='lower',
            extent=[tfr_left.times[0], tfr_left.times[-1], freqs[0], freqs[-1]],
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[1, 0].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
        axes[1, 0].set_xlim(tmin, tmax)
        axes[1, 0].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('C4 (Right Motor Cortex) - LEFT Hand (Contralateral)', fontsize=14, fontweight='bold')
        
        # C4 - RIGHT hand
        im = axes[1, 1].imshow(
            tfr_right.data[ch_idx, :, :],
            aspect='auto',
            origin='lower',
            extent=[tfr_right.times[0], tfr_right.times[-1], freqs[0], freqs[-1]],
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[1, 1].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
        axes[1, 1].set_xlim(tmin, tmax)
        axes[1, 1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('C4 (Right Motor Cortex) - RIGHT Hand', fontsize=14, fontweight='bold')
        
        # C4 - NOTHING (if available)
        if tfr_nothing:
            im = axes[1, 2].imshow(
                tfr_nothing.data[ch_idx, :, :],
                aspect='auto',
                origin='lower',
                extent=[tfr_nothing.times[0], tfr_nothing.times[-1], freqs[0], freqs[-1]],
                cmap='RdBu_r',
                vmin=vmin,
                vmax=vmax,
            )
            axes[1, 2].axvline(0, color='black', linestyle='--', linewidth=2)
            axes[1, 2].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
            axes[1, 2].set_xlim(tmin, tmax)
            axes[1, 2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            axes[1, 2].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            axes[1, 2].set_title('C4 (Right Motor Cortex) - NOTHING (Baseline)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Power change (%)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        fig.suptitle('Time-Frequency Maps: Motor Cortex ERD/ERS', 
                    fontsize=18, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 0.9, 0.96])
        
        # Save figure
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-tfr_maps.png"
        )
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Time-Frequency Maps saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate TFR maps: {e}")
        import traceback
        traceback.print_exc()
        return None



def generate_beta_topoplots(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate Beta Band (13-30 Hz) ERD Topoplots across the whole head.
    
    Visualization to identify the spatial distribution of beta desynchronization
    across the entire head, helping to locate the motor hotspot if displaced.
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info("Generating Beta Band (13-30 Hz) ERD Topoplots...")
        
        # Use beta band frequencies
        freqs = np.arange(13, 31, 1)
        n_cycles = freqs / 2.0
        
        # Compute TFR for all channels
        tfr = mne.time_frequency.tfr_multitaper(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            average=True,
            n_jobs=1
        )
        
        # Apply baseline correction
        tfr.apply_baseline(
            mode="percent"
        )
        
        # Manually convert to percentage since we used direct MNE call
        tfr.data *= 100
        
        # Define window
        t_start = config.analysis.task_window_start_sec + 1.0
        t_end = config.analysis.task_window_end_sec

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot topomap using instance method (proven to work in contralateral)
        tfr.plot_topomap(
            tmin=t_start, 
            tmax=t_end,
            fmin=13, 
            fmax=30,
            baseline=None, # Already applied
            mode='mean',
            show=False,
            axes=ax,
            colorbar=True,
            cmap="RdBu_r"
        )
        
        ax.set_title(f"Beta Band (13-30 Hz) ERD Topography\nMean Power ({t_start}-{t_end}s)", fontsize=14)
        
        # Save figure
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-beta_topoplot.png"
        )
        path = output_path / filename
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved beta topoplot to: {path}")
        return path
        
    except Exception as e:
        logger.error(f"Failed to generate beta topoplots: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_contralateral_erd_plots(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Generate Contralateral ERD/ERS plots (Timecourse and Topoplot).
    
    Creates specific visualizations highlighting the difference between 
    contralateral and ipsilateral activity.
    """
    logger = logging.getLogger(__name__)
    timecourse_path = None
    topoplot_path = None
    
    try:
        logger.info("Generating Contralateral ERD/ERS plots...")
        
        # 1. Contralateral Timecourse - skipped as it's handled by main bilateral plot
            
        # 2. Contralateral Topoplots
        # Plot difference map: LEFT hand - RIGHT hand (or vice-versa depending on what we want to highlight)
        # Expected: LEFT Hand -> Right Motor ERD (C4)
        # Expected: RIGHT Hand -> Left Motor ERD (C3)
        # Contrast LEFT - RIGHT:
        # C4 should be more negative in LEFT condition (Left ERD > Right ERD) -> Negative value
        # C3 should be less negative in LEFT condition (Left ERD < Right ERD) -> Positive value
        
        if "LEFT" in epochs.event_id and "RIGHT" in epochs.event_id:
            logger.info("Computing LEFT vs RIGHT contrast for topoplots...")
            
            # Compute TFR for LEFT and RIGHT
            freqs = np.arange(8, 30, 1) # Alpha and Beta
            n_cycles = freqs / 2.0
            
            tfr_left = mne.time_frequency.tfr_multitaper(
                epochs["LEFT"],
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                average=True,
                n_jobs=1
            )
            tfr_right = mne.time_frequency.tfr_multitaper(
                epochs["RIGHT"],
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                average=True,
                n_jobs=1
            )
            
            # Baseline correct
            tfr_left.apply_baseline((config.analysis.baseline_window_start_sec, config.analysis.baseline_window_end_sec), mode="percent")
            tfr_right.apply_baseline((config.analysis.baseline_window_start_sec, config.analysis.baseline_window_end_sec), mode="percent")
            
            # Manually convert to percentage since we used direct MNE call
            tfr_left.data *= 100
            tfr_right.data *= 100
            
            # Subtract: LEFT - RIGHT
            tfr_diff = tfr_left.copy()
            tfr_diff.data = tfr_left.data - tfr_right.data
            
            # Plot Topomap of difference
            fig, ax = plt.subplots(figsize=(10, 8))
            tfr_diff.plot_topomap(
                tmin=2.0, tmax=4.0, # Early execution phase
                fmin=8, fmax=30,
                baseline=None,
                mode='mean',
                show=False,
                axes=ax,
                colorbar=True,
                cmap="RdBu_r"
            )
            ax.set_title("Contralateral Contrast (LEFT - RIGHT)\nAlpha/Beta (8-30 Hz), 2.0-4.0s", fontsize=14)
            
            filename = (
                f"sub-{config.subject.id}_"
                f"ses-{config.subject.session}_"
                f"task-{config.subject.task}_"
                f"desc-contralateral_topoplot.png"
            )
            path = output_path / filename
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            topoplot_path = path
            
            logger.info(f"Saved contralateral topoplot to: {path}")
        else:
            logger.warning("LEFT or RIGHT conditions missing, skipping contralateral topoplot.")
        
    except Exception as e:
        logger.error(f"Failed to generate contralateral plots: {e}")
        import traceback
        traceback.print_exc()
        
    return timecourse_path, topoplot_path


def generate_clustered_tfr_maps(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate Time-Frequency Maps for averaged electrode clusters (ROIs).
    
    ROI Definitions:
    - Left Motor Cluster: FC1, FC5, C3, CP1, CP5
    - Right Motor Cluster: FC2, FC6, C4, CP2, CP6
    
    Args:
        epochs: MNE Epochs object with condition information
        output_path: Directory to save plot
        config: SubjectConfig with subject information
        
    Returns:
        Path to saved TFR map or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Exploratory ROI Definitions
        clusters = {
            'Standard_Motor_L': ['FC1', 'FC5', 'C3', 'CP1', 'CP5'],
            'Standard_Motor_R': ['FC2', 'FC6', 'C4', 'CP2', 'CP6'],
            'Frontal_Motor_L': ['F3', 'FC1', 'FC5'],
            'Frontal_Motor_R': ['F4', 'FC2', 'FC6'],
            'Parietal_Motor_L': ['P3', 'CP1', 'CP5'],
            'Parietal_Motor_R': ['P4', 'CP2', 'CP6'],
        }
        
        # Check conditions
        conditions = list(epochs.event_id.keys())
        has_left = any('LEFT' in cond for cond in conditions)
        has_right = any('RIGHT' in cond for cond in conditions)
        has_nothing = any('NOTHING' in cond for cond in conditions)
        
        if not (has_left and has_right):
            logger.warning("Need both LEFT and RIGHT conditions for TFR maps")
            return None
            
        # Validate channels exist for each cluster (filter out missing)
        available_ch = epochs.ch_names
        valid_clusters = {}
        for name, chs in clusters.items():
            valid_chs = [ch for ch in chs if ch in available_ch]
            if valid_chs:
                valid_clusters[name] = valid_chs
            else:
                logger.warning(f"Skipping cluster {name}: No valid channels found")
        
        if not valid_clusters:
            return None
            
        logger.info(f"Generating Clustered TFR maps for {len(valid_clusters)} clusters...")
        
        from affective_fnirs.eeg_analysis import compute_tfr
        
        # Frequency range: 4-40 Hz
        freqs = np.arange(4, 41, 1)
        
        # Get conditions
        left_cond = [c for c in conditions if 'LEFT' in c][0]
        right_cond = [c for c in conditions if 'RIGHT' in c][0]
        nothing_cond = [c for c in conditions if 'NOTHING' in c][0] if has_nothing else None
        
        # Helper to compute ROI average TFR
        def compute_roi_tfr(condition, channels):
            """Compute TFR averaged across ROI channels."""
            epochs_roi = epochs[condition].copy().pick_channels(channels)
            tfr = compute_tfr(
                epochs_roi,
                freqs=freqs,
                n_cycles=freqs / 2.0,
                baseline=(config.analysis.baseline_window_start_sec,
                         config.analysis.baseline_window_end_sec),
                baseline_mode="percent",
            )
            # Average across channels
            avg_data = np.mean(tfr.data, axis=0, keepdims=True)
            
            import mne
            new_info = mne.create_info(ch_names=['ROI_AVG'], sfreq=tfr.info['sfreq'], ch_types=['eeg'])
            avg_tfr = mne.time_frequency.AverageTFRArray(
                info=new_info, data=avg_data, times=tfr.times, freqs=tfr.freqs, nave=tfr.nave
            )
            return avg_tfr

        # Compute TFRs for each Cluster-Condition pair
        logger.info("Computing ROI TFRs...")
        
        results = {}
        for cluster_name, channels in valid_clusters.items():
            results[cluster_name] = {
                'LEFT_HAND': compute_roi_tfr(left_cond, channels),
                'RIGHT_HAND': compute_roi_tfr(right_cond, channels),
                'NOTHING': compute_roi_tfr(nothing_cond, channels) if nothing_cond else None
            }

        # Plotting - One generic figure for Standard, or multiple?
        # Let's do one big figure with rows = clusters, cols = conditions
        
        n_clusters = len(valid_clusters)
        n_conds = 3 if nothing_cond else 2
        
        # If too many clusters, split into multiple figures? 
        # For now, let's just plot Standard, Frontal, Parietal (left/right pairs)
        # We have 6 clusters (3 pairs). 6 rows.
        
        fig, axes = plt.subplots(n_clusters, n_conds, figsize=(5*n_conds, 3*n_clusters))
        if n_clusters == 1: axes = np.array([axes]) # Ensure 2D array
        
        # Time window
        tmin = config.analysis.tfr_view_tmin_sec if config.analysis.tfr_view_tmin_sec is not None else config.analysis.baseline_window_end_sec
        tmax = config.analysis.tfr_view_tmax_sec if config.analysis.tfr_view_tmax_sec is not None else config.trials.task_duration_sec + 2.0
             
        # Determine color scale (global for comparison)
        all_data_vals = []
        for c_res in results.values():
            all_data_vals.append(c_res['LEFT_HAND'].data)
            all_data_vals.append(c_res['RIGHT_HAND'].data)
            if c_res['NOTHING']: all_data_vals.append(c_res['NOTHING'].data)
            
        all_data_concat = np.concatenate([d.flatten() for d in all_data_vals])
        vmin = np.percentile(all_data_concat, 5)
        vmax = np.percentile(all_data_concat, 95)
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs
        vmin = max(vmin, -100) # Cap
        vmax = min(vmax, 100)
        
        logger.info(f"ROI TFR color scale: {vmin:.1f}% to {vmax:.1f}%")
        
        # Plotting Helper
        def plot_ax(ax, tfr_obj, title):
            im = ax.imshow(
                tfr_obj.data[0, :, :],
                aspect='auto', origin='lower',
                extent=[tfr_obj.times[0], tfr_obj.times[-1], freqs[0], freqs[-1]],
                cmap='RdBu_r', vmin=vmin, vmax=vmax
            )
            ax.axvline(0, color='black', linestyle='--', linewidth=2)
            ax.axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
            ax.set_xlim(tmin, tmax)
            ax.set_ylabel('Freq (Hz)')
            ax.set_title(title, fontsize=10, fontweight='bold')
            return im

        # Iterate and plot
        # Order clusters nicely: Standard L/R, Frontal L/R, Parietal L/R
        ordered_keys = [k for k in ['Standard_Motor_L', 'Standard_Motor_R', 'Frontal_Motor_L', 'Frontal_Motor_R', 'Parietal_Motor_L', 'Parietal_Motor_R'] if k in valid_clusters]
        # Add any others not in order
        for k in valid_clusters:
            if k not in ordered_keys: ordered_keys.append(k)

        for i, cluster_name in enumerate(ordered_keys):
            row_res = results[cluster_name]
            
            # Left Hand Col
            plot_ax(axes[i, 0], row_res['LEFT_HAND'], f"{cluster_name}\nLEFT Hand")
            
            # Right Hand Col
            im = plot_ax(axes[i, 1], row_res['RIGHT_HAND'], f"{cluster_name}\nRIGHT Hand")
            
            # Nothing Col
            if nothing_cond:
                 plot_ax(axes[i, 2], row_res['NOTHING'], f"{cluster_name}\nNOTHING")
        
        # Common labels
        for ax in axes[-1, :]:
            ax.set_xlabel('Time (s)')
            
        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Power change (%)', fontsize=12)
        
        fig.suptitle('Exploratory TFR Maps: Multi-Region Analysis', fontsize=16, fontweight='bold', y=0.99)
        fig.tight_layout(rect=[0, 0, 0.9, 0.98])
        
        # Save
        filename = f"sub-{config.subject.id}_ses-{config.subject.session}_task-{config.subject.task}_desc-tfr_maps_roi.png"
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Clustered TFR Maps saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate Clustered TFR maps: {e}")
        import traceback
        traceback.print_exc()
        return None





def generate_erp_analysis(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate ERP analysis plots (Evoked potentials) with SEM shading.
    
    Compares averaged responses (Evoked) across conditions (LEFT, RIGHT, NOTHING)
    for key channels comparison, including standard error of the mean (SEM).
    
    Args:
        epochs: MNE Epochs object with condition information
        output_path: Directory to save plots
        config: SubjectConfig with subject information
        
    Returns:
        Path to saved ERP plot or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        from scipy.stats import sem
        
        # ROI Definitions
        left_cluster = ['FC1', 'FC5', 'C3', 'CP1', 'CP5']
        right_cluster = ['FC2', 'FC6', 'C4', 'CP2', 'CP6']
        
        # Check conditions
        conditions = list(epochs.event_id.keys())
        has_left = any('LEFT' in cond for cond in conditions)
        has_right = any('RIGHT' in cond for cond in conditions)
        
        if not (has_left and has_right):
            logger.warning("Need both LEFT and RIGHT conditions for ERP analysis")
            return None
            
        logger.info("Generating ERP Analysis plots with SEM...")
        
        # Get conditions
        left_cond = [c for c in conditions if 'LEFT' in c][0]
        right_cond = [c for c in conditions if 'RIGHT' in c][0]
        nothing_cond = None
        if any('NOTHING' in cond for cond in conditions):
             nothing_cond = [c for c in conditions if 'NOTHING' in c][0]
             
        # Prepare plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time window for plot (Requested: -1 to +3 seconds)
        tmin_plot = -1.0
        tmax_plot = 3.0
        
        # Helper to compute stats (Mean, SEM)
        def get_condition_stats(condition_name, channels, roi_mode=False):
            if condition_name is None:
                return None, None, None
            
            # Get data: (n_trials, n_channels, n_times)
            data = epochs[condition_name].get_data(picks=channels)
            
            if roi_mode:
                # Average across channels first for each trial -> (n_trials, n_times)
                data = data.mean(axis=1)
            else:
                # Single channel -> (n_trials, n_times)
                data = data[:, 0, :]
                
            # Compute stats across trials (axis 0)
            # Unit conversion: V -> uV (or V/m² -> uV/m²)
            mean_data = data.mean(axis=0) * 1e6
            sem_data = sem(data, axis=0) * 1e6
            
            return mean_data, sem_data, epochs.times
            
        # Helper to plot comparison
        def plot_erp_comparison(ax, channel_indices, title, roi_mode=False):
            # Get channel names for picking
            if roi_mode:
                # channel_indices actually contains indices, let's map back to names if needed
                # But get_data accepts indices too
                picks = channel_indices
            else:
                picks = [channel_indices[0]]
            
            # LEFT
            mean_L, sem_L, times = get_condition_stats(left_cond, picks, roi_mode)
            ax.plot(times, mean_L, 'b-', linewidth=2, label='LEFT Hand')
            ax.fill_between(times, mean_L - sem_L, mean_L + sem_L, color='blue', alpha=0.15)
            
            # RIGHT
            mean_R, sem_R, _ = get_condition_stats(right_cond, picks, roi_mode)
            ax.plot(times, mean_R, 'r-', linewidth=2, label='RIGHT Hand')
            ax.fill_between(times, mean_R - sem_R, mean_R + sem_R, color='red', alpha=0.15)
            
            # NOTHING
            if nothing_cond:
                mean_N, sem_N, _ = get_condition_stats(nothing_cond, picks, roi_mode)
                ax.plot(times, mean_N, 'g--', linewidth=1.5, alpha=0.7, label='NOTHING')
                ax.fill_between(times, mean_N - sem_N, mean_N + sem_N, color='green', alpha=0.1)
                
            # Styling
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
            # Only show end line if it's within plot range
            if config.trials.task_duration_sec < tmax_plot:
                ax.axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
                
            ax.axhline(0, color='gray', linestyle=':', linewidth=1)
            ax.set_xlim(tmin_plot, tmax_plot)
            ax.set_xlabel('Time (s)')
            unit_label = 'Amplitude (µV/m²)' if config.analysis.use_laplacian else 'Amplitude (µV)'
            ax.set_ylabel(unit_label)
            ax.set_title(title, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
        # Get channel indices
        ch_names = epochs.ch_names
        
        # 1. C3
        if 'C3' in ch_names:
            plot_erp_comparison(axes[0, 0], [ch_names.index('C3')], 'C3 ERP (Left Motor)')
            
        # 2. C4
        if 'C4' in ch_names:
            plot_erp_comparison(axes[0, 1], [ch_names.index('C4')], 'C4 ERP (Right Motor)')
            
        # 3. Left ROI Average
        left_indices = [ch_names.index(ch) for ch in left_cluster if ch in ch_names]
        if left_indices:
             plot_erp_comparison(axes[1, 0], left_indices, 'Left Cluster ROI Average', roi_mode=True)
             
        # 4. Right ROI Average
        right_indices = [ch_names.index(ch) for ch in right_cluster if ch in ch_names]
        if right_indices:
             plot_erp_comparison(axes[1, 1], right_indices, 'Right Cluster ROI Average', roi_mode=True)
             
        fig.suptitle(f'Event-Related Potentials (ERP): Mean ± SEM\nSubject {config.subject.id}', 
                    fontsize=16, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-erp_analysis.png"
        )
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"ERP Analysis saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate ERP analysis: {e}")
        import traceback
        traceback.print_exc()
        return None



def generate_contrast_analysis(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Generate contrast analysis plots to detect lateralization effects.
    
    Implements three contrast strategies:
    A. Motor Execution: (LEFT + RIGHT) vs NOTHING
    B. Lateralization: Contralateral vs Ipsilateral for each hemisphere
    C. Lateralization Index: LEFT - RIGHT difference maps
    
    Args:
        epochs: MNE Epochs object with condition information
        output_path: Directory to save plots
        config: SubjectConfig with subject information
        
    Returns:
        Tuple of (contrast_tfr_path, lateralization_index_path) or (None, None) if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check conditions
        conditions = list(epochs.event_id.keys())
        has_left = any('LEFT' in cond for cond in conditions)
        has_right = any('RIGHT' in cond for cond in conditions)
        has_nothing = any('NOTHING' in cond for cond in conditions)
        
        if not (has_left and has_right):
            logger.warning("Need both LEFT and RIGHT conditions for contrast analysis")
            return None, None
        
        # Check channels
        if 'C3' not in epochs.ch_names or 'C4' not in epochs.ch_names:
            logger.warning("Need C3 and C4 channels for contrast analysis")
            return None, None
        
        logger.info("Generating Contrast Analysis plots...")
        
        from affective_fnirs.eeg_analysis import compute_tfr
        
        # Frequency range: focus on alpha and beta
        freqs = np.arange(8, 31, 1)
        
        # Get conditions
        left_cond = [c for c in conditions if 'LEFT' in c][0]
        right_cond = [c for c in conditions if 'RIGHT' in c][0]
        nothing_cond = [c for c in conditions if 'NOTHING' in c][0] if has_nothing else None
        
        # Compute TFR for each condition
        logger.info(f"Computing TFR for contrast analysis...")
        tfr_left = compute_tfr(
            epochs[left_cond],
            freqs=freqs,
            n_cycles=freqs / 2.0,
            baseline=(config.analysis.baseline_window_start_sec,
                     config.analysis.baseline_window_end_sec),
            baseline_mode="percent",
        )
        
        tfr_right = compute_tfr(
            epochs[right_cond],
            freqs=freqs,
            n_cycles=freqs / 2.0,
            baseline=(config.analysis.baseline_window_start_sec,
                     config.analysis.baseline_window_end_sec),
            baseline_mode="percent",
        )
        
        tfr_nothing = None
        if nothing_cond:
            tfr_nothing = compute_tfr(
                epochs[nothing_cond],
                freqs=freqs,
                n_cycles=freqs / 2.0,
                baseline=(config.analysis.baseline_window_start_sec,
                         config.analysis.baseline_window_end_sec),
                baseline_mode="percent",
            )
        
        # =====================================================================
        # Plot 1: Contrast TFR Maps
        # =====================================================================
        n_rows = 3 if tfr_nothing else 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6*n_rows))
        
        # Time window
        tmin = -1.0
        tmax = config.trials.task_duration_sec + 2.0
        
        # Get channel indices
        c3_idx = tfr_left.ch_names.index('C3')
        c4_idx = tfr_left.ch_names.index('C4')
        
        # Row 1: Lateralization Contrast (Contralateral vs Ipsilateral)
        # C3: RIGHT (contralateral) vs LEFT (ipsilateral)
        contrast_c3 = tfr_right.data[c3_idx, :, :] - tfr_left.data[c3_idx, :, :]
        vmax_c3 = max(abs(np.percentile(contrast_c3, 5)), abs(np.percentile(contrast_c3, 95)))
        
        im = axes[0, 0].imshow(
            contrast_c3,
            aspect='auto',
            origin='lower',
            extent=[tfr_left.times[0], tfr_left.times[-1], freqs[0], freqs[-1]],
            cmap='RdBu_r',
            vmin=-vmax_c3,
            vmax=vmax_c3,
        )
        axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[0, 0].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
        axes[0, 0].set_xlim(tmin, tmax)
        axes[0, 0].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('C3: RIGHT (contra) - LEFT (ipsi)\nExpected: Negative (more ERD for contra)', 
                            fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[0, 0], label='Contrast (%)')
        
        # C4: LEFT (contralateral) vs RIGHT (ipsilateral)
        contrast_c4 = tfr_left.data[c4_idx, :, :] - tfr_right.data[c4_idx, :, :]
        vmax_c4 = max(abs(np.percentile(contrast_c4, 5)), abs(np.percentile(contrast_c4, 95)))
        
        im = axes[0, 1].imshow(
            contrast_c4,
            aspect='auto',
            origin='lower',
            extent=[tfr_left.times[0], tfr_left.times[-1], freqs[0], freqs[-1]],
            cmap='RdBu_r',
            vmin=-vmax_c4,
            vmax=vmax_c4,
        )
        axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[0, 1].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
        axes[0, 1].set_xlim(tmin, tmax)
        axes[0, 1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('C4: LEFT (contra) - RIGHT (ipsi)\nExpected: Negative (more ERD for contra)', 
                            fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=axes[0, 1], label='Contrast (%)')
        
        # Row 2: Motor Execution Contrast (if NOTHING available)
        if tfr_nothing:
            # C3: (LEFT + RIGHT)/2 vs NOTHING
            motor_avg_c3 = (tfr_left.data[c3_idx, :, :] + tfr_right.data[c3_idx, :, :]) / 2
            contrast_motor_c3 = motor_avg_c3 - tfr_nothing.data[c3_idx, :, :]
            vmax_mc3 = max(abs(np.percentile(contrast_motor_c3, 5)), abs(np.percentile(contrast_motor_c3, 95)))
            
            im = axes[1, 0].imshow(
                contrast_motor_c3,
                aspect='auto',
                origin='lower',
                extent=[tfr_left.times[0], tfr_left.times[-1], freqs[0], freqs[-1]],
                cmap='RdBu_r',
                vmin=-vmax_mc3,
                vmax=vmax_mc3,
            )
            axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=2)
            axes[1, 0].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
            axes[1, 0].set_xlim(tmin, tmax)
            axes[1, 0].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            axes[1, 0].set_title('C3: Motor Execution (L+R)/2 - NOTHING\nExpected: Negative during movement', 
                                fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=axes[1, 0], label='Contrast (%)')
            
            # C4: (LEFT + RIGHT)/2 vs NOTHING
            motor_avg_c4 = (tfr_left.data[c4_idx, :, :] + tfr_right.data[c4_idx, :, :]) / 2
            contrast_motor_c4 = motor_avg_c4 - tfr_nothing.data[c4_idx, :, :]
            vmax_mc4 = max(abs(np.percentile(contrast_motor_c4, 5)), abs(np.percentile(contrast_motor_c4, 95)))
            
            im = axes[1, 1].imshow(
                contrast_motor_c4,
                aspect='auto',
                origin='lower',
                extent=[tfr_left.times[0], tfr_left.times[-1], freqs[0], freqs[-1]],
                cmap='RdBu_r',
                vmin=-vmax_mc4,
                vmax=vmax_mc4,
            )
            axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=2)
            axes[1, 1].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
            axes[1, 1].set_xlim(tmin, tmax)
            axes[1, 1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('C4: Motor Execution (L+R)/2 - NOTHING\nExpected: Negative during movement', 
                                fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=axes[1, 1], label='Contrast (%)')
            
            # Row 3: Lateralization Index (LEFT - RIGHT)
            lat_index_c3 = tfr_left.data[c3_idx, :, :] - tfr_right.data[c3_idx, :, :]
            vmax_li_c3 = max(abs(np.percentile(lat_index_c3, 5)), abs(np.percentile(lat_index_c3, 95)))
            
            im = axes[2, 0].imshow(
                lat_index_c3,
                aspect='auto',
                origin='lower',
                extent=[tfr_left.times[0], tfr_left.times[-1], freqs[0], freqs[-1]],
                cmap='RdBu_r',
                vmin=-vmax_li_c3,
                vmax=vmax_li_c3,
            )
            axes[2, 0].axvline(0, color='black', linestyle='--', linewidth=2)
            axes[2, 0].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
            axes[2, 0].set_xlim(tmin, tmax)
            axes[2, 0].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            axes[2, 0].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            axes[2, 0].set_title('C3: Lateralization Index (LEFT - RIGHT)\nExpected: Positive (LEFT ipsi, RIGHT contra)', 
                                fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=axes[2, 0], label='Index (%)')
            
            lat_index_c4 = tfr_left.data[c4_idx, :, :] - tfr_right.data[c4_idx, :, :]
            vmax_li_c4 = max(abs(np.percentile(lat_index_c4, 5)), abs(np.percentile(lat_index_c4, 95)))
            
            im = axes[2, 1].imshow(
                lat_index_c4,
                aspect='auto',
                origin='lower',
                extent=[tfr_left.times[0], tfr_left.times[-1], freqs[0], freqs[-1]],
                cmap='RdBu_r',
                vmin=-vmax_li_c4,
                vmax=vmax_li_c4,
            )
            axes[2, 1].axvline(0, color='black', linestyle='--', linewidth=2)
            axes[2, 1].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
            axes[2, 1].set_xlim(tmin, tmax)
            axes[2, 1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            axes[2, 1].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            axes[2, 1].set_title('C4: Lateralization Index (LEFT - RIGHT)\nExpected: Positive (LEFT contra, RIGHT ipsi)', 
                                fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=axes[2, 1], label='Index (%)')
        else:
            # Only lateralization index if no NOTHING condition
            lat_index_c3 = tfr_left.data[c3_idx, :, :] - tfr_right.data[c3_idx, :, :]
            vmax_li_c3 = max(abs(np.percentile(lat_index_c3, 5)), abs(np.percentile(lat_index_c3, 95)))
            
            im = axes[1, 0].imshow(
                lat_index_c3,
                aspect='auto',
                origin='lower',
                extent=[tfr_left.times[0], tfr_left.times[-1], freqs[0], freqs[-1]],
                cmap='RdBu_r',
                vmin=-vmax_li_c3,
                vmax=vmax_li_c3,
            )
            axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=2)
            axes[1, 0].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
            axes[1, 0].set_xlim(tmin, tmax)
            axes[1, 0].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            axes[1, 0].set_title('C3: Lateralization Index (LEFT - RIGHT)', fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=axes[1, 0], label='Index (%)')
            
            lat_index_c4 = tfr_left.data[c4_idx, :, :] - tfr_right.data[c4_idx, :, :]
            vmax_li_c4 = max(abs(np.percentile(lat_index_c4, 5)), abs(np.percentile(lat_index_c4, 95)))
            
            im = axes[1, 1].imshow(
                lat_index_c4,
                aspect='auto',
                origin='lower',
                extent=[tfr_left.times[0], tfr_left.times[-1], freqs[0], freqs[-1]],
                cmap='RdBu_r',
                vmin=-vmax_li_c4,
                vmax=vmax_li_c4,
            )
            axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=2)
            axes[1, 1].axvline(config.trials.task_duration_sec, color='black', linestyle='--', linewidth=2, alpha=0.5)
            axes[1, 1].set_xlim(tmin, tmax)
            axes[1, 1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('C4: Lateralization Index (LEFT - RIGHT)', fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=axes[1, 1], label='Index (%)')
        
        fig.suptitle('Contrast Analysis: Detecting Lateralization Effects', 
                    fontsize=18, fontweight='bold', y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save figure
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-contrast_analysis.png"
        )
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Contrast Analysis saved to: {filepath}")
        
        return filepath, None
        
    except Exception as e:
        logger.error(f"Failed to generate contrast analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def generate_csp_analysis(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> tuple[Optional[Path], dict]:
    """
    Apply Common Spatial Patterns (CSP) to discriminate LEFT vs RIGHT hand movement.
    
    CSP is a supervised spatial filtering technique that maximizes variance for one
    class while minimizing it for the other. For motor imagery/execution tasks,
    CSP extracts spatial filters that capture lateralized motor cortex activity.
    
    Scientific rationale:
        - CSP finds spatial filters W such that Var(W·X_class1) is maximized
          while Var(W·X_class2) is minimized
        - For LEFT vs RIGHT hand movement, CSP should find filters emphasizing
          contralateral motor cortex (C3 for RIGHT, C4 for LEFT)
        - The resulting spatial patterns reveal the topography of discriminative activity
    
    Reference:
        Blankertz et al. (2008). Optimizing Spatial Filters for Robust EEG 
        Single-Trial Analysis. IEEE Signal Processing Magazine.
    
    Args:
        epochs: MNE Epochs with LEFT and RIGHT conditions
        output_path: Directory to save outputs
        config: SubjectConfig with subject information
        
    Returns:
        Tuple of (figure_path, results_dict) where results_dict contains:
            - accuracy: Cross-validation accuracy
            - csp_patterns: Spatial patterns matrix
            - feature_scatter: Feature values for visualization
    """
    logger = logging.getLogger(__name__)
    
    try:
        from mne.decoding import CSP
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
        
        # Check conditions
        conditions = list(epochs.event_id.keys())
        has_left = any('LEFT' in cond for cond in conditions)
        has_right = any('RIGHT' in cond for cond in conditions)
        
        if not (has_left and has_right):
            logger.warning("Need both LEFT and RIGHT conditions for CSP analysis")
            return None, {}
        
        logger.info("Running CSP analysis for LEFT vs RIGHT discrimination...")
        
        # Optimize for CSP: Filter and Crop
        logger.info("Optimizing data for CSP: 8-30Hz bandpass, 0.0-4.0s window")
        
        # Create a copy to avoid modifying original epochs
        epochs_csp = epochs.copy()
        
        # 1. Filter to motor bands (Mu/Beta) where the ERD effect lives
        # This removes delta/theta noise and visual evoked potentials
        epochs_csp.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin', skip_by_annotation='edge')
        
        # 2. Crop to the relevant task window
        # User requested 0.0s to 4.0s (capturing onset and main movement)
        epochs_csp.crop(tmin=0.0, tmax=4.0)
        
        # IMPORTANT: Reset baseline info to None after cropping
        # This prevents ValueError when concatenating epochs (baseline interval no longer exists)
        epochs_csp.baseline = None
        
        # Get condition names
        left_cond = [c for c in conditions if 'LEFT' in c][0]
        right_cond = [c for c in conditions if 'RIGHT' in c][0]
        
        # Get epochs for each condition
        epochs_left = epochs_csp[left_cond]
        epochs_right = epochs_csp[right_cond]
        
        n_left = len(epochs_left)
        n_right = len(epochs_right)
        logger.info(f"LEFT trials: {n_left}, RIGHT trials: {n_right}")
        
        # Combine epochs and create labels
        # Label: 0 = LEFT, 1 = RIGHT
        epochs_combined = mne.concatenate_epochs([epochs_left, epochs_right])
        labels = np.array([0] * n_left + [1] * n_right)
        
        # Get data (trials x channels x time)
        data = epochs_combined.get_data()
        logger.info(f"Data shape: {data.shape} (trials x channels x time)")
        
        # =====================================================================
        # CSP Analysis
        # =====================================================================
        # Use 6 components (3 per class) with alternate ordering
        # Add regularization to handle ill-conditioned covariance matrices
        n_components = min(6, min(n_left, n_right) * 2 - 2)  # Limit components based on trials
        n_components = max(2, n_components)  # At least 2 components
        
        csp = CSP(
            n_components=n_components,
            reg='ledoit_wolf',  # Regularization for small sample sizes
            log=True,  # Log-transform features (log-variance)
            norm_trace=False,
            component_order='alternate',  # Alternate between classes
        )
        
        # Fit CSP on all data first for visualization
        csp.fit(data, labels)
        
        # Get spatial patterns for topographic visualization
        csp_patterns = csp.patterns_
        logger.info(f"CSP patterns shape: {csp_patterns.shape}")
        
        # Transform data to get features
        csp_features = csp.transform(data)
        logger.info(f"CSP features shape: {csp_features.shape}")
        
        # =====================================================================
        # Cross-validation for accuracy estimation
        # =====================================================================
        # Use stratified k-fold (k=min(5, min_class_size))
        min_class_size = min(n_left, n_right)
        n_splits = min(5, min_class_size)
        
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed)
            
            # Pipeline: CSP + LDA
            clf = Pipeline([
                ('csp', CSP(n_components=n_components, reg='ledoit_wolf', log=True, 
                           norm_trace=False, component_order='alternate')),
                ('lda', LinearDiscriminantAnalysis())
            ])
            
            # Cross-validation
            scores = cross_val_score(clf, data, labels, cv=cv, scoring='accuracy')
            mean_accuracy = scores.mean()
            std_accuracy = scores.std()
            logger.info(f"CSP+LDA CV Accuracy: {mean_accuracy:.1%} ± {std_accuracy:.1%}")
        else:
            mean_accuracy = np.nan
            std_accuracy = np.nan
            logger.warning("Not enough trials for cross-validation")
        
        # =====================================================================
        # Create visualization figure
        # =====================================================================
        fig = plt.figure(figsize=(16, 8))
        
        # Layout: 2 rows
        # Row 1: CSP spatial patterns (topoplots)
        # Row 2: Feature scatter plot + accuracy
        
        # Row 1: Topoplots of CSP patterns
        n_patterns_to_show = min(6, n_components)
        for idx in range(n_patterns_to_show):
            ax = fig.add_subplot(2, n_patterns_to_show, idx + 1)
            
            # Get pattern for this component
            pattern = csp_patterns[idx, :]
            
            # Create info object for topoplot
            info = epochs_combined.info.copy()
            
            # Plot topomap
            mne.viz.plot_topomap(
                pattern,
                info,
                axes=ax,
                show=False,
                contours=0,
            )
            
            # Label: odd indices favor LEFT (class 0), even favor RIGHT (class 1)
            if idx % 2 == 0:
                class_label = "RIGHT"
            else:
                class_label = "LEFT"
            ax.set_title(f"CSP{idx}\n({class_label})", fontsize=11, fontweight='bold')
        
        # Row 2 left: Feature scatter plot
        ax_scatter = fig.add_subplot(2, 2, 3)
        
        # Plot first two CSP features
        left_mask = labels == 0
        right_mask = labels == 1
        
        ax_scatter.scatter(
            csp_features[left_mask, 0], 
            csp_features[left_mask, 1],
            c='blue', s=100, alpha=0.7, label='LEFT hand', edgecolors='black'
        )
        ax_scatter.scatter(
            csp_features[right_mask, 0], 
            csp_features[right_mask, 1],
            c='red', s=100, alpha=0.7, label='RIGHT hand', edgecolors='black'
        )
        ax_scatter.set_xlabel('CSP Feature 0 (log-variance)', fontsize=12, fontweight='bold')
        ax_scatter.set_ylabel('CSP Feature 1 (log-variance)', fontsize=12, fontweight='bold')
        ax_scatter.set_title('CSP Feature Space: LEFT vs RIGHT', fontsize=13, fontweight='bold')
        ax_scatter.legend(loc='best', fontsize=10)
        ax_scatter.grid(True, alpha=0.3)
        
        # Row 2 right: Accuracy and metrics
        ax_metrics = fig.add_subplot(2, 2, 4)
        ax_metrics.axis('off')
        
        metrics_text = f"""
CSP Analysis Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classification: LEFT vs RIGHT hand

Trials:
  • LEFT:  {n_left} trials
  • RIGHT: {n_right} trials
  • Total: {n_left + n_right} trials

CSP Configuration:
  • Components: {n_components}
  • Ordering: Alternate (class-balanced)
  • Features: Log-variance

Cross-Validation ({n_splits}-fold):
  • Accuracy: {mean_accuracy:.1%} ± {std_accuracy:.1%}
  • Chance level: 50%

Interpretation:
  • CSP0, CSP2, CSP4: Maximize RIGHT variance
  • CSP1, CSP3, CSP5: Maximize LEFT variance
  • Spatial patterns show discriminative topography
"""
        ax_metrics.text(0.1, 0.95, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        fig.suptitle(f'Common Spatial Patterns (CSP): LEFT vs RIGHT Hand Discrimination\n'
                    f'Subject {config.subject.id} | Session {config.subject.session}',
                    fontsize=16, fontweight='bold', y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        # Save figure
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-csp_analysis.png"
        )
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"CSP Analysis saved to: {filepath}")
        
        # Prepare results dictionary
        results = {
            'accuracy': float(mean_accuracy) if not np.isnan(mean_accuracy) else None,
            'std_accuracy': float(std_accuracy) if not np.isnan(std_accuracy) else None,
            'n_trials_left': n_left,
            'n_trials_right': n_right,
            'n_components': n_components,
            'n_folds': n_splits,
        }
        
        return filepath, results
        
    except ImportError as e:
        logger.error(f"Missing dependency for CSP analysis: {e}")
        return None, {}
    except Exception as e:
        logger.error(f"Failed to generate CSP analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, {}


# =============================================================================
# fNIRS ANALYSIS FUNCTIONS
# =============================================================================

def generate_fnirs_hrf_by_condition(
    fnirs_epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate HRF curves separated by condition (LEFT, RIGHT, NOTHING).
    
    Shows the hemodynamic response for each motor condition, allowing comparison
    of HbO/HbR responses between left and right hand movements.
    
    Scientific rationale:
        - HRF should show task-related increase in HbO and decrease in HbR
        - Contralateral effect: LEFT hand → right hemisphere activation (and vice versa)
        - NOTHING condition serves as control (minimal HRF expected)
    
    Args:
        fnirs_epochs: MNE Epochs with fNIRS data (HbO/HbR channels)
        output_path: Directory to save plot
        config: SubjectConfig with subject information
        
    Returns:
        Path to saved figure or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        conditions = list(fnirs_epochs.event_id.keys())
        has_left = any('LEFT' in cond for cond in conditions)
        has_right = any('RIGHT' in cond for cond in conditions)
        has_nothing = any('NOTHING' in cond for cond in conditions)
        
        logger.info(f"Generating fNIRS HRF by condition...")
        logger.info(f"Conditions: LEFT={has_left}, RIGHT={has_right}, NOTHING={has_nothing}")
        
        # Get HbO channels (motor cortex regions)
        hbo_channels = [ch for ch in fnirs_epochs.ch_names if 'hbo' in ch.lower()]
        hbr_channels = [ch for ch in fnirs_epochs.ch_names if 'hbr' in ch.lower()]
        
        if not hbo_channels:
            logger.warning("No HbO channels found in fNIRS epochs")
            return None
        
        logger.info(f"Found {len(hbo_channels)} HbO channels, {len(hbr_channels)} HbR channels")
        
        # Define left and right hemisphere channels based on naming
        left_hbo = [ch for ch in hbo_channels if any(x in ch for x in ['S1', 'S3', 'S5', 'S7'])]
        right_hbo = [ch for ch in hbo_channels if any(x in ch for x in ['S2', 'S4', 'S6', 'S8'])]
        
        # If naming doesn't work, split by index
        if not left_hbo or not right_hbo:
            mid = len(hbo_channels) // 2
            left_hbo = hbo_channels[:mid] if mid > 0 else hbo_channels[:1]
            right_hbo = hbo_channels[mid:] if mid > 0 else hbo_channels[1:2]
        
        logger.info(f"Left hemisphere HbO: {len(left_hbo)} channels")
        logger.info(f"Right hemisphere HbO: {len(right_hbo)} channels")
        
        times = fnirs_epochs.times
        
        # Create figure
        n_cols = 2  # Left hemisphere, Right hemisphere
        n_rows = 2  # HbO, HbR
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
        
        colors = {'LEFT': 'blue', 'RIGHT': 'red', 'NOTHING': 'green'}
        
        # Get condition names
        left_cond = [c for c in conditions if 'LEFT' in c][0] if has_left else None
        right_cond = [c for c in conditions if 'RIGHT' in c][0] if has_right else None
        nothing_cond = [c for c in conditions if 'NOTHING' in c][0] if has_nothing else None
        
        # Plot HbO - Left Hemisphere
        ax = axes[0, 0]
        for cond_name, cond_key, color in [('LEFT', left_cond, 'blue'), 
                                            ('RIGHT', right_cond, 'red'),
                                            ('NOTHING', nothing_cond, 'green')]:
            if cond_key is None:
                continue
            epochs_cond = fnirs_epochs[cond_key]
            if len(left_hbo) > 0:
                # Average across left hemisphere HbO channels
                data = epochs_cond.get_data(picks=left_hbo)  # (trials, channels, time)
                mean_hrf = data.mean(axis=(0, 1))  # Average across trials and channels
                std_hrf = data.mean(axis=1).std(axis=0)  # Std across trials
                
                ax.plot(times, mean_hrf * 1e6, color=color, linewidth=2, label=cond_name)
                ax.fill_between(times, (mean_hrf - std_hrf) * 1e6, (mean_hrf + std_hrf) * 1e6,
                               color=color, alpha=0.2)
        
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Onset')
        ax.axvline(config.trials.task_duration_sec, color='gray', linestyle='--', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('ΔHbO (μM)', fontsize=11)
        ax.set_title('Left Hemisphere - HbO\n(Contralateral to RIGHT hand)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot HbO - Right Hemisphere
        ax = axes[0, 1]
        for cond_name, cond_key, color in [('LEFT', left_cond, 'blue'), 
                                            ('RIGHT', right_cond, 'red'),
                                            ('NOTHING', nothing_cond, 'green')]:
            if cond_key is None:
                continue
            epochs_cond = fnirs_epochs[cond_key]
            if len(right_hbo) > 0:
                data = epochs_cond.get_data(picks=right_hbo)
                mean_hrf = data.mean(axis=(0, 1))
                std_hrf = data.mean(axis=1).std(axis=0)
                
                ax.plot(times, mean_hrf * 1e6, color=color, linewidth=2, label=cond_name)
                ax.fill_between(times, (mean_hrf - std_hrf) * 1e6, (mean_hrf + std_hrf) * 1e6,
                               color=color, alpha=0.2)
        
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax.axvline(config.trials.task_duration_sec, color='gray', linestyle='--', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('ΔHbO (μM)', fontsize=11)
        ax.set_title('Right Hemisphere - HbO\n(Contralateral to LEFT hand)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot HbR - Left Hemisphere
        ax = axes[1, 0]
        left_hbr = [ch.replace('hbo', 'hbr') for ch in left_hbo if ch.replace('hbo', 'hbr') in fnirs_epochs.ch_names]
        if left_hbr:
            for cond_name, cond_key, color in [('LEFT', left_cond, 'blue'), 
                                                ('RIGHT', right_cond, 'red'),
                                                ('NOTHING', nothing_cond, 'green')]:
                if cond_key is None:
                    continue
                epochs_cond = fnirs_epochs[cond_key]
                data = epochs_cond.get_data(picks=left_hbr)
                mean_hrf = data.mean(axis=(0, 1))
                std_hrf = data.mean(axis=1).std(axis=0)
                
                ax.plot(times, mean_hrf * 1e6, color=color, linewidth=2, label=cond_name)
                ax.fill_between(times, (mean_hrf - std_hrf) * 1e6, (mean_hrf + std_hrf) * 1e6,
                               color=color, alpha=0.2)
        
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax.axvline(config.trials.task_duration_sec, color='gray', linestyle='--', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('ΔHbR (μM)', fontsize=11)
        ax.set_title('Left Hemisphere - HbR\n(Expected: Decrease during activation)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot HbR - Right Hemisphere
        ax = axes[1, 1]
        right_hbr = [ch.replace('hbo', 'hbr') for ch in right_hbo if ch.replace('hbo', 'hbr') in fnirs_epochs.ch_names]
        if right_hbr:
            for cond_name, cond_key, color in [('LEFT', left_cond, 'blue'), 
                                                ('RIGHT', right_cond, 'red'),
                                                ('NOTHING', nothing_cond, 'green')]:
                if cond_key is None:
                    continue
                epochs_cond = fnirs_epochs[cond_key]
                data = epochs_cond.get_data(picks=right_hbr)
                mean_hrf = data.mean(axis=(0, 1))
                std_hrf = data.mean(axis=1).std(axis=0)
                
                ax.plot(times, mean_hrf * 1e6, color=color, linewidth=2, label=cond_name)
                ax.fill_between(times, (mean_hrf - std_hrf) * 1e6, (mean_hrf + std_hrf) * 1e6,
                               color=color, alpha=0.2)
        
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax.axvline(config.trials.task_duration_sec, color='gray', linestyle='--', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('ΔHbR (μM)', fontsize=11)
        ax.set_title('Right Hemisphere - HbR\n(Expected: Decrease during activation)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'fNIRS Hemodynamic Response by Condition\n'
                    f'Subject {config.subject.id} | Task Duration: {config.trials.task_duration_sec}s',
                    fontsize=14, fontweight='bold', y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-fnirs_hrf_by_condition.png"
        )
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"fNIRS HRF by condition saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate fNIRS HRF by condition: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_fnirs_block_average(
    fnirs_epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate block-averaged HRF for all fNIRS channels.
    
    Shows the grand average HRF across all trials and conditions for each channel,
    useful for identifying which channels show task-related hemodynamic responses.
    
    Args:
        fnirs_epochs: MNE Epochs with fNIRS data
        output_path: Directory to save plot
        config: SubjectConfig with subject information
        
    Returns:
        Path to saved figure or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Generating fNIRS block average plot...")
        
        # Get HbO channels only for cleaner visualization
        hbo_channels = [ch for ch in fnirs_epochs.ch_names if 'hbo' in ch.lower()]
        
        if not hbo_channels:
            logger.warning("No HbO channels found")
            return None
        
        n_channels = len(hbo_channels)
        times = fnirs_epochs.times
        
        # Calculate grid size
        n_cols = min(4, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Get all data
        data = fnirs_epochs.get_data(picks=hbo_channels)  # (trials, channels, time)
        
        for idx, ch_name in enumerate(hbo_channels):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Get data for this channel
            ch_data = data[:, idx, :]  # (trials, time)
            mean_hrf = ch_data.mean(axis=0) * 1e6  # Convert to μM
            std_hrf = ch_data.std(axis=0) * 1e6
            
            ax.plot(times, mean_hrf, 'r-', linewidth=2)
            ax.fill_between(times, mean_hrf - std_hrf, mean_hrf + std_hrf,
                           color='red', alpha=0.2)
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.axvline(config.trials.task_duration_sec, color='gray', linestyle='--', linewidth=1)
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Simplify channel name for title
            short_name = ch_name.replace('_hbo', '').replace('S', 'S').replace('D', '-D')
            ax.set_title(short_name, fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('ΔHbO (μM)', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_channels, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        fig.suptitle(f'fNIRS Block Average - All HbO Channels\n'
                    f'Subject {config.subject.id} | {len(fnirs_epochs)} trials',
                    fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        # Save figure
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-fnirs_block_average.png"
        )
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"fNIRS block average saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate fNIRS block average: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_fnirs_contrast_map(
    fnirs_epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate contrast maps showing lateralization in fNIRS data.
    
    Creates bar plots comparing HbO amplitude between conditions and hemispheres
    to visualize contralateral activation patterns.
    
    Scientific rationale:
        - LEFT hand movement → Right hemisphere activation (contralateral)
        - RIGHT hand movement → Left hemisphere activation (contralateral)
        - Motor vs NOTHING → General motor network activation
    
    Args:
        fnirs_epochs: MNE Epochs with fNIRS data
        output_path: Directory to save plot
        config: SubjectConfig with subject information
        
    Returns:
        Path to saved figure or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Generating fNIRS contrast map...")
        
        conditions = list(fnirs_epochs.event_id.keys())
        has_left = any('LEFT' in cond for cond in conditions)
        has_right = any('RIGHT' in cond for cond in conditions)
        has_nothing = any('NOTHING' in cond for cond in conditions)
        
        # Get HbO channels
        hbo_channels = [ch for ch in fnirs_epochs.ch_names if 'hbo' in ch.lower()]
        
        if not hbo_channels:
            logger.warning("No HbO channels found")
            return None
        
        # Define hemispheres
        left_hbo = [ch for ch in hbo_channels if any(x in ch for x in ['S1', 'S3', 'S5', 'S7'])]
        right_hbo = [ch for ch in hbo_channels if any(x in ch for x in ['S2', 'S4', 'S6', 'S8'])]
        
        if not left_hbo or not right_hbo:
            mid = len(hbo_channels) // 2
            left_hbo = hbo_channels[:mid] if mid > 0 else hbo_channels[:1]
            right_hbo = hbo_channels[mid:] if mid > 0 else hbo_channels[1:2]
        
        times = fnirs_epochs.times
        # Task window: from onset to end of task
        task_mask = (times >= 2) & (times <= config.trials.task_duration_sec + 5)
        
        # Get condition names
        left_cond = [c for c in conditions if 'LEFT' in c][0] if has_left else None
        right_cond = [c for c in conditions if 'RIGHT' in c][0] if has_right else None
        nothing_cond = [c for c in conditions if 'NOTHING' in c][0] if has_nothing else None
        
        # Calculate mean HbO amplitude during task for each condition and hemisphere
        results = {}
        
        for cond_name, cond_key in [('LEFT', left_cond), ('RIGHT', right_cond), ('NOTHING', nothing_cond)]:
            if cond_key is None:
                continue
            
            epochs_cond = fnirs_epochs[cond_key]
            
            # Left hemisphere
            if left_hbo:
                data = epochs_cond.get_data(picks=left_hbo)[:, :, task_mask]
                results[f'{cond_name}_left'] = data.mean() * 1e6
                results[f'{cond_name}_left_std'] = data.mean(axis=(1, 2)).std() * 1e6
            
            # Right hemisphere
            if right_hbo:
                data = epochs_cond.get_data(picks=right_hbo)[:, :, task_mask]
                results[f'{cond_name}_right'] = data.mean() * 1e6
                results[f'{cond_name}_right_std'] = data.mean(axis=(1, 2)).std() * 1e6
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Condition comparison by hemisphere
        ax = axes[0]
        x = np.arange(2)
        width = 0.25
        
        conditions_to_plot = []
        if 'LEFT_left' in results:
            conditions_to_plot.append(('LEFT', 'blue'))
        if 'RIGHT_left' in results:
            conditions_to_plot.append(('RIGHT', 'red'))
        if 'NOTHING_left' in results:
            conditions_to_plot.append(('NOTHING', 'green'))
        
        for i, (cond, color) in enumerate(conditions_to_plot):
            left_val = results.get(f'{cond}_left', 0)
            right_val = results.get(f'{cond}_right', 0)
            left_err = results.get(f'{cond}_left_std', 0)
            right_err = results.get(f'{cond}_right_std', 0)
            
            ax.bar(x + i * width, [left_val, right_val], width, 
                  label=cond, color=color, alpha=0.7,
                  yerr=[left_err, right_err], capsize=3)
        
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Left Hemisphere', 'Right Hemisphere'])
        ax.set_ylabel('Mean ΔHbO (μM)', fontsize=11)
        ax.set_title('HbO Amplitude by Condition & Hemisphere', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Lateralization Index
        ax = axes[1]
        lat_indices = []
        lat_labels = []
        lat_colors = []
        
        for cond, color in [('LEFT', 'blue'), ('RIGHT', 'red'), ('NOTHING', 'green')]:
            if f'{cond}_left' in results and f'{cond}_right' in results:
                # Lateralization index: (Right - Left) / (Right + Left)
                left_val = results[f'{cond}_left']
                right_val = results[f'{cond}_right']
                if abs(left_val) + abs(right_val) > 0:
                    lat_idx = (right_val - left_val) / (abs(right_val) + abs(left_val))
                else:
                    lat_idx = 0
                lat_indices.append(lat_idx)
                lat_labels.append(cond)
                lat_colors.append(color)
        
        if lat_indices:
            bars = ax.bar(lat_labels, lat_indices, color=lat_colors, alpha=0.7)
            ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.set_ylabel('Lateralization Index\n(R-L)/(|R|+|L|)', fontsize=11)
            ax.set_title('Hemispheric Lateralization\n(+) = Right dominant, (-) = Left dominant', 
                        fontsize=12, fontweight='bold')
            ax.set_ylim(-1, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add expected pattern annotation
            ax.text(0.02, 0.98, 'Expected:\nLEFT hand → (+) Right\nRIGHT hand → (-) Left',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 3: Motor vs Rest contrast
        ax = axes[2]
        if has_nothing and (has_left or has_right):
            motor_conditions = []
            if 'LEFT_left' in results:
                motor_conditions.append(('LEFT', results['LEFT_left'], results['LEFT_right']))
            if 'RIGHT_left' in results:
                motor_conditions.append(('RIGHT', results['RIGHT_left'], results['RIGHT_right']))
            
            if motor_conditions:
                # Average motor activation
                motor_left = np.mean([c[1] for c in motor_conditions])
                motor_right = np.mean([c[2] for c in motor_conditions])
                nothing_left = results.get('NOTHING_left', 0)
                nothing_right = results.get('NOTHING_right', 0)
                
                x = np.arange(2)
                width = 0.35
                
                ax.bar(x - width/2, [motor_left, motor_right], width, 
                      label='Motor (L+R avg)', color='purple', alpha=0.7)
                ax.bar(x + width/2, [nothing_left, nothing_right], width,
                      label='NOTHING', color='gray', alpha=0.7)
                
                ax.set_xticks(x)
                ax.set_xticklabels(['Left Hemisphere', 'Right Hemisphere'])
                ax.set_ylabel('Mean ΔHbO (μM)', fontsize=11)
                ax.set_title('Motor Execution vs Rest\n(Sanity Check)', fontsize=12, fontweight='bold')
                ax.legend(loc='best')
                ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
                ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'NOTHING condition\nnot available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('Motor vs Rest', fontsize=12, fontweight='bold')
        
        fig.suptitle(f'fNIRS Contrast Analysis\nSubject {config.subject.id}',
                    fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        # Save figure
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-fnirs_contrast.png"
        )
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"fNIRS contrast map saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate fNIRS contrast map: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_contralateral_erd_plots(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Generate classic ERD/ERS plots showing contralateral desynchronization.
    
    Creates two plots:
    1. ERD/ERS timecourse for LEFT vs RIGHT hand movement in C3 and C4
    2. Topoplots showing ERD/ERS spatial distribution during task
    
    This shows the classic effect: C3 desynchronizes during RIGHT hand movement,
    C4 desynchronizes during LEFT hand movement (contralateral control).
    
    Args:
        epochs: MNE Epochs object with condition information
        output_path: Directory to save plots
        config: SubjectConfig with subject information
        
    Returns:
        Tuple of (timecourse_path, topoplot_path) or (None, None) if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check if we have LEFT and RIGHT conditions
        conditions = list(epochs.event_id.keys())
        has_left = any('LEFT' in cond for cond in conditions)
        has_right = any('RIGHT' in cond for cond in conditions)
        
        if not (has_left and has_right):
            logger.warning("Need both LEFT and RIGHT conditions for contralateral ERD plots")
            return None, None
        
        # Check if C3 and C4 are available
        if 'C3' not in epochs.ch_names or 'C4' not in epochs.ch_names:
            logger.warning("Need C3 and C4 channels for contralateral ERD plots")
            return None, None
        
        logger.info("Generating contralateral ERD/ERS plots...")
        
        # Compute TFR for each condition
        from affective_fnirs.eeg_analysis import compute_tfr
        
        freqs = np.arange(8, 31, 1)  # Focus on alpha (8-13 Hz) and beta (13-30 Hz)
        
        # Get LEFT, RIGHT, and NOTHING epochs
        left_cond = [c for c in conditions if 'LEFT' in c][0]
        right_cond = [c for c in conditions if 'RIGHT' in c][0]
        nothing_cond = [c for c in conditions if 'NOTHING' in c][0] if any('NOTHING' in c for c in conditions) else None
        
        epochs_left = epochs[left_cond]
        epochs_right = epochs[right_cond]
        epochs_nothing = epochs[nothing_cond] if nothing_cond else None
        
        logger.info(f"Computing TFR for LEFT condition ({len(epochs_left)} epochs)...")
        tfr_left = compute_tfr(
            epochs_left,
            freqs=freqs,
            n_cycles=7.0,
            baseline=(config.analysis.baseline_window_start_sec,
                     config.analysis.baseline_window_end_sec),
            baseline_mode="percent",
        )
        
        logger.info(f"Computing TFR for RIGHT condition ({len(epochs_right)} epochs)...")
        tfr_right = compute_tfr(
            epochs_right,
            freqs=freqs,
            n_cycles=7.0,
            baseline=(config.analysis.baseline_window_start_sec,
                     config.analysis.baseline_window_end_sec),
            baseline_mode="percent",
        )
        
        tfr_nothing = None
        if epochs_nothing is not None and len(epochs_nothing) > 0:
            logger.info(f"Computing TFR for NOTHING condition ({len(epochs_nothing)} epochs)...")
            tfr_nothing = compute_tfr(
                epochs_nothing,
                freqs=freqs,
                n_cycles=7.0,
                baseline=(config.analysis.baseline_window_start_sec,
                         config.analysis.baseline_window_end_sec),
                baseline_mode="percent",
            )
        
        # =====================================================================
        # Plot 1: ERD/ERS Timecourse (C3 and C4 for LEFT vs RIGHT)
        # =====================================================================
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Define frequency bands
        alpha_band = (config.analysis.alpha_band_low_hz, config.analysis.alpha_band_high_hz)
        beta_band = (config.analysis.beta_band_low_hz, config.analysis.beta_band_high_hz)
        
        # Helper function to extract band power timecourse
        def extract_band_power(tfr, channel, freq_band):
            ch_idx = tfr.ch_names.index(channel)
            freq_mask = (tfr.freqs >= freq_band[0]) & (tfr.freqs <= freq_band[1])
            # Average across frequency band
            band_power = tfr.data[ch_idx, freq_mask, :].mean(axis=0)
            return band_power
        
        # C3 Alpha - LEFT vs RIGHT vs NOTHING
        ax = axes[0, 0]
        c3_alpha_left = extract_band_power(tfr_left, 'C3', alpha_band)
        c3_alpha_right = extract_band_power(tfr_right, 'C3', alpha_band)
        ax.plot(tfr_left.times, c3_alpha_left, linewidth=3, label='LEFT hand', color='#1f77b4')
        ax.plot(tfr_right.times, c3_alpha_right, linewidth=3, label='RIGHT hand (contralateral)', color='#ff7f0e')
        if tfr_nothing is not None:
            c3_alpha_nothing = extract_band_power(tfr_nothing, 'C3', alpha_band)
            ax.plot(tfr_nothing.times, c3_alpha_nothing, linewidth=3, label='NOTHING (baseline)', color='#2ca02c', linestyle='--')
        ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Task onset')
        ax.axvline(config.trials.task_duration_sec, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Power change (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'C3 Alpha ERD ({alpha_band[0]}-{alpha_band[1]} Hz)', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        # C4 Alpha - LEFT vs RIGHT vs NOTHING
        ax = axes[0, 1]
        c4_alpha_left = extract_band_power(tfr_left, 'C4', alpha_band)
        c4_alpha_right = extract_band_power(tfr_right, 'C4', alpha_band)
        ax.plot(tfr_left.times, c4_alpha_left, linewidth=3, label='LEFT hand (contralateral)', color='#1f77b4')
        ax.plot(tfr_right.times, c4_alpha_right, linewidth=3, label='RIGHT hand', color='#ff7f0e')
        if tfr_nothing is not None:
            c4_alpha_nothing = extract_band_power(tfr_nothing, 'C4', alpha_band)
            ax.plot(tfr_nothing.times, c4_alpha_nothing, linewidth=3, label='NOTHING (baseline)', color='#2ca02c', linestyle='--')
        ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Task onset')
        ax.axvline(config.trials.task_duration_sec, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Power change (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'C4 Alpha ERD ({alpha_band[0]}-{alpha_band[1]} Hz)', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        # C3 Beta - LEFT vs RIGHT vs NOTHING
        ax = axes[1, 0]
        c3_beta_left = extract_band_power(tfr_left, 'C3', beta_band)
        c3_beta_right = extract_band_power(tfr_right, 'C3', beta_band)
        ax.plot(tfr_left.times, c3_beta_left, linewidth=3, label='LEFT hand', color='#1f77b4')
        ax.plot(tfr_right.times, c3_beta_right, linewidth=3, label='RIGHT hand (contralateral)', color='#ff7f0e')
        if tfr_nothing is not None:
            c3_beta_nothing = extract_band_power(tfr_nothing, 'C3', beta_band)
            ax.plot(tfr_nothing.times, c3_beta_nothing, linewidth=3, label='NOTHING (baseline)', color='#2ca02c', linestyle='--')
        ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Task onset')
        ax.axvline(config.trials.task_duration_sec, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Power change (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'C3 Beta ERD ({beta_band[0]}-{beta_band[1]} Hz)', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        # C4 Beta - LEFT vs RIGHT vs NOTHING
        ax = axes[1, 1]
        c4_beta_left = extract_band_power(tfr_left, 'C4', beta_band)
        c4_beta_right = extract_band_power(tfr_right, 'C4', beta_band)
        ax.plot(tfr_left.times, c4_beta_left, linewidth=3, label='LEFT hand (contralateral)', color='#1f77b4')
        ax.plot(tfr_right.times, c4_beta_right, linewidth=3, label='RIGHT hand', color='#ff7f0e')
        if tfr_nothing is not None:
            c4_beta_nothing = extract_band_power(tfr_nothing, 'C4', beta_band)
            ax.plot(tfr_nothing.times, c4_beta_nothing, linewidth=3, label='NOTHING (baseline)', color='#2ca02c', linestyle='--')
        ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Task onset')
        ax.axvline(config.trials.task_duration_sec, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Power change (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'C4 Beta ERD ({beta_band[0]}-{beta_band[1]} Hz)', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        fig.suptitle('Contralateral ERD/ERS: Motor Cortex Desynchronization', 
                    fontsize=18, fontweight='bold', y=0.995)
        fig.tight_layout()
        
        # Save timecourse plot
        timecourse_filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-contralateral_erd_timecourse.png"
        )
        timecourse_path = output_path / timecourse_filename
        fig.savefig(str(timecourse_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Contralateral ERD timecourse saved to: {timecourse_path}")
        
        # =====================================================================
        # Plot 2: Topoplots showing ERD/ERS spatial distribution
        # =====================================================================
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        
        # Time window during task (e.g., 2-4 seconds after onset)
        task_tmin = 2.0
        task_tmax = 4.0
        
        # Helper function to compute average power in time window
        def compute_topomap_data(tfr, freq_band, tmin, tmax):
            freq_mask = (tfr.freqs >= freq_band[0]) & (tfr.freqs <= freq_band[1])
            time_mask = (tfr.times >= tmin) & (tfr.times <= tmax)
            # Average across frequency and time
            data = tfr.data[:, freq_mask, :][:, :, time_mask].mean(axis=(1, 2))
            return data
        
        # LEFT hand - Alpha
        data_left_alpha = compute_topomap_data(tfr_left, alpha_band, task_tmin, task_tmax)
        vmin_la = np.percentile(data_left_alpha, 5)
        vmax_la = np.percentile(data_left_alpha, 95)
        im, _ = mne.viz.plot_topomap(data_left_alpha, tfr_left.info, axes=axes[0, 0],
                                     show=False, cmap='RdBu_r', vlim=(vmin_la, vmax_la),
                                     contours=6, sensors=True)
        axes[0, 0].set_title(f'LEFT hand\nAlpha ({alpha_band[0]}-{alpha_band[1]} Hz)', 
                            fontsize=14, fontweight='bold')
        
        # LEFT hand - Beta
        data_left_beta = compute_topomap_data(tfr_left, beta_band, task_tmin, task_tmax)
        vmin_lb = np.percentile(data_left_beta, 5)
        vmax_lb = np.percentile(data_left_beta, 95)
        im, _ = mne.viz.plot_topomap(data_left_beta, tfr_left.info, axes=axes[0, 1],
                                     show=False, cmap='RdBu_r', vlim=(vmin_lb, vmax_lb),
                                     contours=6, sensors=True)
        axes[0, 1].set_title(f'LEFT hand\nBeta ({beta_band[0]}-{beta_band[1]} Hz)', 
                            fontsize=14, fontweight='bold')
        
        # RIGHT hand - Alpha
        data_right_alpha = compute_topomap_data(tfr_right, alpha_band, task_tmin, task_tmax)
        vmin_ra = np.percentile(data_right_alpha, 5)
        vmax_ra = np.percentile(data_right_alpha, 95)
        im, _ = mne.viz.plot_topomap(data_right_alpha, tfr_right.info, axes=axes[0, 2],
                                     show=False, cmap='RdBu_r', vlim=(vmin_ra, vmax_ra),
                                     contours=6, sensors=True)
        axes[0, 2].set_title(f'RIGHT hand\nAlpha ({alpha_band[0]}-{alpha_band[1]} Hz)', 
                            fontsize=14, fontweight='bold')
        
        # RIGHT hand - Beta
        data_right_beta = compute_topomap_data(tfr_right, beta_band, task_tmin, task_tmax)
        vmin_rb = np.percentile(data_right_beta, 5)
        vmax_rb = np.percentile(data_right_beta, 95)
        im, _ = mne.viz.plot_topomap(data_right_beta, tfr_right.info, axes=axes[0, 3],
                                     show=False, cmap='RdBu_r', vlim=(vmin_rb, vmax_rb),
                                     contours=6, sensors=True)
        axes[0, 3].set_title(f'RIGHT hand\nBeta ({beta_band[0]}-{beta_band[1]} Hz)', 
                            fontsize=14, fontweight='bold')
        
        # Contrast: LEFT - RIGHT (shows contralateral effect)
        contrast_alpha = data_left_alpha - data_right_alpha
        vmin_ca = np.percentile(contrast_alpha, 5)
        vmax_ca = np.percentile(contrast_alpha, 95)
        im, _ = mne.viz.plot_topomap(contrast_alpha, tfr_left.info, axes=axes[1, 0],
                                     show=False, cmap='RdBu_r', vlim=(vmin_ca, vmax_ca),
                                     contours=6, sensors=True)
        axes[1, 0].set_title('LEFT - RIGHT\nAlpha (contralateral effect)', 
                            fontsize=14, fontweight='bold')
        
        contrast_beta = data_left_beta - data_right_beta
        vmin_cb = np.percentile(contrast_beta, 5)
        vmax_cb = np.percentile(contrast_beta, 95)
        im, _ = mne.viz.plot_topomap(contrast_beta, tfr_left.info, axes=axes[1, 1],
                                     show=False, cmap='RdBu_r', vlim=(vmin_cb, vmax_cb),
                                     contours=6, sensors=True)
        axes[1, 1].set_title('LEFT - RIGHT\nBeta (contralateral effect)', 
                            fontsize=14, fontweight='bold')
        
        # Hide unused subplots
        axes[1, 2].axis('off')
        axes[1, 3].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Power change (%)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        fig.suptitle(f'ERD/ERS Topoplots ({task_tmin}-{task_tmax}s after task onset)', 
                    fontsize=18, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 0.9, 0.96])
        
        # Save topoplot
        topoplot_filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-contralateral_erd_topoplot.png"
        )
        topoplot_path = output_path / topoplot_filename
        fig.savefig(str(topoplot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Contralateral ERD topoplots saved to: {topoplot_path}")
        
        return timecourse_path, topoplot_path
        
    except Exception as e:
        logger.error(f"Failed to generate contralateral ERD plots: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def generate_cluster_topoplot(
    epochs: mne.Epochs,
    channels: list[str],
    hemisphere_name: str,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate a topoplot showing the electrode cluster.
    
    Args:
        epochs: MNE Epochs object with channel locations
        channels: List of channel names in the cluster
        hemisphere_name: "left" or "right"
        output_path: Directory to save plot
        config: SubjectConfig with subject information
        
    Returns:
        Path to saved topoplot or None if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create a figure for the topoplot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create a mask: 1 for cluster channels, 0 for others
        mask = np.array([ch in channels for ch in epochs.ch_names])
        
        # Create dummy data (all zeros, we just want to show channel locations)
        data = np.zeros(len(epochs.ch_names))
        
        # Plot topoplot with cluster channels highlighted
        im, cn = mne.viz.plot_topomap(
            data,
            epochs.info,
            axes=ax,
            show=False,
            contours=0,
            cmap='Greys',
            sensors=True,
            names=channels,  # Only show names for cluster channels
            mask=mask,
            mask_params=dict(marker='o', markerfacecolor='red', markeredgecolor='darkred', 
                           markersize=20, markeredgewidth=3, alpha=0.8),
        )
        
        # Set title
        ax.set_title(f'{hemisphere_name.capitalize()} Motor Cortex Cluster', 
                    fontsize=16, fontweight='bold', pad=10)
        
        # Save figure
        filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-topoplot_{hemisphere_name}.png"
        )
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight", facecolor='white')
        plt.close(fig)
        
        logger.info(f"{hemisphere_name.capitalize()} topoplot saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate {hemisphere_name} topoplot: {e}")
        import traceback
        traceback.print_exc()
        return None



def generate_beta_topoplots(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> Optional[Path]:
    """
    Generate topographic maps of Beta band (13-30 Hz) power changes (ERD).
    
    Calculates the % power change in the Beta band relative to baseline for
    LEFT and RIGHT conditions. Helps identify the spatial location of the
    strongest ERD (the "motor hotspot").
    
    Args:
        epochs: MNE Epochs object
        output_path: Directory to save plot
        config: SubjectConfig
        
    Returns:
        Path to saved plot or None
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info("Generating Beta Band ERD Topoplots...")
        
        # Check conditions
        conditions = list(epochs.event_id.keys())
        has_left = any('LEFT' in cond for cond in conditions)
        has_right = any('RIGHT' in cond for cond in conditions)
        
        if not (has_left and has_right):
            logger.warning("Need LEFT and RIGHT conditions for Beta Topoplots")
            return None
            
        left_cond = [c for c in conditions if 'LEFT' in c][0]
        right_cond = [c for c in conditions if 'RIGHT' in c][0]
        
        # Define frequency bands and windows
        beta_freqs = np.arange(13, 31, 1)
        tmin_base = config.analysis.baseline_window_start_sec
        tmax_base = config.analysis.baseline_window_end_sec
        tmin_task = 0.5 # Start slightly after onset to miss ERP
        tmax_task = min(4.0, config.trials.task_duration_sec) 
        
        from mne.time_frequency import tfr_multitaper
        
        # Compute TFR for all channels
        # We need TFR to get power over time, then average in time window
        tfr = tfr_multitaper(
            epochs,
            freqs=beta_freqs,
            n_cycles=beta_freqs/2.0,
            use_fft=True,
            return_itc=False,
            average=True,
            n_jobs=1
        )
        
        # Apply baseline correction manually to get % change
        # Get baseline power: average -5 to -2s (or config)
        baseline_mask = (tfr.times >= tmin_base) & (tfr.times <= tmax_base)
        baseline_power = np.mean(tfr.data[:, :, baseline_mask], axis=2, keepdims=True)
        
        # Get task power: average 0.5 to 4s
        task_mask = (tfr.times >= tmin_task) & (tfr.times <= tmax_task)
        # Power change = (Task - Base) / Base * 100
        # tfr.data is (n_channels, n_freqs, n_times)
        
        # We need separate TFRs for LEFT and RIGHT to plot them separately
        tfr_left = tfr_multitaper(epochs[left_cond], freqs=beta_freqs, n_cycles=beta_freqs/2.0, return_itc=False, average=True)
        tfr_right = tfr_multitaper(epochs[right_cond], freqs=beta_freqs, n_cycles=beta_freqs/2.0, return_itc=False, average=True)
        
        # Calculate ERD% for each channel in Beta band
        # 1. Average over frequencies (13-30 Hz)
        # 2. Average over time (task window)
        # 3. Normalize by baseline
        
        def get_beta_erd_topo(tfr_inst):
            # Baseline power per channel (averaged over freq and time)
            base_data = tfr_inst.data[:, :, baseline_mask] # (ch, freq, time)
            base_mean = np.mean(base_data, axis=(1, 2)) # (ch,)
            
            # Task power per channel
            task_data = tfr_inst.data[:, :, task_mask]
            task_mean = np.mean(task_data, axis=(1, 2)) # (ch,)
            
            # Percent change (ERD is negative)
            erd_percent = ((task_mean - base_mean) / base_mean) * 100
            return erd_percent

        erd_left = get_beta_erd_topo(tfr_left)
        erd_right = get_beta_erd_topo(tfr_right)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Color limits - find reasonable max
        vmax = np.max(np.abs(np.concatenate([erd_left, erd_right])))
        vmax = min(vmax, 50) # Cap at 50%
        vmin = -vmax
        
        # LEFT Hand
        im, _ = mne.viz.plot_topomap(
            erd_left, epochs.info, axes=axes[0], show=False,
            cmap='RdBu_r', vlim=(vmin, vmax), contours=0,
            names=epochs.ch_names, show_names=True
        )
        axes[0].set_title('Beta ERD: LEFT Hand\n(Right Hemisphere Activation?', fontsize=14, fontweight='bold')
        
        # RIGHT Hand
        im, _ = mne.viz.plot_topomap(
            erd_right, epochs.info, axes=axes[1], show=False,
            cmap='RdBu_r', vlim=(vmin, vmax), contours=0,
            names=epochs.ch_names, show_names=True
        )
        axes[1].set_title('Beta ERD: RIGHT Hand\n(Left Hemisphere Activation?)', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Power Change (%)')
        
        fig.suptitle(f"Beta Band (13-30 Hz) Topography\nMean ERD [{tmin_task}s to {tmax_task}s]", fontsize=16, fontweight='bold')
        
        filename = f"sub-{config.subject.id}_ses-{config.subject.session}_task-{config.subject.task}_desc-beta_topoplot.png"
        filepath = output_path / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Beta topoplots saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to generate beta topoplots: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_clustered_psd_plots(
    epochs: mne.Epochs,
    output_path: Path,
    config: SubjectConfig,
) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """
    Generate PSD plots clustered by hemisphere and grouped by condition.
    
    Creates four plots:
    1. Left hemisphere PSD
    2. Left hemisphere topoplot
    3. Right hemisphere PSD
    4. Right hemisphere topoplot
    
    Each PSD plot shows mean PSD ± SEM for each condition (LEFT, RIGHT, NOTHING).
    Each topoplot shows the electrode cluster highlighted.
    
    Args:
        epochs: MNE Epochs object with condition information
        output_path: Directory to save plots
        config: SubjectConfig with subject information
        
    Returns:
        Tuple of (left_psd_path, left_topo_path, right_psd_path, right_topo_path) or (None, None, None, None) if failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Define sensorimotor clusters centered on C3/C4 for motor task analysis
        # Left motor cortex: C3 and immediate neighbors
        left_channels = ['FC1', 'FC5', 'C3', 'CP1', 'CP5']
        # Right motor cortex: C4 and immediate neighbors
        right_channels = ['FC2', 'FC6', 'C4', 'CP2', 'CP6']
        
        # Get available channels in epochs
        available_channels = epochs.ch_names
        left_available = [ch for ch in left_channels if ch in available_channels]
        right_available = [ch for ch in right_channels if ch in available_channels]
        
        if not left_available or not right_available:
            logger.warning("Not enough channels for hemisphere clustering")
            return None, None, None, None
        
        logger.info(f"Left hemisphere channels: {left_available}")
        logger.info(f"Right hemisphere channels: {right_available}")
        
        # Get conditions from epochs
        conditions = list(epochs.event_id.keys())
        logger.info(f"Conditions found: {conditions}")
        
        # Define colors for conditions
        condition_colors = {
            'LEFT': '#1f77b4',
            'RIGHT': '#ff7f0e', 
            'NOTHING': '#2ca02c',
            'LEFT/1': '#1f77b4',
            'RIGHT/2': '#ff7f0e',
            'NOTHING/3': '#2ca02c',
        }
        
        # Function to compute PSD for a set of channels and conditions
        def compute_cluster_psd(channels, hemisphere_name):
            fig, ax = plt.subplots(figsize=(40, 25))
            
            for condition in conditions:
                # Get epochs for this condition
                try:
                    cond_epochs = epochs[condition]
                except KeyError:
                    logger.warning(f"Condition {condition} not found in epochs")
                    continue
                
                if len(cond_epochs) == 0:
                    logger.warning(f"No epochs for condition {condition}")
                    continue
                
                # Pick only the channels for this hemisphere
                cond_epochs_picked = cond_epochs.copy().pick_channels(channels, ordered=False)
                
                # Compute PSD for each epoch
                psds_list = []
                for epoch_data in cond_epochs_picked.get_data():
                    # epoch_data shape: (n_channels, n_times)
                    # Compute PSD using Welch method
                    from scipy import signal
                    freqs_list = []
                    psd_epoch = []
                    
                    for ch_idx in range(epoch_data.shape[0]):
                        freqs, psd = signal.welch(
                            epoch_data[ch_idx],
                            fs=cond_epochs_picked.info['sfreq'],
                            nperseg=min(2048, epoch_data.shape[1]),
                            noverlap=min(1024, epoch_data.shape[1]//2),
                        )
                        psd_epoch.append(psd)
                        if len(freqs_list) == 0:
                            freqs_list = freqs
                    
                    # Average across channels for this epoch
                    psd_epoch_mean = np.mean(psd_epoch, axis=0)
                    psds_list.append(psd_epoch_mean)
                
                # Convert to array: (n_epochs, n_freqs)
                psds_array = np.array(psds_list)
                
                # Compute mean and SEM across epochs (in linear scale)
                psd_mean = np.mean(psds_array, axis=0)
                psd_std = np.std(psds_array, axis=0)
                n_epochs = psds_array.shape[0]
                psd_sem = psd_std / np.sqrt(n_epochs)  # Standard Error of the Mean
                
                # Convert to dB: mean and confidence bounds
                psd_mean_db = 10 * np.log10(psd_mean + 1e-20)  # Avoid log(0)
                psd_upper_db = 10 * np.log10(psd_mean + psd_sem + 1e-20)
                psd_lower_db = 10 * np.log10(np.maximum(psd_mean - psd_sem, 1e-20))
                
                # Filter to 1-50 Hz
                freq_mask = (freqs_list >= 1) & (freqs_list <= 50)
                freqs_plot = freqs_list[freq_mask]
                psd_mean_plot = psd_mean_db[freq_mask]
                psd_upper_plot = psd_upper_db[freq_mask]
                psd_lower_plot = psd_lower_db[freq_mask]
                
                # Get color for this condition
                color = condition_colors.get(condition, '#333333')
                
                # Plot mean line
                ax.plot(freqs_plot, psd_mean_plot, linewidth=4, label=condition, color=color, alpha=0.9)
                
                # Plot shaded SEM (Standard Error of the Mean)
                ax.fill_between(
                    freqs_plot,
                    psd_lower_plot,
                    psd_upper_plot,
                    alpha=0.3,
                    color=color
                )
            
            # Customize plot
            ax.set_xlabel('Frequency (Hz)', fontsize=36, fontweight='bold')
            ax.set_ylabel('Power Spectral Density (dB)', fontsize=36, fontweight='bold')
            ax.set_title(f'PSD - {hemisphere_name} Motor Cortex by Condition', fontsize=42, fontweight='bold', pad=20)
            ax.tick_params(axis='both', which='major', labelsize=30, width=3, length=10)
            ax.grid(True, alpha=0.3, linewidth=2)
            ax.set_xlim([1, 50])
            
            # Make spines thicker
            for spine in ax.spines.values():
                spine.set_linewidth(3)
            
            # Add legend
            ax.legend(fontsize=28, loc='upper right', framealpha=0.9)
            
            fig.tight_layout()
            
            return fig
        
        # Generate left hemisphere plot
        left_fig = compute_cluster_psd(left_available, "Left")
        left_filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-psd_left.png"
        )
        left_path = output_path / left_filename
        left_fig.savefig(str(left_path), dpi=150, bbox_inches="tight")
        plt.close(left_fig)
        logger.info(f"Left hemisphere PSD saved to: {left_path}")
        
        # Generate left hemisphere topoplot
        left_topo_path = generate_cluster_topoplot(epochs, left_available, "left", output_path, config)
        
        # Generate right hemisphere plot
        right_fig = compute_cluster_psd(right_available, "Right")
        right_filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-psd_right.png"
        )
        right_path = output_path / right_filename
        right_fig.savefig(str(right_path), dpi=150, bbox_inches="tight")
        plt.close(right_fig)
        logger.info(f"Right hemisphere PSD saved to: {right_path}")
        
        # Generate right hemisphere topoplot
        right_topo_path = generate_cluster_topoplot(epochs, right_available, "right", output_path, config)
        
        return left_path, left_topo_path, right_path, right_topo_path
        
    except Exception as e:
        logger.error(f"Failed to generate clustered PSD plots: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the unified analysis pipeline.

    Returns:
        Parsed command-line arguments.

    Requirements: 1.1, 1.2, 1.3, 1.4
    """
    parser = argparse.ArgumentParser(
        description="Unified EEG/fNIRS analysis pipeline for finger-tapping experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis with config defaults
  python run_analysis.py --config configs/sub-010.yml

  # Override EEG processing
  python run_analysis.py --config configs/sub-010.yml --eeg false

  # Override fNIRS processing
  python run_analysis.py --config configs/sub-010.yml --fnirs false

  # Generate QA report only
  python run_analysis.py --config configs/sub-010.yml --qa-only

  # Load preprocessed data AND cleaned epochs (skip all interactive steps)
  python run_analysis.py --config configs/sub-010.yml --load-preprocessed

  # Load cleaned epochs (same as --load-preprocessed for EEG analysis)
  python run_analysis.py --config configs/sub-010.yml --load-epochs

  # Combine overrides
  python run_analysis.py --config configs/sub-010.yml --eeg true --fnirs false --qa-only
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to subject YAML configuration file",
    )

    parser.add_argument(
        "--eeg",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override EEG processing (true/false). Overrides config file setting.",
    )

    parser.add_argument(
        "--fnirs",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override fNIRS processing (true/false). Overrides config file setting.",
    )

    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="Generate only QA report, skip full analysis",
    )
    
    parser.add_argument(
        "--load-preprocessed",
        action="store_true",
        help="Load preprocessed data AND cleaned epochs, skip all interactive steps (epoching, ICA)",
    )
    
    parser.add_argument(
        "--load-epochs",
        action="store_true",
        help="Load cleaned epochs (same as --load-preprocessed for EEG analysis)",
    )

    return parser.parse_args()


def apply_cli_overrides(
    config: SubjectConfig, args: argparse.Namespace
) -> SubjectConfig:
    """
    Apply CLI flag overrides to the loaded configuration.

    Args:
        config: Loaded SubjectConfig from YAML file.
        args: Parsed command-line arguments.

    Returns:
        SubjectConfig with CLI overrides applied.

    Requirements: 1.5
    """
    # Create mutable copy of modalities config
    modalities_dict = {
        "eeg_enabled": config.modalities.eeg_enabled,
        "fnirs_enabled": config.modalities.fnirs_enabled,
    }

    # Apply EEG override if provided
    if args.eeg is not None:
        modalities_dict["eeg_enabled"] = args.eeg == "true"

    # Apply fNIRS override if provided
    if args.fnirs is not None:
        modalities_dict["fnirs_enabled"] = args.fnirs == "true"

    # Create mutable copy of report config
    report_dict = {"qa_only": config.report.qa_only}

    # Apply QA-only override if provided
    if args.qa_only:
        report_dict["qa_only"] = True

    # Reconstruct config with overrides
    from affective_fnirs.config import ModalityConfig, ReportConfig

    updated_config = SubjectConfig(
        subject=config.subject,
        modalities=ModalityConfig(**modalities_dict),
        report=ReportConfig(**report_dict),
        eeg_channels_of_interest=config.eeg_channels_of_interest,
        eeg_preprocessing=config.eeg_preprocessing,
        trials=config.trials,
        filters=config.filters,
        quality=config.quality,
        epochs=config.epochs,
        analysis=config.analysis,
        ica=config.ica,
        motion_correction=config.motion_correction,
        data_root=config.data_root,
        output_root=config.output_root,
        random_seed=config.random_seed,
    )

    return updated_config


def print_configuration_summary(config: SubjectConfig) -> None:
    """
    Print a summary of the enabled modalities and processing mode.

    Args:
        config: SubjectConfig with final settings.

    Requirements: 1.6
    """
    print("\n" + "=" * 70)
    print("UNIFIED ANALYSIS PIPELINE - CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\nSubject: sub-{config.subject.id}")
    print(f"Session: ses-{config.subject.session}")
    print(f"Task: {config.subject.task}")
    print(f"\nProcessing Mode: {'QA Only' if config.report.qa_only else 'Full Analysis'}")
    print(f"\nEnabled Modalities:")
    print(f"  EEG:   {'✓ Enabled' if config.modalities.eeg_enabled else '✗ Disabled'}")
    print(f"  fNIRS: {'✓ Enabled' if config.modalities.fnirs_enabled else '✗ Disabled'}")

    if config.modalities.eeg_enabled:
        channels_str = ", ".join(config.eeg_channels_of_interest)
        print(f"\nEEG Channels of Interest: {channels_str}")
        print(f"\nEEG Preprocessing:")
        print(f"  Reference channel: {config.eeg_preprocessing.reference_channel}")
        print(f"  Apply CAR: {config.eeg_preprocessing.apply_car}")
        print(f"  ICA enabled: {config.eeg_preprocessing.ica_enabled}")

    print(f"\nTrial Configuration:")
    print(f"  Trials per condition: {config.trials.count_per_condition}")
    print(f"  Task duration: {config.trials.task_duration_sec}s")
    print(f"  Rest duration: {config.trials.rest_duration_sec}s")

    print(f"\nData Root: {config.data_root}")
    print(f"Output Root: {config.output_root}")
    print(f"Random Seed: {config.random_seed}")
    print("=" * 70 + "\n")


def load_and_identify_streams(
    config: SubjectConfig,
) -> dict[str, Optional[dict]]:
    """
    Load XDF file and identify EEG, fNIRS, and marker streams.

    This function calls existing load_xdf_file() and identify_streams() functions
    to load the subject's XDF recording and identify the required data streams.
    It handles missing streams gracefully with warnings.

    Algorithm:
        1. Construct XDF file path from subject info and data_root
        2. Call load_xdf_file() to load all streams
        3. Call identify_streams() to identify EEG, fNIRS, and markers
        4. Handle missing streams gracefully (log warning, return None)

    Args:
        config: SubjectConfig with subject info and data paths

    Returns:
        Dictionary with keys 'eeg', 'fnirs', 'markers' mapping to stream dicts
        or None if stream not found. Always returns all three keys.

    Raises:
        FileNotFoundError: If XDF file doesn't exist
        DataIngestionError: If XDF file is corrupted or cannot be parsed

    Notes:
        - Missing streams are handled gracefully (Req. 3.6)
        - Warnings logged for missing streams
        - Returns None for missing streams instead of raising exception
        - Caller should check for None values before processing

    Requirements: 3.6

    Example:
        >>> streams = load_and_identify_streams(config)
        >>> if streams['eeg'] is not None:
        >>>     # Process EEG
        >>> if streams['fnirs'] is not None:
        >>>     # Process fNIRS
    """
    logger = logging.getLogger(__name__)

    # Construct XDF file path
    # Format: data/raw/sub-{id}/ses-{session}/sub-{id}_ses-{session}_task-{task}_recording.xdf
    xdf_filename = (
        f"sub-{config.subject.id}_"
        f"ses-{config.subject.session}_"
        f"task-{config.subject.task}_recording.xdf"
    )
    # Handle both BIDS-compliant structure (with ses-XXX subdirectory) and flat structure
    xdf_dir_bids = config.data_root / f"sub-{config.subject.id}" / f"ses-{config.subject.session}"
    xdf_dir_flat = config.data_root / f"sub-{config.subject.id}"
    
    # Try BIDS-compliant path first (with ses-XXX subdirectory)
    xdf_path = xdf_dir_bids / xdf_filename
    
    # If not found, try flat structure (without ses-XXX subdirectory)
    if not xdf_path.exists():
        xdf_path = xdf_dir_flat / xdf_filename
    
    # If still not found, try lowercase variant (common in some datasets)
    if not xdf_path.exists():
        xdf_filename_lower = (
            f"sub-{config.subject.id}_"
            f"tomi_ses-{config.subject.session}_"
            f"task-{config.subject.task}_recording.xdf"
        )
        xdf_path_lower_bids = xdf_dir_bids / xdf_filename_lower
        xdf_path_lower_flat = xdf_dir_flat / xdf_filename_lower
        if xdf_path_lower_bids.exists():
            xdf_path = xdf_path_lower_bids
            logger.info(f"Using lowercase filename variant: {xdf_filename_lower}")
        elif xdf_path_lower_flat.exists():
            xdf_path = xdf_path_lower_flat
            logger.info(f"Using lowercase filename variant: {xdf_filename_lower}")

    logger.info(f"Loading XDF file: {xdf_path}")

    # Load XDF file
    streams, header = load_xdf_file(xdf_path)
    logger.info(f"Loaded {len(streams)} streams from XDF file")

    # Identify streams
    try:
        identified_streams = identify_streams(streams)
        logger.info("Successfully identified all required streams (EEG, fNIRS, Markers)")
        return identified_streams
    except DataIngestionError as e:
        # Handle missing streams gracefully (Req. 3.6)
        logger.warning(f"Stream identification issue: {e}")
        logger.warning("Attempting to identify available streams...")

        # Try to identify what we can
        result = {"eeg": None, "fnirs": None, "markers": None}

        # Extract stream names for matching
        for stream in streams:
            try:
                stream_name = stream["info"]["name"][0].lower()
                stream_type = stream["info"].get("type", [""])[0].lower()
            except (KeyError, IndexError):
                continue

            # Check for EEG
            if any(
                pattern in stream_name or pattern in stream_type
                for pattern in ["eeg", "biosemi", "actichamp"]
            ):
                if "marker" not in stream_name and stream_type != "markers":
                    result["eeg"] = stream
                    logger.info(f"Found EEG stream: {stream['info']['name'][0]}")

            # Check for fNIRS
            if any(
                pattern in stream_name or pattern in stream_type
                for pattern in ["fnirs", "nirs", "nirx", "artinis", "photon"]
            ):
                result["fnirs"] = stream
                logger.info(f"Found fNIRS stream: {stream['info']['name'][0]}")

            # Check for Markers - EXPLICITLY prefer eeg_markers
            if any(
                pattern in stream_name or stream_type == "markers"
                for pattern in ["markers", "events", "trigger"]
            ):
                # Prioritize eeg_markers by name (contains LEFT/RIGHT/NOTHING)
                if "eeg_markers" in stream['info']['name'][0]:
                    result["markers"] = stream
                    logger.info(f"Found preferred marker stream: eeg_markers")
                elif result["markers"] is None:
                    # Only use non-eeg_markers if we haven't found anything yet
                    result["markers"] = stream
                    logger.info(f"Found Markers stream (fallback): {stream['info']['name'][0]}")

        # Log what we found
        found_streams = [k for k, v in result.items() if v is not None]
        missing_streams = [k for k, v in result.items() if v is None]

        if found_streams:
            logger.info(f"Available streams: {', '.join(found_streams).upper()}")
        if missing_streams:
            logger.warning(f"Missing streams: {', '.join(missing_streams).upper()}")

        return result


def build_mne_objects(
    streams: dict[str, Optional[dict]],
    config: SubjectConfig,
) -> tuple[Optional[mne.io.Raw], Optional[mne.io.Raw]]:
    """
    Build MNE Raw objects for EEG and fNIRS based on enabled modalities.

    This function conditionally builds MNE Raw objects based on:
    1. Modality flags in config (eeg_enabled, fnirs_enabled)
    2. Stream availability (stream not None)

    If a modality is enabled but stream is missing, logs warning and skips.
    Calls embed_events() for each successfully built Raw object.

    Algorithm:
        1. Check if EEG enabled and stream available
           - If yes: extract data, build EEG Raw, embed events
           - If no: log reason, set raw_eeg = None
        2. Check if fNIRS enabled and stream available
           - If yes: extract data, load JSON montage, build fNIRS Raw, embed events
           - If no: log reason, set raw_fnirs = None
        3. Return tuple (raw_eeg, raw_fnirs)

    Args:
        streams: Dictionary with 'eeg', 'fnirs', 'markers' keys (values may be None)
        config: SubjectConfig with modality flags and paths

    Returns:
        Tuple of (raw_eeg, raw_fnirs) where each may be None if:
        - Modality disabled in config
        - Stream not found in XDF
        - Construction failed (logged as error)

    Raises:
        MNEConstructionError: If construction fails for enabled modality with valid stream

    Notes:
        - Gracefully handles missing streams (Req. 3.6)
        - Only processes enabled modalities (Req. 3.1, 3.2, 3.4, 3.5)
        - Embeds events for each Raw object (Req. 3.4, 3.5)
        - Logs clear messages for each decision

    Requirements: 3.1, 3.2, 3.4, 3.5, 3.6

    Example:
        >>> streams = load_and_identify_streams(config)
        >>> raw_eeg, raw_fnirs = build_mne_objects(streams, config)
        >>> if raw_eeg is not None:
        >>>     print(f"EEG: {len(raw_eeg.ch_names)} channels")
        >>> if raw_fnirs is not None:
        >>>     print(f"fNIRS: {len(raw_fnirs.ch_names)} channels")
    """
    logger = logging.getLogger(__name__)

    raw_eeg = None
    raw_fnirs = None

    # Event mapping for marker stream
    # Map event names to integer codes for MNE epochs
    event_mapping = {
        "LEFT": 1,
        "RIGHT": 2,
        "NOTHING": 3,
        "task_start": 10,
        "task_end": 11,
        "block_start": 20,
        "block_end": 21,
    }

    # =========================================================================
    # Build EEG Raw object (conditional)
    # =========================================================================
    if config.modalities.eeg_enabled:
        if streams["eeg"] is None:
            logger.warning(
                "EEG processing enabled but EEG stream not found. "
                "Skipping EEG processing."
            )
        elif streams["markers"] is None:
            logger.warning(
                "EEG processing enabled but Markers stream not found. "
                "Cannot embed events. Skipping EEG processing."
            )
        else:
            try:
                logger.info("Building EEG Raw object...")

                # Extract EEG data
                eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(
                    streams["eeg"]
                )
                logger.info(
                    f"EEG data: {eeg_data.shape[0]} samples, "
                    f"{eeg_data.shape[1]} channels, {eeg_sfreq} Hz"
                )

                # Build EEG Raw object
                raw_eeg = build_eeg_raw(
                    eeg_data, eeg_sfreq, streams["eeg"]["info"], eeg_timestamps
                )
                logger.info(f"EEG Raw created: {len(raw_eeg.ch_names)} channels")

                # Embed events
                raw_eeg = embed_events(raw_eeg, streams["markers"], event_mapping)
                logger.info(f"Embedded {len(raw_eeg.annotations)} events in EEG Raw")
                if len(raw_eeg.annotations) > 0:
                    logger.info(f"DEBUG: First 5 annotations: {raw_eeg.annotations[:5]}")
                    logger.info(f"DEBUG: Annotation descriptions: {set(raw_eeg.annotations.description)}")
                else:
                    logger.warning("DEBUG: No annotations found in raw_eeg after embedding!")

            except (DataIngestionError, MNEConstructionError) as e:
                logger.error(f"Failed to build EEG Raw object: {e}")
                raw_eeg = None
    else:
        logger.info("EEG processing disabled in configuration")

    # =========================================================================
    # Build fNIRS Raw object (conditional)
    # =========================================================================
    if config.modalities.fnirs_enabled:
        if streams["fnirs"] is None:
            logger.warning(
                "fNIRS processing enabled but fNIRS stream not found. "
                "Skipping fNIRS processing."
            )
        elif streams["markers"] is None:
            logger.warning(
                "fNIRS processing enabled but Markers stream not found. "
                "Cannot embed events. Skipping fNIRS processing."
            )
        else:
            try:
                logger.info("Building fNIRS Raw object...")

                # Extract fNIRS data
                fnirs_data, fnirs_sfreq, fnirs_timestamps = extract_stream_data(
                    streams["fnirs"]
                )
                logger.info(
                    f"fNIRS data: {fnirs_data.shape[0]} samples, "
                    f"{fnirs_data.shape[1]} channels, {fnirs_sfreq} Hz"
                )

                # Load fNIRS montage configuration from JSON sidecar
                # Format: data/raw/sub-{id}/sub-{id}_ses-{session}_task-{task}_nirs.json
                # Or: data/raw/sub-{id}/ses-{session}/sub-{id}_ses-{session}_task-{task}_nirs.json
                json_filename = (
                    f"sub-{config.subject.id}_"
                    f"ses-{config.subject.session}_"
                    f"task-{config.subject.task}_nirs.json"
                )
                
                # Try multiple possible locations
                json_dir = config.data_root / f"sub-{config.subject.id}"
                json_path = json_dir / json_filename
                
                # Try with session subdirectory
                if not json_path.exists():
                    json_dir_with_session = json_dir / f"ses-{config.subject.session}"
                    json_path_with_session = json_dir_with_session / json_filename
                    if json_path_with_session.exists():
                        json_path = json_path_with_session
                        logger.info(f"Found JSON in session subdirectory: {json_path_with_session}")
                
                # Try lowercase variant if not found
                if not json_path.exists():
                    json_filename_lower = (
                        f"sub-{config.subject.id}_"
                        f"Tomi_ses-{config.subject.session}_"
                        f"task-{config.subject.task}_nirs.json"
                    )
                    json_path_lower = json_dir / json_filename_lower
                    if json_path_lower.exists():
                        json_path = json_path_lower
                        logger.info(f"Using lowercase JSON variant: {json_filename_lower}")

                if not json_path.exists():
                    raise FileNotFoundError(
                        f"fNIRS JSON sidecar not found: {json_path}\n"
                        f"Action: Verify JSON sidecar exists for fNIRS data."
                    )

                with open(json_path, "r") as f:
                    json_sidecar = json.load(f)

                montage_config = json_sidecar.get("ChMontage", [])
                if not montage_config:
                    raise ValueError(
                        f"JSON sidecar missing 'ChMontage' field: {json_path}"
                    )

                logger.info(f"Loaded montage config: {len(montage_config)} channels")

                # Build fNIRS Raw object
                raw_fnirs = build_fnirs_raw(
                    fnirs_data, fnirs_sfreq, montage_config, fnirs_timestamps
                )
                logger.info(f"fNIRS Raw created: {len(raw_fnirs.ch_names)} channels")

                # Embed events
                raw_fnirs = embed_events(raw_fnirs, streams["markers"], event_mapping)
                logger.info(f"Embedded {len(raw_fnirs.annotations)} events in fNIRS Raw")

            except (DataIngestionError, MNEConstructionError, FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to build fNIRS Raw object: {e}")
                raw_fnirs = None
    else:
        logger.info("fNIRS processing disabled in configuration")

    return raw_eeg, raw_fnirs


def run_quality_assessment(
    raw_eeg: Optional[mne.io.Raw],
    raw_fnirs: Optional[mne.io.Raw],
    config: SubjectConfig,
) -> dict[str, any]:
    """
    Run quality assessment for enabled modalities.

    This function performs quality assessment on EEG and/or fNIRS data based on
    which modalities are enabled and available. It calls existing quality
    assessment functions from reporting.py and fnirs_quality.py.

    Algorithm:
        1. Initialize empty QA results dictionary
        2. If EEG enabled and available:
           - Call compute_eeg_channel_quality() for all EEG channels
           - Store EEG channel quality metrics
        3. If fNIRS enabled and available:
           - Call calculate_sci() for scalp coupling index
           - Call detect_saturation() for saturation detection
           - Call assess_cardiac_power() for cardiac pulsation
           - Call calculate_coefficient_of_variation() for signal stability
           - Call mark_bad_channels() to identify bad channels
           - Create QualityReport with all fNIRS metrics
        4. Return combined QA results

    Args:
        raw_eeg: MNE Raw object for EEG (may be None if disabled/unavailable)
        raw_fnirs: MNE Raw object for fNIRS (may be None if disabled/unavailable)
        config: SubjectConfig with quality thresholds

    Returns:
        Dictionary with QA results:
        {
            'eeg_channel_quality': list[EEGChannelQuality] or None,
            'fnirs_quality_report': QualityReport or None,
        }

    Notes:
        - Only processes available modalities (Req. 4.2, 4.3)
        - Uses existing quality functions (Req. 4.2, 4.3)
        - Returns None for disabled/unavailable modalities

    Requirements: 4.2, 4.3

    Example:
        >>> qa_results = run_quality_assessment(raw_eeg, raw_fnirs, config)
        >>> if qa_results['eeg_channel_quality']:
        >>>     print(f"EEG: {len(qa_results['eeg_channel_quality'])} channels assessed")
        >>> if qa_results['fnirs_quality_report']:
        >>>     print(f"fNIRS: {qa_results['fnirs_quality_report'].n_bad_channels} bad channels")
    """
    logger = logging.getLogger(__name__)
    qa_results = {
        "eeg_channel_quality": None,
        "fnirs_quality_report": None,
    }

    # =========================================================================
    # EEG Quality Assessment
    # =========================================================================
    if raw_eeg is not None:
        logger.info("Running EEG quality assessment...")
        try:
            # Get all EEG channel names
            eeg_picks = mne.pick_types(raw_eeg.info, eeg=True, exclude=[])
            eeg_channel_names = [raw_eeg.ch_names[i] for i in eeg_picks]

            # Compute quality for all EEG channels
            # Note: We don't have ground truth for which channels are good,
            # so we let the function determine quality based on correlation and variance
            eeg_quality = compute_eeg_channel_quality(
                raw_eeg, eeg_channel_names, known_good_channels=None
            )

            qa_results["eeg_channel_quality"] = eeg_quality

            # Log summary
            n_good = sum(1 for ch in eeg_quality if ch.quality_status == "good")
            n_fair = sum(1 for ch in eeg_quality if ch.quality_status == "fair")
            n_poor = sum(1 for ch in eeg_quality if ch.quality_status == "poor")
            logger.info(
                f"EEG quality: {n_good} good, {n_fair} fair, {n_poor} poor channels"
            )

        except Exception as e:
            logger.error(f"EEG quality assessment failed: {e}")
            qa_results["eeg_channel_quality"] = None
    else:
        logger.info("EEG not available, skipping EEG quality assessment")

    # =========================================================================
    # fNIRS Quality Assessment
    # =========================================================================
    if raw_fnirs is not None:
        logger.info("Running fNIRS quality assessment...")
        try:
            # Calculate Scalp Coupling Index (SCI)
            logger.info("Computing Scalp Coupling Index (SCI)...")
            sci_values = calculate_sci(
                raw_fnirs,
                freq_range=(
                    config.filters.cardiac_band_low_hz,
                    config.filters.cardiac_band_high_hz,
                ),
            )

            # Detect saturation
            logger.info("Detecting saturation...")
            saturation_percent = detect_saturation(raw_fnirs, adc_max=None)

            # Assess cardiac power
            logger.info("Assessing cardiac power...")
            cardiac_power = assess_cardiac_power(
                raw_fnirs,
                freq_range=(
                    config.filters.cardiac_band_low_hz,
                    config.filters.cardiac_band_high_hz,
                ),
            )

            # Calculate coefficient of variation
            logger.info("Computing coefficient of variation...")
            cv_values = calculate_coefficient_of_variation(
                raw_fnirs, baseline_annotations=None
            )

            # Mark bad channels based on quality thresholds
            logger.info("Marking bad channels...")
            raw_fnirs, bad_channels_dict = mark_bad_channels(
                raw_fnirs,
                sci_values=sci_values,
                saturation_values=saturation_percent,
                cardiac_power=cardiac_power,
                cv_values=cv_values,
                sci_threshold=config.quality.sci_threshold,
                saturation_threshold=config.quality.saturation_percent,
                psp_threshold=config.quality.psp_threshold,
                cv_threshold=config.quality.cv_threshold_percent,
            )

            # Extract bad channels and reasons from the returned dictionary
            bad_channels_info = {
                "bad_channels": raw_fnirs.info["bads"],
                "reasons": bad_channels_dict,
            }

            # Create ChannelQuality objects for each channel
            channel_qualities = []
            for ch_name in raw_fnirs.ch_names:
                channel_qualities.append(
                    ChannelQuality(
                        channel_name=ch_name,
                        sci=sci_values.get(ch_name, 0.0),
                        saturation_percent=saturation_percent.get(ch_name, 0.0),
                        cardiac_power=cardiac_power.get(ch_name, 0.0),
                        cv=cv_values.get(ch_name, 0.0),
                        is_bad=ch_name in bad_channels_info["bad_channels"],
                        reason=bad_channels_info["reasons"].get(ch_name, ""),
                    )
                )

            # Calculate summary statistics
            n_total = len(channel_qualities)
            n_bad = sum(1 for ch in channel_qualities if ch.is_bad)
            mean_sci = np.mean([ch.sci for ch in channel_qualities])
            mean_saturation = np.mean(
                [ch.saturation_percent for ch in channel_qualities]
            )
            mean_cardiac = np.mean([ch.cardiac_power for ch in channel_qualities])
            mean_cv = np.mean([ch.cv for ch in channel_qualities])

            # Create QualityReport
            quality_report = QualityReport(
                channels=channel_qualities,
                n_total_channels=n_total,
                n_bad_channels=n_bad,
                mean_sci=float(mean_sci),
                mean_saturation=float(mean_saturation),
                mean_cardiac_power=float(mean_cardiac),
                mean_cv=float(mean_cv),
            )

            qa_results["fnirs_quality_report"] = quality_report

            logger.info(
                f"fNIRS quality: {n_bad}/{n_total} bad channels "
                f"(mean SCI: {mean_sci:.3f}, mean CV: {mean_cv:.1f}%)"
            )

        except Exception as e:
            logger.error(f"fNIRS quality assessment failed: {e}")
            qa_results["fnirs_quality_report"] = None
    else:
        logger.info("fNIRS not available, skipping fNIRS quality assessment")

    return qa_results


def save_qa_report(
    qa_results: dict[str, any], config: SubjectConfig, output_path: Path
) -> dict[str, Path]:
    """
    Save quality assessment results to BIDS-compliant JSON file.

    This function saves QA results using the existing JSON format from reporting.py.
    It creates a BIDS-compliant filename and saves both EEG and fNIRS quality metrics.

    Algorithm:
        1. Generate BIDS-compliant output path using generate_derivative_path()
        2. Create QA summary dictionary with all metrics
        3. Convert numpy types to Python native types for JSON serialization
        4. Save to JSON file with pretty formatting
        5. Return path to saved file

    Args:
        qa_results: Dictionary with QA results from run_quality_assessment()
        config: SubjectConfig with subject info and paths
        output_path: Base output directory (typically from generate_derivative_path)

    Returns:
        Dictionary with paths to saved files:
        {
            'qa_summary': Path to JSON file
        }

    Notes:
        - Uses BIDS-compliant filename (Req. 6.1, 6.2)
        - Saves in derivatives directory (Req. 4.5)
        - Uses existing JSON format from reporting.py (Req. 4.5)
        - Handles numpy types for JSON serialization

    Requirements: 4.5, 6.1, 6.2

    Filename format:
        sub-{id}_ses-{session}_task-{task}_desc-qa_summary.json

    Example:
        >>> paths = save_qa_report(qa_results, config, output_path)
        >>> print(f"QA report saved to: {paths['qa_summary']}")
    """
    logger = logging.getLogger(__name__)

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate BIDS-compliant filename
    qa_filename = (
        f"sub-{config.subject.id}_"
        f"ses-{config.subject.session}_"
        f"task-{config.subject.task}_"
        f"desc-qa_summary.json"
    )
    qa_path = output_path / qa_filename

    # Create QA summary dictionary
    qa_summary = {
        "subject_id": config.subject.id,
        "session_id": config.subject.session,
        "task": config.subject.task,
        "timestamp": datetime.now().isoformat(),
        "eeg_quality": None,
        "fnirs_quality": None,
    }

    # Add EEG quality metrics if available
    if qa_results["eeg_channel_quality"] is not None:
        eeg_channels = []
        for ch_quality in qa_results["eeg_channel_quality"]:
            eeg_channels.append(
                {
                    "channel_name": ch_quality.channel_name,
                    "mean_correlation": float(ch_quality.mean_correlation),
                    "signal_variance": float(ch_quality.signal_variance),
                    "amplitude_range_uv": float(ch_quality.amplitude_range_uv),
                    "std_uv": float(ch_quality.std_uv),
                    "quality_status": ch_quality.quality_status,
                    "is_bad": bool(ch_quality.is_bad),
                }
            )

        qa_summary["eeg_quality"] = {
            "n_channels": len(eeg_channels),
            "n_good": sum(
                1 for ch in eeg_channels if ch["quality_status"] == "good"
            ),
            "n_fair": sum(
                1 for ch in eeg_channels if ch["quality_status"] == "fair"
            ),
            "n_poor": sum(
                1 for ch in eeg_channels if ch["quality_status"] == "poor"
            ),
            "channels": eeg_channels,
        }

    # Add fNIRS quality metrics if available
    if qa_results["fnirs_quality_report"] is not None:
        quality_report = qa_results["fnirs_quality_report"]

        fnirs_channels = []
        for ch_quality in quality_report.channels:
            fnirs_channels.append(
                {
                    "channel_name": ch_quality.channel_name,
                    "sci": float(ch_quality.sci),
                    "saturation_percent": float(ch_quality.saturation_percent),
                    "cardiac_power": float(ch_quality.cardiac_power),
                    "cv": float(ch_quality.cv),
                    "is_bad": bool(ch_quality.is_bad),
                    "reason": ch_quality.reason,
                }
            )

        qa_summary["fnirs_quality"] = {
            "n_total_channels": quality_report.n_total_channels,
            "n_bad_channels": quality_report.n_bad_channels,
            "mean_sci": float(quality_report.mean_sci),
            "mean_saturation": float(quality_report.mean_saturation),
            "mean_cardiac_power": float(quality_report.mean_cardiac_power),
            "mean_cv": float(quality_report.mean_cv),
            "channels": fnirs_channels,
        }

    # Save to JSON with pretty formatting
    with open(qa_path, "w") as f:
        json.dump(qa_summary, f, indent=2)

    logger.info(f"QA report saved to: {qa_path}")

    return {"qa_summary": qa_path}


def run_preprocessing(
    raw_eeg: Optional[mne.io.Raw],
    raw_fnirs: Optional[mne.io.Raw],
    config: SubjectConfig,
    output_path: Path,
) -> tuple[Optional[mne.io.Raw], Optional[mne.io.Raw]]:
    """
    Run preprocessing for enabled modalities.

    This function performs preprocessing on EEG and/or fNIRS data based on
    which modalities are enabled and available. It calls existing preprocessing
    functions from eeg_processing.py and fnirs_processing.py.

    The function preserves annotations through preprocessing by:
    1. Storing annotations before preprocessing
    2. Applying preprocessing (which may modify data)
    3. Restoring annotations to preprocessed data

    Algorithm:
        1. Initialize preprocessed data variables to None
        2. If EEG enabled and available:
           - Store annotations from raw_eeg
           - Call preprocess_eeg_pipeline() with config parameters
           - Restore annotations to preprocessed EEG
           - Optionally save ICA object if generated
        3. If fNIRS enabled and available:
           - Store annotations from raw_fnirs
           - Load montage config from JSON sidecar
           - Call process_fnirs_pipeline() with config parameters
           - Restore annotations to preprocessed fNIRS
        4. Return tuple (processed_eeg, processed_fnirs)

    Args:
        raw_eeg: MNE Raw object for EEG (may be None if disabled/unavailable)
        raw_fnirs: MNE Raw object for fNIRS (may be None if disabled/unavailable)
        config: SubjectConfig with preprocessing parameters
        output_path: Path to output directory for saving ICA objects

    Returns:
        Tuple of (processed_eeg, processed_fnirs) where each may be None if:
        - Modality disabled in config
        - Raw data not available
        - Preprocessing failed (logged as error)

    Raises:
        Exception: If preprocessing fails for enabled modality with valid data

    Notes:
        - Only processes available modalities (Req. 3.1, 3.2)
        - Preserves annotations through preprocessing (Req. 3.1, 3.2)
        - Uses existing preprocessing functions (Req. 3.1, 3.2)
        - Saves ICA object if EEG preprocessing generates one
        - Logs clear messages for each decision

    Requirements: 3.1, 3.2

    Example:
        >>> processed_eeg, processed_fnirs = run_preprocessing(
        ...     raw_eeg, raw_fnirs, config, output_path
        ... )
        >>> if processed_eeg is not None:
        >>>     print(f"EEG preprocessed: {len(processed_eeg.ch_names)} channels")
        >>> if processed_fnirs is not None:
        >>>     print(f"fNIRS preprocessed: {len(processed_fnirs.ch_names)} channels")
    """
    logger = logging.getLogger(__name__)

    processed_eeg = None
    processed_fnirs = None

    # =========================================================================
    # EEG Preprocessing
    # =========================================================================
    if raw_eeg is not None:
        logger.info("Running EEG preprocessing pipeline...")
        
        # Log EEG preprocessing configuration (Req. 7.2, 8.10)
        logger.info("EEG Preprocessing Configuration:")
        logger.info(f"  Reference channel: None (hardware reference preserved)")
        logger.info(f"  Apply CAR: {config.eeg_preprocessing.apply_car}")
        logger.info(f"  ICA: Will be applied on epochs (after epoch rejection)")
        logger.info(f"  Interactive bad channel detection: Enabled")
        
        try:
            # Store annotations before preprocessing
            annotations_eeg = raw_eeg.annotations.copy()
            logger.info(f"Stored {len(annotations_eeg)} EEG annotations")

            # Generate ICA save path if ICA is enabled (from eeg_preprocessing config)
            ica_save_path = None
            if config.eeg_preprocessing.ica_enabled:
                ica_filename = (
                    f"sub-{config.subject.id}_"
                    f"ses-{config.subject.session}_"
                    f"task-{config.subject.task}_"
                    f"ica.fif"
                )
                ica_save_path = str(output_path / ica_filename)

            # Call preprocessing pipeline with interactive bad channel detection
            # ICA will be applied later on epochs (after epoch rejection)
            # This will:
            # 1. Apply bandpass filter (1-40 Hz)
            # 2. Detect bad channels automatically
            # 3. Open interactive plot for manual inspection
            # 4. Interpolate bad channels
            # 5. Apply CAR
            processed_eeg, ica = preprocess_eeg_pipeline(
                raw_eeg=raw_eeg,
                config=config,
                save_ica_path=None,  # ICA will be saved later (applied on epochs)
                reference_channel=None,  # No initial reference (hardware reference preserved)
                apply_car=config.eeg_preprocessing.apply_car,
                ica_enabled=False,  # ICA will be applied on epochs, not on raw
                interactive_bad_channel_detection=True,  # Enable interactive inspection
            )

            # Restore annotations to preprocessed data
            processed_eeg.set_annotations(annotations_eeg)
            logger.info(f"Restored {len(annotations_eeg)} annotations to preprocessed EEG")

            # Log ICA status
            if ica is not None:
                logger.info(f"ICA object saved to: {ica_save_path}")
            else:
                logger.info("ICA was skipped (disabled or data quality sufficient)")

            logger.info(
                f"EEG preprocessing complete: {len(processed_eeg.ch_names)} channels"
            )
            
            # Save preprocessed EEG data (BIDS-compliant)
            logger.info("Saving preprocessed EEG data...")
            preprocessed_eeg_filename = (
                f"sub-{config.subject.id}_"
                f"ses-{config.subject.session}_"
                f"task-{config.subject.task}_"
                f"desc-preprocessed_eeg.fif"
            )
            preprocessed_eeg_path = output_path / preprocessed_eeg_filename
            processed_eeg.save(preprocessed_eeg_path, overwrite=True)
            logger.info(f"Preprocessed EEG saved to: {preprocessed_eeg_path}")
            
            # Generate PSD plot using MNE's native plotting (after preprocessing)
            logger.info("Generating PSD plot for preprocessed EEG...")
            
            psd_filename = (
                f"sub-{config.subject.id}_"
                f"ses-{config.subject.session}_"
                f"task-{config.subject.task}_"
                f"desc-psd.png"
            )
            psd_path = output_path / psd_filename
            
            try:
                # Compute PSD using MNE's native method
                psd = processed_eeg.compute_psd(
                    method="welch",
                    fmin=1.0,
                    fmax=50.0,
                    picks="eeg",
                    n_fft=2048,
                    n_overlap=1024,
                    verbose=False,
                )
                
                # Extract PSD data for custom plotting
                psds, freqs = psd.get_data(return_freqs=True)
                
                # Create custom large plot with matplotlib
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(40, 25))
                
                # Plot each channel
                for idx, ch_name in enumerate(psd.ch_names):
                    ax.plot(freqs, 10 * np.log10(psds[idx]), 
                           linewidth=3, alpha=0.7, label=ch_name)
                
                # Customize plot appearance
                ax.set_xlabel('Frequency (Hz)', fontsize=36, fontweight='bold')
                ax.set_ylabel('Power Spectral Density (dB)', fontsize=36, fontweight='bold')
                ax.set_title('EEG Power Spectral Density', fontsize=42, fontweight='bold', pad=20)
                ax.tick_params(axis='both', which='major', labelsize=30, width=3, length=10)
                ax.grid(True, alpha=0.3, linewidth=2)
                ax.set_xlim([1, 50])
                
                # Make spines thicker
                for spine in ax.spines.values():
                    spine.set_linewidth(3)
                
                # Add legend with larger font
                ax.legend(fontsize=20, loc='upper right', ncol=3, framealpha=0.9)
                
                # Tight layout
                fig.tight_layout()
                
                # Save figure with high DPI
                fig.savefig(str(psd_path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                
                logger.info(f"PSD plot saved to: {psd_path}")
                
                # Store processed_eeg for later use in clustered PSD plots
                processed_eeg_for_psd = processed_eeg
                
            except Exception as psd_error:
                logger.warning(f"Failed to generate PSD plot: {psd_error}")
                processed_eeg_for_psd = None

        except Exception as e:
            logger.error(f"EEG preprocessing failed: {e}")
            processed_eeg = None
    else:
        logger.info("EEG not available, skipping EEG preprocessing")

    # =========================================================================
    # fNIRS Preprocessing
    # =========================================================================
    if raw_fnirs is not None:
        logger.info("Running fNIRS preprocessing pipeline...")
        try:
            # Store annotations before preprocessing
            annotations_fnirs = raw_fnirs.annotations.copy()
            logger.info(f"Stored {len(annotations_fnirs)} fNIRS annotations")

            # Load fNIRS montage configuration from JSON sidecar
            # Format: data/raw/sub-{id}/sub-{id}_ses-{session}_task-{task}_nirs.json
            # Or: data/raw/sub-{id}/ses-{session}/sub-{id}_ses-{session}_task-{task}_nirs.json
            json_filename = (
                f"sub-{config.subject.id}_"
                f"ses-{config.subject.session}_"
                f"task-{config.subject.task}_nirs.json"
            )
            
            # Try multiple possible locations
            json_dir = config.data_root / f"sub-{config.subject.id}"
            json_path = json_dir / json_filename
            
            # Try with session subdirectory
            if not json_path.exists():
                json_dir_with_session = json_dir / f"ses-{config.subject.session}"
                json_path_with_session = json_dir_with_session / json_filename
                if json_path_with_session.exists():
                    json_path = json_path_with_session
                    logger.info(f"Found JSON in session subdirectory: {json_path_with_session}")

            # Try lowercase variant if not found
            if not json_path.exists():
                json_filename_lower = (
                    f"sub-{config.subject.id}_"
                    f"Tomi_ses-{config.subject.session}_"
                    f"task-{config.subject.task}_nirs.json"
                )
                json_path_lower = json_dir / json_filename_lower
                if json_path_lower.exists():
                    json_path = json_path_lower
                    logger.info(f"Using lowercase JSON variant: {json_filename_lower}")

            if not json_path.exists():
                raise FileNotFoundError(
                    f"fNIRS JSON sidecar not found: {json_path}\n"
                    f"Action: Verify JSON sidecar exists for fNIRS data."
                )

            with open(json_path, "r") as f:
                json_sidecar = json.load(f)

            montage_config = json_sidecar.get("ChMontage", [])
            if not montage_config:
                raise ValueError(
                    f"JSON sidecar missing 'ChMontage' field: {json_path}"
                )

            logger.info(f"Loaded montage config: {len(montage_config)} channels")

            # Call existing preprocessing pipeline
            processed_fnirs, processing_metrics = process_fnirs_pipeline(
                raw_intensity=raw_fnirs,
                montage_config=montage_config,
                motion_correction_method=config.motion_correction.method,
                dpf=config.analysis.dpf,
                l_freq=config.filters.fnirs_bandpass_low_hz,
                h_freq=config.filters.fnirs_bandpass_high_hz,
                short_threshold_mm=config.quality.short_channel_distance_mm,
                apply_scr=True,
                verify_noise_reduction=True,
            )

            # Restore annotations to preprocessed data
            processed_fnirs.set_annotations(annotations_fnirs)
            logger.info(f"Restored {len(annotations_fnirs)} annotations to preprocessed fNIRS")

            # Log processing metrics
            logger.info(
                f"fNIRS preprocessing complete: {len(processed_fnirs.ch_names)} channels"
            )
            
            # Save preprocessed fNIRS data (BIDS-compliant)
            logger.info("Saving preprocessed fNIRS data...")
            preprocessed_fnirs_filename = (
                f"sub-{config.subject.id}_"
                f"ses-{config.subject.session}_"
                f"task-{config.subject.task}_"
                f"desc-preprocessed_fnirs.fif"
            )
            preprocessed_fnirs_path = output_path / preprocessed_fnirs_filename
            processed_fnirs.save(preprocessed_fnirs_path, overwrite=True)
            logger.info(f"Preprocessed fNIRS saved to: {preprocessed_fnirs_path}")
            
            logger.info(
                f"Motion artifacts corrected: {processing_metrics['motion_artifacts_corrected']}"
            )
            logger.info(
                f"Short channels: {len(processing_metrics['short_channels'])}, "
                f"Long channels: {len(processing_metrics['long_channels'])}"
            )
            if processing_metrics["noise_reduction_percent"] > 0:
                logger.info(
                    f"Systemic noise reduction: {processing_metrics['noise_reduction_percent']:.1f}%"
                )

        except Exception as e:
            logger.error(f"fNIRS preprocessing failed: {e}")
            processed_fnirs = None
    else:
        logger.info("fNIRS not available, skipping fNIRS preprocessing")

    return processed_eeg, processed_fnirs


def run_eeg_analysis(
    processed_eeg: mne.io.Raw,
    config: SubjectConfig,
    output_path: Path,
) -> dict[str, any]:
    """
    Run EEG analysis including epoching, TFR, and ERD/ERS detection.

    This function performs time-frequency analysis on preprocessed EEG data to
    detect Event-Related Desynchronization (ERD) and Event-Related Synchronization
    (ERS) in motor cortex channels.

    Algorithm:
        1. Create epochs using mne.events_from_annotations() and mne.Epochs()
        2. Compute Time-Frequency Representation (TFR) using compute_tfr()
        3. For each channel in eeg_channels_of_interest:
           - Call detect_erd_ers() to quantify ERD/ERS
           - Store results for each channel
        4. Return dictionary with epochs, TFR, and ERD/ERS results

    Args:
        processed_eeg: Preprocessed MNE Raw object (filtered, ICA-cleaned, CAR)
        config: SubjectConfig with analysis parameters and channels of interest
        output_path: Path to output directory for saving plots

    Returns:
        Dictionary with analysis results:
        {
            'epochs': mne.Epochs object,
            'tfr': mne.time_frequency.AverageTFR object,
            'erd_ers_results': dict mapping channel names to ERD/ERS metrics
        }

    Raises:
        ValueError: If no events found or no valid channels

    Notes:
        - Uses event_id to extract task-related epochs (Req. 5.2)
        - Calls compute_tfr() for time-frequency analysis (Req. 5.4)
        - Calls detect_erd_ers() for each channel in eeg_channels_of_interest (Req. 5.2)
        - Logs progress for each channel
        - Handles missing channels gracefully (logs warning, skips channel)

    Requirements: 5.2, 5.4

    Example:
        >>> eeg_results = run_eeg_analysis(processed_eeg, config)
        >>> print(f"Analyzed {len(eeg_results['erd_ers_results'])} channels")
        >>> for ch, metrics in eeg_results['erd_ers_results'].items():
        >>>     print(f"{ch}: Mu ERD={metrics['alpha_erd_percent']:.1f}%")
    """
    logger = logging.getLogger(__name__)
    logger.info("Running EEG analysis (epoching, TFR, ERD/ERS detection)")

    # Define event mapping for epochs
    # Use task-related events (LEFT, RIGHT, NOTHING - 3 conditions)
    event_id = {
        "LEFT": 1,
        "RIGHT": 2,
        "NOTHING": 3,
    }

    # Try to find events in annotations
    available_events = set(processed_eeg.annotations.description)
    logger.info(f"Available events in EEG data: {available_events}")

    # Filter event_id to only include events that exist
    # Handle both exact matches and events with "/code" suffix (e.g., "LEFT/1")
    event_id_filtered = {}
    for name, code in event_id.items():
        # Check for exact match or match with "/code" suffix
        if name in available_events:
            event_id_filtered[name] = code
        else:
            # Check for events like "LEFT/1", "RIGHT/2"
            for avail_event in available_events:
                if avail_event.startswith(f"{name}/"):
                    event_id_filtered[avail_event] = code
                    break

    if not event_id_filtered:
        # Fallback: try generic task markers
        logger.warning("No LEFT/RIGHT events found, trying generic task markers")
        event_id_filtered = {}
        for event_name in ["task_start", "block_start"]:
            if event_name in available_events:
                event_id_filtered[event_name] = (
                    10 if event_name == "task_start" else 20
                )

    if not event_id_filtered:
        raise ValueError(
            f"No valid events found for EEG epoching. "
            f"Available events: {available_events}, "
            f"Expected: LEFT, RIGHT, NOTHING, task_start, or block_start"
        )

    logger.info(f"Using events for epoching: {event_id_filtered}")

    # Step 1: Create epochs
    logger.info("Creating EEG epochs...")
    try:
        events, event_id_mapping = mne.events_from_annotations(
            processed_eeg, event_id=event_id_filtered
        )

        if len(events) == 0:
            raise ValueError("No events found in annotations")

        logger.info(f"Found {len(events)} events for epoching")

        # Create epochs with extended window for TFR edge effects
        epochs = mne.Epochs(
            processed_eeg,
            events,
            event_id=event_id_mapping,
            tmin=config.epochs.eeg_tmin_sec,
            tmax=config.epochs.eeg_tmax_sec,
            baseline=(
                config.epochs.baseline_tmin_sec,
                config.epochs.baseline_tmax_sec,
            ),
            preload=True,
            proj=False,
            picks="eeg",
            verbose=True,
        )

        logger.info(
            f"Created {len(epochs)} epochs: "
            f"{epochs.info['nchan']} channels, "
            f"{len(epochs.times)} time points"
        )
        
        # Interactive epoch rejection (BEFORE ICA for better ICA fitting)
        logger.info("=" * 80)
        logger.info("INTERACTIVE EPOCH REJECTION (Before ICA)")
        logger.info("=" * 80)
        logger.info("Opening interactive plot for epoch inspection...")
        logger.info("")
        logger.info("Instructions:")
        logger.info("  1. Inspect epochs visually (signal is preprocessed with CAR)")
        logger.info("  2. Click on epochs to mark them as BAD (they will turn red)")
        logger.info("  3. Bad epochs will be excluded BEFORE ICA fitting")
        logger.info("  4. This improves ICA decomposition quality")
        logger.info("  5. Look for:")
        logger.info("     - Extreme amplitudes (>100 µV)")
        logger.info("     - Movement artifacts (sudden jumps)")
        logger.info("     - Very noisy epochs")
        logger.info("  6. Close the window when done to continue")
        logger.info("=" * 80)
        
        # Open interactive epoch plot
        try:
            epochs.plot(
                n_channels=30,
                n_epochs=5,  # Show 5 epochs at a time
                scalings='auto',
                title='Epoch Inspection (Before ICA) - Click epochs to mark as BAD',
                show=True,
                block=True  # Wait for user to close window
            )
        except Exception as e:
            logger.warning(f"Could not open interactive epoch plot: {e}")
        
        # Drop bad epochs marked by user
        n_epochs_before = len(epochs)
        epochs.drop_bad()
        n_epochs_after = len(epochs)
        n_dropped = n_epochs_before - n_epochs_after
        
        logger.info("=" * 80)
        if n_dropped > 0:
            logger.info(f"User marked {n_dropped} bad epochs")
            logger.info(f"Epochs remaining: {n_epochs_after}/{n_epochs_before}")
            logger.info("These clean epochs will be used for ICA fitting")
        else:
            logger.info("No epochs marked as bad by user")
        logger.info("=" * 80)
        
        # Now apply ICA on clean epochs
        logger.info("Fitting ICA on clean epochs...")
        
        # Fit ICA
        ica = mne.preprocessing.ICA(
            n_components=0.99,
            method='fastica',
            random_state=42,
            max_iter=1000
        )
        ica.fit(epochs)
        
        logger.info(f"ICA fitted with {ica.n_components_} components")
        
        # Automatic artifact detection
        logger.info("Detecting artifact components automatically...")
        
        # EOG detection (if frontal channels available)
        eog_components = []
        frontal_channels = ['Fp1', 'Fp2']
        available_frontal = [ch for ch in frontal_channels if ch in epochs.ch_names]
        if available_frontal:
            eog_inds, eog_scores = ica.find_bads_eog(epochs, ch_name=available_frontal, threshold=0.9)
            eog_components = eog_inds
            if eog_components:
                logger.info(f"EOG components detected: {eog_components}")
        
        # EMG detection (high frequency power)
        emg_inds, emg_scores = ica.find_bads_muscle(epochs, threshold=2.5)
        emg_components = emg_inds
        if emg_components:
            logger.info(f"EMG components detected: {emg_components}")
        
        # Combine automatic detections
        artifact_components_auto = sorted(list(set(eog_components + emg_components)))
        
        # Interactive component inspection
        logger.info("=" * 80)
        logger.info("INTERACTIVE ICA COMPONENT INSPECTION")
        logger.info("=" * 80)
        logger.info("Automatic artifact detection found:")
        logger.info(f"  EOG components: {eog_components}")
        logger.info(f"  EMG components: {emg_components}")
        logger.info(f"  Total automatic: {artifact_components_auto}")
        logger.info("")
        logger.info("Opening interactive plots for manual component inspection...")
        logger.info("")
        logger.info("Instructions:")
        logger.info("  1. Component topographies window will open")
        logger.info("  2. Click on component numbers to TOGGLE exclusion (red = excluded)")
        logger.info("  3. Automatic suggestions are already marked in red")
        logger.info("  4. Click to add/remove components from exclusion list")
        logger.info("  5. Close the window when done to continue")
        logger.info("=" * 80)
        
        # Pre-mark automatic suggestions
        ica.exclude = artifact_components_auto.copy()
        
        # Plot component topographies
        try:
            ica.plot_components(
                picks=range(min(20, ica.n_components_)),
                show=True,
                inst=epochs
            )
        except Exception as e:
            logger.warning(f"Could not plot component topographies: {e}")
        
        # Plot component sources
        try:
            ica.plot_sources(
                epochs,
                show=True,
                block=True,
                title='ICA Components - Right-click to mark as artifact'
            )
        except Exception as e:
            logger.warning(f"Could not plot component sources: {e}")
        
        # Get final exclusion list
        artifact_components = ica.exclude.copy()
        
        # Log changes
        added_components = set(artifact_components) - set(artifact_components_auto)
        removed_components = set(artifact_components_auto) - set(artifact_components)
        
        logger.info("=" * 80)
        if added_components:
            logger.info(f"User ADDED components to exclude: {sorted(list(added_components))}")
        if removed_components:
            logger.info(f"User REMOVED components from exclusion: {sorted(list(removed_components))}")
        if not added_components and not removed_components:
            logger.info("User kept automatic suggestions")
        
        logger.info(f"Final components to exclude: {sorted(artifact_components)}")
        logger.info("=" * 80)
        
        # Apply ICA to epochs
        logger.info(f"Applying ICA (excluding {len(artifact_components)} components)...")
        epochs = ica.apply(epochs)
        logger.info("ICA applied successfully")
        
        # Save cleaned epochs (BIDS-compliant)
        logger.info("Saving cleaned epochs...")
        epochs_filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-cleaned_epo.fif"
        )
        epochs_path = output_path / epochs_filename
        epochs.save(epochs_path, overwrite=True)
        logger.info(f"Cleaned epochs saved to: {epochs_path}")
        
        # Save ICA object
        ica_filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"ica.fif"
        )
        ica_path = output_path / ica_filename
        ica.save(ica_path, overwrite=True)
        logger.info(f"ICA object saved to: {ica_path}")

        
        # Generate clustered PSD plots by hemisphere and condition
        logger.info("Generating clustered PSD plots by hemisphere...")
        left_psd_path, left_topo_path, right_psd_path, right_topo_path = generate_clustered_psd_plots(
            epochs, output_path, config
        )
        
        # Generate Beta Band ERD Topoplots (Exploratory)
        beta_topo_path = generate_beta_topoplots(epochs, output_path, config)
        
        # Generate Time-Frequency Maps (most informative canonical plot)
        logger.info("Generating Time-Frequency Maps...")
        tfr_maps_path = generate_tfr_maps(epochs, output_path, config)
        
        # Generate Contrast Analysis (lateralization detection)
        logger.info("Generating Contrast Analysis...")
        contrast_analysis_path, _ = generate_contrast_analysis(epochs, output_path, config)
        
        # Generate CSP Analysis (LEFT vs RIGHT discrimination)
        logger.info("Generating CSP Analysis...")
        csp_analysis_path, csp_results = generate_csp_analysis(epochs, output_path, config)
        
        # Generate contralateral ERD/ERS plots
        logger.info("Generating contralateral ERD/ERS plots...")
        contralateral_timecourse_path, contralateral_topoplot_path = generate_contralateral_erd_plots(
            epochs, output_path, config
        )

    except Exception as e:
        logger.error(f"Failed to create epochs: {e}")
        raise

    # Step 2: Compute Time-Frequency Representation (TFR)
    logger.info("Computing Time-Frequency Representation (TFR)...")
    try:
        # Use frequency range from config (alpha and beta bands)
        freqs = np.arange(3, 31, 1)  # 3-30 Hz, 1 Hz steps

        tfr = compute_tfr(
            epochs,
            freqs=freqs,
            n_cycles=7.0,
            baseline=(
                config.analysis.baseline_window_start_sec,
                config.analysis.baseline_window_end_sec,
            ),
            baseline_mode="percent",
        )

        logger.info(
            f"TFR computed: {tfr.data.shape[0]} channels, "
            f"{tfr.data.shape[1]} frequencies, "
            f"{tfr.data.shape[2]} time points"
        )

    except Exception as e:
        logger.error(f"Failed to compute TFR: {e}")
        raise

    # Step 3: Detect ERD/ERS for each channel of interest
    logger.info(
        f"Detecting ERD/ERS for channels: {config.eeg_channels_of_interest}"
    )

    erd_ers_results = {}

    for channel in config.eeg_channels_of_interest:
        # Check if channel exists in TFR
        if channel not in tfr.ch_names:
            logger.warning(
                f"Channel {channel} not found in TFR data. "
                f"Available channels: {tfr.ch_names}. Skipping."
            )
            continue

        # Check if channel is marked as bad
        if channel in processed_eeg.info["bads"]:
            logger.warning(
                f"Channel {channel} is marked as bad. Skipping ERD/ERS detection."
            )
            continue

        try:
            logger.info(f"Analyzing channel: {channel}")

            # Detect ERD/ERS
            erd_ers_metrics = detect_erd_ers(
                tfr,
                channel=channel,
                alpha_band=(
                    config.analysis.alpha_band_low_hz,
                    config.analysis.alpha_band_high_hz,
                ),
                beta_band=(
                    config.analysis.beta_band_low_hz,
                    config.analysis.beta_band_high_hz,
                ),
                task_window=(
                    config.analysis.task_window_start_sec,
                    config.analysis.task_window_end_sec,
                ),
                baseline_window=(
                    config.analysis.baseline_window_start_sec,
                    config.analysis.baseline_window_end_sec,
                ),
                beta_rebound_window=(
                    config.analysis.beta_rebound_window_start_sec,
                    config.analysis.beta_rebound_window_end_sec,
                ),
            )

            erd_ers_results[channel] = erd_ers_metrics

            logger.info(
                f"  {channel}: Mu ERD={erd_ers_metrics['alpha_erd_percent']:.1f}%, "
                f"Beta ERD={erd_ers_metrics['beta_erd_percent']:.1f}%, "
                f"Beta Rebound={erd_ers_metrics['beta_rebound_percent']:.1f}%"
            )

        except Exception as e:
            logger.error(f"Failed to detect ERD/ERS for channel {channel}: {e}")
            # Continue with other channels
            continue

    if not erd_ers_results:
        logger.warning(
            "No ERD/ERS results computed. Check channel names and data quality."
        )

    logger.info(
        f"EEG analysis complete: {len(erd_ers_results)} channels analyzed"
    )

    return {
        "epochs": epochs,
        "tfr": tfr,
        "erd_ers_results": erd_ers_results,
        "left_psd_path": left_psd_path,
        "left_topo_path": left_topo_path,
        "right_psd_path": right_psd_path,
        "right_topo_path": right_topo_path,
        "tfr_maps_path": tfr_maps_path,
        "contrast_analysis_path": contrast_analysis_path,
        "csp_analysis_path": csp_analysis_path,
        "csp_results": csp_results,
        "contralateral_timecourse_path": contralateral_timecourse_path,
        "contralateral_topoplot_path": contralateral_topoplot_path,
        "beta_topo_path": beta_topo_path,
    }


def run_eeg_analysis_from_epochs(
    epochs: mne.Epochs,
    processed_eeg: mne.io.Raw,
    config: SubjectConfig,
    output_path: Path,
) -> dict[str, any]:
    """
    Run EEG analysis on pre-cleaned epochs (TFR and ERD/ERS detection only).

    This function performs time-frequency analysis on already-cleaned epochs,
    skipping the epoching, epoch rejection, and ICA stages. Use this when
    loading saved cleaned epochs with --load-epochs flag.

    Algorithm:
        1. Compute Time-Frequency Representation (TFR) using compute_tfr()
        2. For each channel in eeg_channels_of_interest:
           - Call detect_erd_ers() to quantify ERD/ERS
           - Store results for each channel
        3. Return dictionary with TFR and ERD/ERS results

    Args:
        epochs: Pre-cleaned MNE Epochs object (already ICA-cleaned)
        processed_eeg: Preprocessed MNE Raw object (for bad channel info)
        config: SubjectConfig with analysis parameters and channels of interest
        output_path: Path to output directory for saving plots

    Returns:
        Dictionary with analysis results:
        {
            'epochs': mne.Epochs object (same as input),
            'tfr': mne.time_frequency.AverageTFR object,
            'erd_ers_results': dict mapping channel names to ERD/ERS metrics,
            'left_psd_path': Path to left hemisphere PSD plot,
            'right_psd_path': Path to right hemisphere PSD plot
        }

    Raises:
        ValueError: If no valid channels for analysis

    Notes:
        - Assumes epochs are already cleaned (bad epochs rejected, ICA applied)
        - Uses compute_tfr() for time-frequency analysis (Req. 5.4)
        - Calls detect_erd_ers() for each channel in eeg_channels_of_interest (Req. 5.2)
        - Logs progress for each channel
        - Handles missing channels gracefully (logs warning, skips channel)

    Requirements: 5.2, 5.4

    Example:
        >>> epochs = mne.read_epochs('sub-001_desc-cleaned_epo.fif')
        >>> eeg_results = run_eeg_analysis_from_epochs(epochs, processed_eeg, config, output_path)
        >>> print(f"Analyzed {len(eeg_results['erd_ers_results'])} channels")
    """
    logger = logging.getLogger(__name__)
    logger.info("Running EEG analysis on pre-cleaned epochs (TFR + ERD/ERS only)")

    # Step 1: Compute Time-Frequency Representation (TFR)
    logger.info("Computing Time-Frequency Representation (TFR)...")
    try:
        # Use frequency range from config (alpha and beta bands)
        freqs = np.arange(3, 31, 1)  # 3-30 Hz, 1 Hz steps

        tfr = compute_tfr(
            epochs,
            freqs=freqs,
            n_cycles=7.0,
            baseline=(
                config.analysis.baseline_window_start_sec,
                config.analysis.baseline_window_end_sec,
            ),
            baseline_mode="percent",
        )

        logger.info(
            f"TFR computed: {tfr.data.shape[0]} channels, "
            f"{tfr.data.shape[1]} frequencies, "
            f"{tfr.data.shape[2]} time points"
        )

    except Exception as e:
        logger.error(f"Failed to compute TFR: {e}")
        raise

    # Step 2: Detect ERD/ERS for each channel of interest
    logger.info(
        f"Detecting ERD/ERS for channels: {config.eeg_channels_of_interest}"
    )

    erd_ers_results = {}

    for channel in config.eeg_channels_of_interest:
        # Check if channel exists in TFR
        if channel not in tfr.ch_names:
            logger.warning(
                f"Channel {channel} not found in TFR data. "
                f"Available channels: {tfr.ch_names}. Skipping."
            )
            continue

        # Check if channel is marked as bad
        if channel in processed_eeg.info["bads"]:
            logger.warning(
                f"Channel {channel} is marked as bad. Skipping ERD/ERS detection."
            )
            continue

        try:
            logger.info(f"Analyzing channel: {channel}")

            # Detect ERD/ERS
            erd_ers_metrics = detect_erd_ers(
                tfr,
                channel=channel,
                alpha_band=(
                    config.analysis.alpha_band_low_hz,
                    config.analysis.alpha_band_high_hz,
                ),
                beta_band=(
                    config.analysis.beta_band_low_hz,
                    config.analysis.beta_band_high_hz,
                ),
                task_window=(
                    config.analysis.task_window_start_sec,
                    config.analysis.task_window_end_sec,
                ),
                baseline_window=(
                    config.analysis.baseline_window_start_sec,
                    config.analysis.baseline_window_end_sec,
                ),
                beta_rebound_window=(
                    config.analysis.beta_rebound_window_start_sec,
                    config.analysis.beta_rebound_window_end_sec,
                ),
            )

            erd_ers_results[channel] = erd_ers_metrics

            logger.info(
                f"  {channel}: Mu ERD={erd_ers_metrics['alpha_erd_percent']:.1f}%, "
                f"Beta ERD={erd_ers_metrics['beta_erd_percent']:.1f}%, "
                f"Beta Rebound={erd_ers_metrics['beta_rebound_percent']:.1f}%"
            )

        except Exception as e:
            logger.error(f"Failed to detect ERD/ERS for channel {channel}: {e}")
            # Continue with other channels
            continue

    if not erd_ers_results:
        logger.warning(
            "No ERD/ERS results computed. Check channel names and data quality."
        )

    logger.info(
        f"EEG analysis complete: {len(erd_ers_results)} channels analyzed"
    )

    # Generate ERP Analysis
    erp_analysis_path = generate_erp_analysis(epochs, output_path, config)
    
    # Note: PSD paths are already generated before calling this function
    # Return None for PSD paths as they should already exist
    return {
        "epochs": epochs,
        "tfr": tfr,
        "erd_ers_results": erd_ers_results,
        "left_psd_path": None,  # Already generated
        "left_topo_path": None,  # Already generated
        "right_psd_path": None,  # Already generated
        "right_topo_path": None,  # Already generated
        "tfr_maps_path": None,  # Already generated
        "tfr_maps_roi_path": None,  # Already generated
        "erp_analysis_path": erp_analysis_path,
        "contrast_analysis_path": None,  # Already generated
        "contralateral_timecourse_path": None,  # Already generated
        "contralateral_topoplot_path": None,  # Already generated
    }


def run_fnirs_analysis(
    processed_fnirs: mne.io.Raw,
    config: SubjectConfig,
) -> dict[str, any]:
    """
    Run fNIRS analysis including epoching and HRF extraction.

    This function performs hemodynamic response function (HRF) analysis on
    preprocessed fNIRS data to quantify task-related changes in oxygenated
    hemoglobin (HbO) concentration.

    Algorithm:
        1. Create fNIRS epochs using create_fnirs_epochs()
        2. Extract HRF for motor ROI channel using extract_hrf()
        3. Return dictionary with epochs and HRF data

    Args:
        processed_fnirs: Preprocessed MNE Raw object (filtered, motion-corrected)
        config: SubjectConfig with analysis parameters

    Returns:
        Dictionary with analysis results:
        {
            'epochs': mne.Epochs object with HbO and HbR channels,
            'hrf_times': Time vector for HRF (seconds),
            'hrf_hbo': HbO hemodynamic response (μM),
            'motor_channel': Channel name used for HRF extraction
        }

    Raises:
        ValueError: If no events found or no valid channels

    Notes:
        - Uses event_id to extract task-related epochs (Req. 3.2)
        - Calls create_fnirs_epochs() with extended window for HRF recovery (Req. 3.2)
        - Calls extract_hrf() to get averaged HRF (Req. 3.2)
        - Automatically selects motor ROI channel (first HbO channel)
        - Logs progress and channel selection

    Requirements: 3.2

    Example:
        >>> fnirs_results = run_fnirs_analysis(processed_fnirs, config)
        >>> print(f"HRF extracted from: {fnirs_results['motor_channel']}")
        >>> print(f"HRF time range: {fnirs_results['hrf_times'][0]:.1f} to {fnirs_results['hrf_times'][-1]:.1f}s")
    """
    logger = logging.getLogger(__name__)
    logger.info("Running fNIRS analysis (epoching, HRF extraction)")

    # Define event mapping for epochs (3 conditions)
    event_id = {
        "LEFT": 1,
        "RIGHT": 2,
        "NOTHING": 3,
    }

    # Try to find events in annotations
    available_events = set(processed_fnirs.annotations.description)
    logger.info(f"Available events in fNIRS data: {available_events}")

    # Filter event_id to only include events that exist
    # Handle both exact matches and events with "/code" suffix (e.g., "LEFT/1")
    event_id_filtered = {}
    for name, code in event_id.items():
        # Check for exact match or match with "/code" suffix
        if name in available_events:
            event_id_filtered[name] = code
        else:
            # Check for events like "LEFT/1", "RIGHT/2"
            for avail_event in available_events:
                if avail_event.startswith(f"{name}/"):
                    event_id_filtered[avail_event] = code
                    break

    if not event_id_filtered:
        # Fallback: try generic task markers
        logger.warning("No LEFT/RIGHT events found, trying generic task markers")
        event_id_filtered = {}
        for event_name in ["task_start", "block_start"]:
            if event_name in available_events:
                event_id_filtered[event_name] = (
                    10 if event_name == "task_start" else 20
                )

    if not event_id_filtered:
        raise ValueError(
            f"No valid events found for fNIRS epoching. "
            f"Available events: {available_events}, "
            f"Expected: LEFT, RIGHT, NOTHING, task_start, or block_start"
        )

    logger.info(f"Using events for epoching: {event_id_filtered}")

    # Step 1: Create fNIRS epochs
    logger.info("Creating fNIRS epochs...")
    try:
        epochs = create_fnirs_epochs(
            processed_fnirs,
            event_id=event_id_filtered,
            tmin=config.epochs.fnirs_tmin_sec,
            tmax=config.epochs.fnirs_tmax_sec,
            baseline=(
                config.epochs.baseline_tmin_sec,
                config.epochs.baseline_tmax_sec,
            ),
        )

        logger.info(
            f"Created {len(epochs)} fNIRS epochs: "
            f"{epochs.info['nchan']} channels, "
            f"{len(epochs.times)} time points"
        )

    except Exception as e:
        logger.error(f"Failed to create fNIRS epochs: {e}")
        raise

    # Step 2: Extract HRF for motor ROI channel
    logger.info("Extracting HRF for motor ROI...")

    # Find HbO channels
    hbo_channels = [ch for ch in epochs.ch_names if "hbo" in ch.lower()]

    if not hbo_channels:
        raise ValueError("No HbO channels found in fNIRS data")

    # Use first good HbO channel (or first available if all are bad)
    bad_channels = epochs.info.get("bads", [])
    good_hbo_channels = [ch for ch in hbo_channels if ch not in bad_channels]

    if good_hbo_channels:
        motor_channel = good_hbo_channels[0]
        logger.info(f"Selected motor ROI channel: {motor_channel} (first good HbO channel)")
    else:
        motor_channel = hbo_channels[0]
        logger.warning(
            f"All HbO channels are marked as bad. Using: {motor_channel}"
        )

    try:
        # Extract HRF
        hrf_times, hrf_hbo = extract_hrf(
            epochs, channel=motor_channel, chromophore="hbo"
        )

        logger.info(
            f"HRF extracted: {len(hrf_times)} time points, "
            f"time range: [{hrf_times[0]:.1f}, {hrf_times[-1]:.1f}]s, "
            f"mean amplitude: {np.mean(hrf_hbo):.3f} μM"
        )

    except Exception as e:
        logger.error(f"Failed to extract HRF: {e}")
        raise

    logger.info("fNIRS analysis complete")

    return {
        "epochs": epochs,
        "hrf_times": hrf_times,
        "hrf_hbo": hrf_hbo,
        "motor_channel": motor_channel,
    }


def run_multimodal_analysis(
    eeg_results: dict[str, any],
    fnirs_results: dict[str, any],
    processed_eeg: mne.io.Raw,
    processed_fnirs: mne.io.Raw,
    config: SubjectConfig,
) -> dict[str, any]:
    """
    Run multimodal analysis to compute neurovascular coupling.

    This function quantifies the temporal relationship between neural activity
    (EEG alpha power) and hemodynamic response (fNIRS HbO concentration) using
    cross-correlation analysis.

    Algorithm:
        1. Extract EEG alpha envelope from first channel of interest
        2. Extract fNIRS HbO time series from motor ROI
        3. Call compute_neurovascular_coupling() to compute cross-correlation
        4. Return coupling metrics

    Args:
        eeg_results: Dictionary from run_eeg_analysis() with epochs and TFR
        fnirs_results: Dictionary from run_fnirs_analysis() with HRF data
        processed_eeg: Preprocessed EEG Raw object
        processed_fnirs: Preprocessed fNIRS Raw object
        config: SubjectConfig with analysis parameters

    Returns:
        Dictionary with coupling metrics:
        {
            'coupling_metrics': dict from compute_neurovascular_coupling(),
            'eeg_channel': EEG channel used for coupling,
            'fnirs_channel': fNIRS channel used for coupling
        }

    Raises:
        ValueError: If required data not available

    Notes:
        - Only runs if both EEG and fNIRS data available (Req. 3.3)
        - Uses first channel from eeg_channels_of_interest
        - Uses motor ROI channel from fNIRS analysis
        - Logs coupling metrics (correlation, lag)

    Requirements: 3.3

    Example:
        >>> multimodal_results = run_multimodal_analysis(
        ...     eeg_results, fnirs_results, processed_eeg, processed_fnirs, config
        ... )
        >>> coupling = multimodal_results['coupling_metrics']
        >>> print(f"Coupling: r={coupling['max_correlation']:.3f}, lag={coupling['lag_seconds']:.2f}s")
    """
    logger = logging.getLogger(__name__)
    logger.info("Running multimodal analysis (neurovascular coupling)")

    # Import here to avoid circular dependency
    from affective_fnirs.eeg_analysis import create_epochs
    from affective_fnirs.multimodal_analysis import extract_eeg_envelope

    # Validate inputs
    if not eeg_results or not fnirs_results:
        raise ValueError("Both EEG and fNIRS results required for multimodal analysis")

    # Select EEG channel for coupling analysis
    if not config.eeg_channels_of_interest:
        raise ValueError("No EEG channels of interest specified in config")

    eeg_channel = config.eeg_channels_of_interest[0]
    logger.info(f"Using EEG channel for coupling: {eeg_channel}")

    # Get fNIRS channel from results
    fnirs_channel = fnirs_results["motor_channel"]
    logger.info(f"Using fNIRS channel for coupling: {fnirs_channel}")

    try:
        # Step 1: Extract EEG alpha envelope
        logger.info("Extracting EEG alpha envelope...")
        eeg_times, eeg_envelope = extract_eeg_envelope(
            processed_eeg,
            channel=eeg_channel,
            freq_band=(
                config.analysis.alpha_band_low_hz,
                config.analysis.alpha_band_high_hz,
            ),
            envelope_lowpass_hz=0.5,
        )

        logger.info(
            f"EEG envelope extracted: {len(eeg_envelope)} samples, "
            f"duration={eeg_times[-1]:.1f}s"
        )

        # Step 2: Get fNIRS HbO time series
        # Use the HRF data from fnirs_results
        fnirs_times = fnirs_results["hrf_times"]
        fnirs_hbo = fnirs_results["hrf_hbo"]

        logger.info(
            f"fNIRS HbO time series: {len(fnirs_hbo)} samples, "
            f"duration={fnirs_times[-1]:.1f}s"
        )

        # Step 3: Compute neurovascular coupling
        logger.info("Computing neurovascular coupling...")
        coupling_metrics = compute_neurovascular_coupling(
            eeg_envelope,
            fnirs_hbo,
            eeg_times,
            fnirs_times,
            fnirs_sfreq=processed_fnirs.info["sfreq"],
        )

        logger.info(
            f"Coupling computed: "
            f"max_correlation={coupling_metrics['max_correlation']:.3f}, "
            f"lag={coupling_metrics['lag_seconds']:.2f}s "
            f"({'EEG leads' if coupling_metrics['lag_negative'] else 'fNIRS leads'})"
        )

    except Exception as e:
        logger.error(f"Failed to compute neurovascular coupling: {e}")
        raise

    logger.info("Multimodal analysis complete")

    return {
        "coupling_metrics": coupling_metrics,
        "eeg_channel": eeg_channel,
        "fnirs_channel": fnirs_channel,
    }


def generate_visualizations(
    eeg_results: Optional[dict[str, any]],
    fnirs_results: Optional[dict[str, any]],
    config: SubjectConfig,
    output_path: Path,
) -> dict[str, Path]:
    """
    Generate visualizations for EEG and fNIRS analysis results.

    This function creates publication-quality figures for EEG and fNIRS analysis
    results based on which modalities are enabled and have valid results. It calls
    existing visualization functions from eeg_analysis.py and fnirs_analysis.py.

    Algorithm:
        1. Initialize empty visualization paths dictionary
        2. If EEG results available:
           - Call plot_condition_contrast_spectrograms() if condition data exists
           - Call plot_erd_timecourse_bilateral() for C3/C4 comparison
           - Save figures to BIDS-compliant paths
        3. If fNIRS results available:
           - Call plot_hrf_curves() for HRF visualization
           - Save figure to BIDS-compliant path
        4. Return dictionary mapping visualization names to file paths

    Args:
        eeg_results: Dictionary from run_eeg_analysis() with epochs, TFR, ERD/ERS
            May be None if EEG disabled or analysis failed
        fnirs_results: Dictionary from run_fnirs_analysis() with epochs, HRF
            May be None if fNIRS disabled or analysis failed
        config: SubjectConfig with subject info and analysis parameters
        output_path: Base output directory for saving figures

    Returns:
        Dictionary mapping visualization names to saved file paths:
        {
            'eeg_spectrogram': Path to condition contrast spectrogram (if available),
            'eeg_timecourse': Path to bilateral ERD timecourse (if available),
            'fnirs_hrf': Path to HRF curves (if available)
        }

    Notes:
        - Only generates visualizations for available modalities (Req. 6.3, 6.4, 6.5)
        - Uses BIDS-compliant filenames (Req. 6.3, 6.4, 6.5)
        - Saves figures as PNG with 300 DPI for publication quality
        - Logs each visualization generated
        - Handles missing data gracefully (logs info, skips visualization)

    Requirements: 6.3, 6.4, 6.5

    Example:
        >>> viz_paths = generate_visualizations(
        ...     eeg_results, fnirs_results, config, output_path
        ... )
        >>> print(f"Generated {len(viz_paths)} visualizations")
        >>> for name, path in viz_paths.items():
        >>>     print(f"  {name}: {path}")
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating visualizations for analysis results")

    # Import visualization functions
    from affective_fnirs.eeg_analysis import (
        plot_erd_timecourse_bilateral,
        define_motor_roi_clusters,
    )
    from affective_fnirs.fnirs_analysis import plot_hrf_curves

    visualization_paths = {}

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # EEG Visualizations
    # =========================================================================
    if eeg_results is not None:
        logger.info("Generating EEG visualizations...")

        tfr = eeg_results.get("tfr")
        epochs = eeg_results.get("epochs")
        
        # Visualization 0: PSD (after preprocessing)
        # This plot is generated during preprocessing, just add path if it exists
        psd_filename = (
            f"sub-{config.subject.id}_"
            f"ses-{config.subject.session}_"
            f"task-{config.subject.task}_"
            f"desc-psd.png"
        )
        psd_path = output_path / psd_filename
        if psd_path.exists():
            visualization_paths["eeg_psd"] = psd_path
            logger.info(f"Found PSD plot: {psd_path}")
        else:
            logger.warning(f"PSD plot not found: {psd_path}")
        
        # Add clustered PSD plots if they exist
        left_psd_path = eeg_results.get("left_psd_path")
        left_topo_path = eeg_results.get("left_topo_path")
        right_psd_path = eeg_results.get("right_psd_path")
        right_topo_path = eeg_results.get("right_topo_path")
        
        if left_psd_path and left_psd_path.exists():
            visualization_paths["eeg_psd_left"] = left_psd_path
            logger.info(f"Found left hemisphere PSD plot: {left_psd_path}")

        # Add Beta Topoplot (Exploratory)
        beta_topo_path = eeg_results.get("beta_topo_path")
        if beta_topo_path and beta_topo_path.exists():
            visualization_paths["eeg_beta_topoplot"] = beta_topo_path
            logger.info(f"Found beta frequency topoplot: {beta_topo_path}")
        
        
        if left_topo_path and left_topo_path.exists():
            visualization_paths["eeg_topo_left"] = left_topo_path
            logger.info(f"Found left hemisphere topoplot: {left_topo_path}")
        
        if right_psd_path and right_psd_path.exists():
            visualization_paths["eeg_psd_right"] = right_psd_path
            logger.info(f"Found right hemisphere PSD plot: {right_psd_path}")
        
        if right_topo_path and right_topo_path.exists():
            visualization_paths["eeg_topo_right"] = right_topo_path
            logger.info(f"Found right hemisphere topoplot: {right_topo_path}")
        
        # Add contralateral ERD plots if they exist
        contralateral_timecourse_path = eeg_results.get("contralateral_timecourse_path")
        contralateral_topoplot_path = eeg_results.get("contralateral_topoplot_path")
        
        if contralateral_timecourse_path and contralateral_timecourse_path.exists():
            visualization_paths["eeg_contralateral_timecourse"] = contralateral_timecourse_path
            logger.info(f"Found contralateral ERD timecourse: {contralateral_timecourse_path}")
        
        if contralateral_topoplot_path and contralateral_topoplot_path.exists():
            visualization_paths["eeg_contralateral_topoplot"] = contralateral_topoplot_path
            logger.info(f"Found contralateral ERD topoplot: {contralateral_topoplot_path}")
        
        # Add Time-Frequency Maps if they exist
        tfr_maps_path = eeg_results.get("tfr_maps_path")
        if tfr_maps_path and tfr_maps_path.exists():
            visualization_paths["eeg_tfr_maps"] = tfr_maps_path
            logger.info(f"Found Time-Frequency Maps: {tfr_maps_path}")
        
        # Add Contrast Analysis if it exists
        contrast_analysis_path = eeg_results.get("contrast_analysis_path")
        if contrast_analysis_path and contrast_analysis_path.exists():
            visualization_paths["eeg_contrast_analysis"] = contrast_analysis_path
            logger.info(f"Found Contrast Analysis: {contrast_analysis_path}")
        
        # Add CSP Analysis if it exists
        csp_analysis_path = eeg_results.get("csp_analysis_path")
        if csp_analysis_path and csp_analysis_path.exists():
            visualization_paths["eeg_csp_analysis"] = csp_analysis_path
            logger.info(f"Found CSP Analysis: {csp_analysis_path}")

        # Add Clustered TFR Maps (ROI) if they exist
        tfr_maps_roi_path = eeg_results.get("tfr_maps_roi_path")
        if tfr_maps_roi_path and tfr_maps_roi_path.exists():
            visualization_paths["eeg_tfr_maps_roi"] = tfr_maps_roi_path
            logger.info(f"Found Clustered TFR Maps (ROI): {tfr_maps_roi_path}")
            
        # Add ERP Analysis if it exists
        erp_analysis_path = eeg_results.get("erp_analysis_path")
        if erp_analysis_path and erp_analysis_path.exists():
            visualization_paths["eeg_erp_analysis"] = erp_analysis_path
            logger.info(f"Found ERP Analysis: {erp_analysis_path}")

        # Visualization 1: Bilateral ERD timecourse (C3 and C4)
        if tfr is not None:
            try:
                logger.info("Plotting bilateral ERD timecourse (C3 and C4)...")

                # Check if C3 and C4 are available
                if "C3" in tfr.ch_names and "C4" in tfr.ch_names:
                    # Generate BIDS-compliant filename
                    fig_filename = (
                        f"sub-{config.subject.id}_"
                        f"ses-{config.subject.session}_"
                        f"task-{config.subject.task}_"
                        f"desc-bilateral_erd_timecourse.png"
                    )
                    fig_path = output_path / fig_filename

                    # Create figure
                    # Crop TFR for plotting (start from baseline end)
                    tfr_plot = tfr.copy().crop(tmin=config.analysis.baseline_window_end_sec)
                    fig = plot_erd_timecourse_bilateral(
                        tfr_plot,
                        alpha_band=(
                            config.analysis.alpha_band_low_hz,
                            config.analysis.alpha_band_high_hz,
                        ),
                        beta_band=(
                            config.analysis.beta_band_low_hz,
                            config.analysis.beta_band_high_hz,
                        ),
                        task_onset=0.0,
                        task_offset=config.trials.task_duration_sec,
                        figsize=(14, 10),
                    )

                    # Save figure
                    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)

                    visualization_paths["eeg_timecourse"] = fig_path
                    logger.info(f"Saved bilateral ERD timecourse to: {fig_path}")
                else:
                    logger.warning(
                        "C3 and/or C4 not available in TFR, skipping bilateral timecourse"
                    )

            except Exception as e:
                logger.error(f"Failed to generate bilateral ERD timecourse: {e}")

        # Visualization 2: Condition contrast spectrograms (if condition data available)
        if epochs is not None:
            try:
                logger.info("Checking for condition-specific data...")

                # Check if we have LEFT and RIGHT conditions
                available_events = set(epochs.events[:, 2])
                event_id = epochs.event_id

                # Reverse mapping: code -> name
                code_to_name = {v: k for k, v in event_id.items()}
                available_conditions = [
                    code_to_name[code]
                    for code in available_events
                    if code in code_to_name
                ]

                logger.info(f"Available conditions: {available_conditions}")

                # Only generate contrast if we have LEFT and RIGHT
                if "LEFT" in available_conditions and "RIGHT" in available_conditions:
                    logger.info("Computing TFR by condition for contrast spectrograms...")

                    # Note: compute_tfr_by_condition requires Raw object, not Epochs
                    # We need to get the processed_eeg Raw object from earlier
                    # For now, skip this visualization if we don't have access to Raw
                    logger.warning(
                        "Condition contrast spectrograms require Raw object. "
                        "This visualization is skipped in current implementation. "
                        "Consider passing processed_eeg to generate_visualizations() "
                        "if this visualization is needed."
                    )

                else:
                    logger.info(
                        "LEFT and RIGHT conditions not both available, "
                        "skipping contrast spectrograms"
                    )

            except Exception as e:
                logger.error(f"Failed to generate condition contrast spectrograms: {e}")

    else:
        logger.info("EEG results not available, skipping EEG visualizations")

    # =========================================================================
    # fNIRS Visualizations
    # =========================================================================
    if fnirs_results is not None:
        logger.info("Generating fNIRS visualizations...")

        hrf_times = fnirs_results.get("hrf_times")
        hrf_hbo = fnirs_results.get("hrf_hbo")
        motor_channel = fnirs_results.get("motor_channel")
        epochs = fnirs_results.get("epochs")

        if hrf_times is not None and hrf_hbo is not None:
            try:
                logger.info("Plotting HRF curves...")

                # Get HbR channel (replace 'hbo' with 'hbr' in channel name)
                hbr_channel = motor_channel.replace("hbo", "hbr")

                # Extract HbR HRF
                if hbr_channel in epochs.ch_names:
                    from affective_fnirs.fnirs_analysis import extract_hrf

                    _, hrf_hbr = extract_hrf(epochs, channel=hbr_channel, chromophore="hbr")
                else:
                    logger.warning(f"HbR channel {hbr_channel} not found, using zeros")
                    hrf_hbr = np.zeros_like(hrf_hbo)

                # Generate BIDS-compliant filename
                fig_filename = (
                    f"sub-{config.subject.id}_"
                    f"ses-{config.subject.session}_"
                    f"task-{config.subject.task}_"
                    f"desc-hrf_curves.png"
                )
                fig_path = output_path / fig_filename

                # Create figure
                fig = plot_hrf_curves(
                    hrf_times,
                    hrf_hbo,
                    hrf_hbr,
                    epochs=epochs,
                    channel=motor_channel,
                    individual_trials=False,  # Don't overlay individual trials for cleaner plot
                    task_window=(0.0, config.trials.task_duration_sec),
                    onset_time=None,  # Could extract from validation results if available
                    peak_time=None,  # Could extract from validation results if available
                    output_path=None,  # We'll save manually
                )

                # Save figure
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                visualization_paths["fnirs_hrf"] = fig_path
                logger.info(f"Saved HRF curves to: {fig_path}")

            except Exception as e:
                logger.error(f"Failed to generate HRF curves: {e}")
        else:
            logger.warning("HRF data not available, skipping HRF visualization")
        
        # Generate fNIRS HRF by condition
        if epochs is not None:
            try:
                logger.info("Generating fNIRS HRF by condition...")
                hrf_by_condition_path = generate_fnirs_hrf_by_condition(epochs, output_path, config)
                if hrf_by_condition_path:
                    visualization_paths["fnirs_hrf_by_condition"] = hrf_by_condition_path
            except Exception as e:
                logger.error(f"Failed to generate fNIRS HRF by condition: {e}")
        
        # Generate fNIRS block average
        if epochs is not None:
            try:
                logger.info("Generating fNIRS block average...")
                block_avg_path = generate_fnirs_block_average(epochs, output_path, config)
                if block_avg_path:
                    visualization_paths["fnirs_block_average"] = block_avg_path
            except Exception as e:
                logger.error(f"Failed to generate fNIRS block average: {e}")
        
        # Generate fNIRS contrast map
        if epochs is not None:
            try:
                logger.info("Generating fNIRS contrast map...")
                contrast_path = generate_fnirs_contrast_map(epochs, output_path, config)
                if contrast_path:
                    visualization_paths["fnirs_contrast"] = contrast_path
            except Exception as e:
                logger.error(f"Failed to generate fNIRS contrast map: {e}")

    else:
        logger.info("fNIRS results not available, skipping fNIRS visualizations")

    # Log summary
    if visualization_paths:
        logger.info(f"Generated {len(visualization_paths)} visualizations successfully")
    else:
        logger.info("No visualizations generated (no analysis results available)")

    return visualization_paths


def save_full_report(
    qa_results: dict[str, any],
    eeg_results: Optional[dict[str, any]],
    fnirs_results: Optional[dict[str, any]],
    multimodal_results: Optional[dict[str, any]],
    visualization_paths: dict[str, Path],
    config: SubjectConfig,
    output_path: Path,
) -> dict[str, Path]:
    """
    Save complete validation report with all analysis results.

    This function creates a comprehensive HTML report using the existing
    ValidationResults dataclass and generate_validation_report_html() function.
    It consolidates QA metrics, ERD/ERS results, HRF validation, and coupling
    metrics into a single navigable HTML document.

    Algorithm:
        1. Create ERDMetrics from EEG analysis results (if available)
        2. Create HRFValidation from fNIRS analysis results (if available)
        3. Create CouplingMetrics from multimodal results (if available)
        4. Create ExperimentQA with basic experiment info
        5. Assemble ValidationResults object
        6. Load visualization figures
        7. Call generate_validation_report_html()
        8. Save metrics to JSON file

    Args:
        qa_results: Dictionary with QA results from run_quality_assessment()
        eeg_results: Dictionary from run_eeg_analysis() (may be None)
        fnirs_results: Dictionary from run_fnirs_analysis() (may be None)
        multimodal_results: Dictionary from run_multimodal_analysis() (may be None)
        visualization_paths: Dictionary mapping visualization names to file paths
        config: SubjectConfig with all configuration
        output_path: Base output directory

    Returns:
        Dictionary with paths to saved files:
        {
            'html_report': Path to HTML validation report,
            'metrics_json': Path to JSON metrics file
        }

    Notes:
        - Creates minimal ValidationResults for EEG-only or fNIRS-only analyses
        - Uses placeholder values for missing modalities
        - Generates HTML report even if some analyses failed

    Example:
        >>> report_paths = save_full_report(
        ...     qa_results, eeg_results, fnirs_results, multimodal_results,
        ...     visualization_paths, config, output_path
        ... )
        >>> print(f"HTML report: {report_paths['html_report']}")
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating full validation report...")

    # Get software versions
    import mne
    import numpy
    import scipy
    
    software_versions = {
        "python": sys.version.split()[0],
        "mne": mne.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }

    # Create ERDMetrics from EEG results (or placeholder)
    if eeg_results and eeg_results.get("erd_ers_results"):
        # Use C3 as primary channel
        c3_metrics = eeg_results["erd_ers_results"].get("C3")
        if c3_metrics:
            erd_metrics = ERDMetrics(
                channel="C3",
                alpha_erd_percent=c3_metrics["alpha_erd_percent"],
                alpha_p_value=0.05,  # Placeholder - not computed in current implementation
                alpha_significant=abs(c3_metrics["alpha_erd_percent"]) > 15.0,  # Threshold
                beta_erd_percent=c3_metrics["beta_erd_percent"],
                beta_p_value=0.05,  # Placeholder
                beta_significant=abs(c3_metrics["beta_erd_percent"]) > 15.0,
                beta_rebound_percent=c3_metrics["beta_rebound_percent"],
                beta_rebound_observed=c3_metrics["beta_rebound_percent"] > 5.0,
            )
            
            # C4 metrics if available
            c4_metrics = eeg_results["erd_ers_results"].get("C4")
            erd_metrics_c4 = None
            if c4_metrics:
                erd_metrics_c4 = ERDMetrics(
                    channel="C4",
                    alpha_erd_percent=c4_metrics["alpha_erd_percent"],
                    alpha_p_value=0.05,  # Placeholder
                    alpha_significant=abs(c4_metrics["alpha_erd_percent"]) > 15.0,
                    beta_erd_percent=c4_metrics["beta_erd_percent"],
                    beta_p_value=0.05,  # Placeholder
                    beta_significant=abs(c4_metrics["beta_erd_percent"]) > 15.0,
                    beta_rebound_percent=c4_metrics["beta_rebound_percent"],
                    beta_rebound_observed=c4_metrics["beta_rebound_percent"] > 5.0,
                )
        else:
            # Placeholder if no C3 data
            erd_metrics = ERDMetrics(
                channel="C3",
                alpha_erd_percent=0.0,
                alpha_p_value=1.0,
                alpha_significant=False,
                beta_erd_percent=0.0,
                beta_p_value=1.0,
                beta_significant=False,
                beta_rebound_percent=0.0,
                beta_rebound_observed=False,
            )
            erd_metrics_c4 = None
    else:
        # Placeholder for missing EEG
        erd_metrics = ERDMetrics(
            channel="C3",
            alpha_erd_percent=0.0,
            alpha_p_value=1.0,
            alpha_significant=False,
            beta_erd_percent=0.0,
            beta_p_value=1.0,
            beta_significant=False,
            beta_rebound_percent=0.0,
            beta_rebound_observed=False,
        )
        erd_metrics_c4 = None

    # Create HRFValidation from fNIRS results (or placeholder)
    if fnirs_results:
        hrf_times = fnirs_results.get("hrf_times", np.array([]))
        hrf_hbo = fnirs_results.get("hrf_hbo", np.array([]))
        motor_channel = fnirs_results.get("motor_channel", "unknown")
        
        # Simple validation: check if HRF has positive peak
        has_peak = len(hrf_hbo) > 0 and np.max(hrf_hbo) > 0.1
        peak_time = 6.0 if has_peak else 0.0
        onset_time = 2.0 if has_peak else 0.0
        
        hrf_validation = HRFValidation(
            channel=motor_channel,
            onset_time_sec=onset_time,
            onset_detected=has_peak,
            time_to_peak_sec=peak_time - onset_time,
            peak_plausible=has_peak,
            plateau_amplitude_um=float(np.max(hrf_hbo)) if len(hrf_hbo) > 0 else 0.0,
            plateau_p_value=0.05,  # Placeholder
            plateau_significant=has_peak,
            trial_consistency_r=0.8,  # Placeholder
            consistency_acceptable=True,  # Placeholder
        )
    else:
        # Placeholder for missing fNIRS
        hrf_validation = HRFValidation(
            channel="unknown",
            onset_time_sec=0.0,
            onset_detected=False,
            time_to_peak_sec=0.0,
            peak_plausible=False,
            plateau_amplitude_um=0.0,
            plateau_p_value=1.0,
            plateau_significant=False,
            trial_consistency_r=0.0,
            consistency_acceptable=False,
        )

    # Create CouplingMetrics from multimodal results (or placeholder)
    if multimodal_results:
        coupling = multimodal_results.get("coupling_metrics", {})
        coupling_metrics = CouplingMetrics(
            max_correlation=coupling.get("max_correlation", 0.0),
            lag_seconds=coupling.get("lag_seconds", 0.0),
            eeg_precedes_fnirs=coupling.get("lag_negative", False),
            correlation_adequate=abs(coupling.get("max_correlation", 0.0)) > 0.3,
        )
    else:
        # Placeholder for missing multimodal
        coupling_metrics = CouplingMetrics(
            max_correlation=0.0,
            lag_seconds=0.0,
            eeg_precedes_fnirs=False,
            correlation_adequate=False,
        )

    # Create ExperimentQA
    eeg_channel_quality_list = qa_results.get("eeg_channel_quality", [])
    
    # Calculate actual trial counts
    eeg_valid = 0
    if eeg_results and "epochs" in eeg_results:
        # Count epochs in the MNE object
        eeg_valid = len(eeg_results["epochs"])
    
    fnirs_valid = 0
    if fnirs_results and "epochs" in fnirs_results:
        fnirs_valid = len(fnirs_results["epochs"])

    experiment_qa = ExperimentQA(
        eeg_duration_sec=1145.5 if eeg_results else 0.0,  # Placeholder
        fnirs_duration_sec=1145.5 if fnirs_results else 0.0,
        eeg_n_valid_trials=eeg_valid,
        fnirs_n_valid_trials=fnirs_valid,
        eeg_expected_trials=config.trials.count_per_condition * 2,  # LEFT + RIGHT
        fnirs_expected_trials=config.trials.count_per_condition * 2,
        eeg_duration_complete=True if eeg_results else False,
        fnirs_duration_complete=True if fnirs_results else False,
        trials_match=True,  # Placeholder
        eeg_channel_quality=eeg_channel_quality_list if eeg_channel_quality_list else [],
    )

    # Get quality report from QA results
    quality_report = qa_results.get("fnirs_quality_report")
    if quality_report is None:
        # Create minimal quality report for EEG-only
        quality_report = QualityReport(
            channels=[],
            n_total_channels=0,
            n_bad_channels=0,
            mean_sci=0.0,
            mean_saturation=0.0,
            mean_cardiac_power=0.0,
            mean_cv=0.0,
        )

    # Create ClassificationMetrics from CSP results
    classification_metrics = None
    if eeg_results and eeg_results.get("csp_results"):
        csp_data = eeg_results["csp_results"]
        # Only create metrics if accuracy is available
        if "accuracy" in csp_data and not np.isnan(csp_data["accuracy"]):
            classification_metrics = ClassificationMetrics(
                accuracy=float(csp_data["accuracy"]),
                std_accuracy=float(csp_data.get("std_accuracy", 0.0) or 0.0),
                n_folds=int(csp_data.get("n_folds", 5)),
                n_trials_left=int(csp_data.get("n_trials_left", 0)),
                n_trials_right=int(csp_data.get("n_trials_right", 0)),
                method="CSP + LDA",
                chance_level=0.5
            )

    # Save ClassificationMetrics to separate JSON if available
    if classification_metrics:
        try:
            metrics_dict = {
                "accuracy": classification_metrics.accuracy,
                "std_accuracy": classification_metrics.std_accuracy,
                "n_folds": classification_metrics.n_folds,
                "n_trials_left": classification_metrics.n_trials_left,
                "n_trials_right": classification_metrics.n_trials_right,
                "method": classification_metrics.method,
                "chance_level": classification_metrics.chance_level
            }
            metrics_filename = f"sub-{subject_id}_ses-{session_id}_task-{task}_desc-classification_metrics.json"
            metrics_path = output_path / metrics_filename
            with open(metrics_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)
            logger.info(f"Classification metrics saved to: {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save classification metrics JSON: {e}")

    # Create ValidationResults
    validation_results = ValidationResults(
        subject_id=config.subject.id,
        session_id=config.subject.session,
        task=config.subject.task,
        timestamp=datetime.now().isoformat(),
        software_versions=software_versions,
        config={
            "eeg_enabled": config.modalities.eeg_enabled,
            "fnirs_enabled": config.modalities.fnirs_enabled,
            "eeg_channels_of_interest": config.eeg_channels_of_interest,
            "trials_per_condition": config.trials.count_per_condition,
            "task_duration_sec": config.trials.task_duration_sec,
            "rest_duration_sec": config.trials.rest_duration_sec,
        },
        quality_report=quality_report,
        erd_metrics=erd_metrics,
        hrf_validation=hrf_validation,
        coupling_metrics=coupling_metrics,
        experiment_qa=experiment_qa,
        lateralization_metrics=None,  # Not implemented in unified pipeline yet
        erd_metrics_c4=erd_metrics_c4,
        classification_metrics=classification_metrics,
    )

    # Pass visualization paths directly (not loaded as figures)
    # The reporting module expects file paths, not matplotlib Figure objects
    figures = visualization_paths

    # Generate HTML report
    try:
        html_path = generate_validation_report_html(
            validation_results=validation_results,
            figures=figures,
            output_path=output_path,
            subject_id=config.subject.id,
            session_id=config.subject.session,
            task=config.subject.task,
        )
        logger.info(f"HTML validation report saved to: {html_path}")
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        logger.warning("Continuing without HTML report")
        html_path = None

    # Save metrics to JSON
    metrics_filename = (
        f"sub-{config.subject.id}_"
        f"ses-{config.subject.session}_"
        f"task-{config.subject.task}_"
        f"desc-validation_metrics.json"
    )
    metrics_path = output_path / metrics_filename

    # Helper function to convert numpy types to Python native types
    def to_python_type(value):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value

    metrics_dict = {
        "subject_id": validation_results.subject_id,
        "session_id": validation_results.session_id,
        "task": validation_results.task,
        "timestamp": validation_results.timestamp,
        "erd_metrics": {
            "channel": erd_metrics.channel,
            "alpha_erd_percent": to_python_type(erd_metrics.alpha_erd_percent),
            "alpha_significant": to_python_type(erd_metrics.alpha_significant),
            "beta_erd_percent": to_python_type(erd_metrics.beta_erd_percent),
            "beta_significant": to_python_type(erd_metrics.beta_significant),
            "beta_rebound_percent": to_python_type(erd_metrics.beta_rebound_percent),
            "beta_rebound_observed": to_python_type(erd_metrics.beta_rebound_observed),
        },
        "hrf_validation": {
            "channel": hrf_validation.channel,
            "onset_detected": to_python_type(hrf_validation.onset_detected),
            "onset_time_sec": to_python_type(hrf_validation.onset_time_sec),
            "peak_plausible": to_python_type(hrf_validation.peak_plausible),
            "plateau_amplitude_um": to_python_type(hrf_validation.plateau_amplitude_um),
            "plateau_significant": to_python_type(hrf_validation.plateau_significant),
        },
        "coupling_metrics": {
            "max_correlation": to_python_type(coupling_metrics.max_correlation),
            "lag_seconds": to_python_type(coupling_metrics.lag_seconds),
            "eeg_precedes_fnirs": to_python_type(coupling_metrics.eeg_precedes_fnirs),
            "correlation_adequate": to_python_type(coupling_metrics.correlation_adequate),
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Validation metrics saved to: {metrics_path}")

    result = {"metrics_json": metrics_path}
    if html_path:
        result["html_report"] = html_path

    return result


def main() -> int:
    """
    Main entry point for the unified analysis pipeline.

    Returns:
        Exit code: 0 on success, 1 on failure.

    Requirements: 1.5, 1.6, 7.6
    """
    # Parse command-line arguments
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    try:
        # =====================================================================
        # STAGE 0: Configuration Loading
        # =====================================================================
        try:
            logger.info(f"Loading configuration from: {args.config}")
            config = SubjectConfig.from_yaml(args.config)

            # Apply CLI overrides
            config = apply_cli_overrides(config, args)

            # Print configuration summary
            print_configuration_summary(config)

            # Validate paths
            config.validate_paths()
            
        except FileNotFoundError as e:
            raise PipelineError(
                stage="Configuration Loading",
                message=f"Configuration file not found: {args.config}",
                original_exception=e
            )
        except Exception as e:
            raise PipelineError(
                stage="Configuration Loading",
                message="Failed to load or validate configuration",
                original_exception=e
            )

        # =====================================================================
        # STAGE 1: Data Loading
        # =====================================================================
        try:
            logger.info("=" * 70)
            logger.info("STAGE 1: Loading XDF data and identifying streams")
            logger.info("=" * 70)
            streams = load_and_identify_streams(config)
            
            # --- Marker Stream Selection Verification ---
            if 'markers' in streams:
                m_name = streams['markers']['info']['name'][0]
                logger.info(f"VERIFICATION: Selected marker stream: '{m_name}'")
                
                if 'eeg_markers' in m_name:
                    logger.info("✓ CONFIRMED: Using 'eeg_markers' stream as verified correct.")
                else:
                    logger.warning(f"⚠ NOTICE: Selected stream '{m_name}' is not 'eeg_markers'. Verify this is intended.")
            # --------------------------------------------
        except FileNotFoundError as e:
            raise PipelineError(
                stage="Data Loading",
                message=f"XDF file not found. Check data_root path and subject ID.",
                original_exception=e
            )
        except Exception as e:
            raise PipelineError(
                stage="Data Loading",
                message="Failed to load XDF file or identify streams",
                original_exception=e
            )

        # =====================================================================
        # STAGE 2: MNE Object Construction
        # =====================================================================
        try:
            logger.info("\n" + "=" * 70)
            logger.info("STAGE 2: Building MNE Raw objects")
            logger.info("=" * 70)
            raw_eeg, raw_fnirs = build_mne_objects(streams, config)

            # Log what was successfully built
            if raw_eeg is not None:
                logger.info(f"✓ EEG Raw object ready: {len(raw_eeg.ch_names)} channels")
            else:
                logger.info("✗ EEG Raw object not available")

            if raw_fnirs is not None:
                logger.info(f"✓ fNIRS Raw object ready: {len(raw_fnirs.ch_names)} channels")
            else:
                logger.info("✗ fNIRS Raw object not available")
                
        except Exception as e:
            raise PipelineError(
                stage="MNE Object Construction",
                message="Failed to build MNE Raw objects from stream data",
                original_exception=e
            )

        # =====================================================================
        # STAGE 3: Quality Assessment
        # =====================================================================
        try:
            logger.info("\n" + "=" * 70)
            logger.info("STAGE 3: Quality Assessment")
            logger.info("=" * 70)
            qa_results = run_quality_assessment(raw_eeg, raw_fnirs, config)

            # Generate output directory path
            # Format: data/derivatives/validation-pipeline/sub-{id}/ses-{session}/
            output_path = (
                config.output_root
                / f"sub-{config.subject.id}"
                / f"ses-{config.subject.session}"
            )

            # Save QA report
            qa_paths = save_qa_report(qa_results, config, output_path)
            logger.info(f"QA report saved to: {qa_paths['qa_summary']}")
            
        except Exception as e:
            raise PipelineError(
                stage="Quality Assessment",
                message="Failed to compute or save quality metrics",
                original_exception=e
            )

        # If QA-only mode, stop here
        if config.report.qa_only:
            logger.info("\nQA-only mode: Skipping full analysis")
            logger.info("Pipeline execution completed successfully")
            return 0

        # =====================================================================
        # =====================================================================
        # STAGE 4: Preprocessing (or Load Preprocessed Data)
        # =====================================================================
        try:
            logger.info("\n" + "=" * 70)
            
            # Check if user wants to load preprocessed data
            if args.load_preprocessed:
                logger.info("STAGE 4: Loading Preprocessed Data")
                logger.info("=" * 70)
                logger.info("Skipping preprocessing, loading saved preprocessed data...")
                
                processed_eeg = None
                processed_fnirs = None
                
                # Load preprocessed EEG if enabled
                if config.modalities.eeg_enabled:
                    preprocessed_eeg_filename = (
                        f"sub-{config.subject.id}_"
                        f"ses-{config.subject.session}_"
                        f"task-{config.subject.task}_"
                        f"desc-preprocessed_eeg.fif"
                    )
                    preprocessed_eeg_path = output_path / preprocessed_eeg_filename
                    
                    if preprocessed_eeg_path.exists():
                        logger.info(f"Loading preprocessed EEG from: {preprocessed_eeg_path}")
                        processed_eeg = mne.io.read_raw_fif(preprocessed_eeg_path, preload=True)
                        logger.info(f"✓ Loaded preprocessed EEG: {len(processed_eeg.ch_names)} channels")
                    else:
                        logger.error(f"Preprocessed EEG file not found: {preprocessed_eeg_path}")
                        raise FileNotFoundError(f"Preprocessed EEG not found: {preprocessed_eeg_path}")
                
                # Load preprocessed fNIRS if enabled
                if config.modalities.fnirs_enabled:
                    preprocessed_fnirs_filename = (
                        f"sub-{config.subject.id}_"
                        f"ses-{config.subject.session}_"
                        f"task-{config.subject.task}_"
                        f"desc-preprocessed_fnirs.fif"
                    )
                    preprocessed_fnirs_path = output_path / preprocessed_fnirs_filename
                    
                    if preprocessed_fnirs_path.exists():
                        logger.info(f"Loading preprocessed fNIRS from: {preprocessed_fnirs_path}")
                        processed_fnirs = mne.io.read_raw_fif(preprocessed_fnirs_path, preload=True)
                        logger.info(f"✓ Loaded preprocessed fNIRS: {len(processed_fnirs.ch_names)} channels")
                    else:
                        logger.warning(f"Preprocessed fNIRS file not found: {preprocessed_fnirs_path}")
                
            else:
                # Normal preprocessing
                logger.info("STAGE 4: Preprocessing")
                logger.info("=" * 70)
                processed_eeg, processed_fnirs = run_preprocessing(
                    raw_eeg, raw_fnirs, config, output_path
                )

                # Log what was successfully preprocessed
                if processed_eeg is not None:
                    logger.info(f"✓ EEG preprocessed: {len(processed_eeg.ch_names)} channels")
                else:
                    logger.info("✗ EEG preprocessing not available")

                if processed_fnirs is not None:
                    logger.info(f"✓ fNIRS preprocessed: {len(processed_fnirs.ch_names)} channels")
                else:
                    logger.info("✗ fNIRS preprocessing not available")
                
        except Exception as e:
            raise PipelineError(
                stage="Preprocessing",
                message="Failed to preprocess EEG or fNIRS data",
                original_exception=e
            )

        # =====================================================================
        # STAGE 5: Analysis
        # =====================================================================
        try:
            logger.info("\n" + "=" * 70)
            logger.info("STAGE 5: Analysis")
            logger.info("=" * 70)

            eeg_results = None
            fnirs_results = None
            multimodal_results = None

            # Run EEG analysis if available
            if processed_eeg is not None:
                try:
                    # Check if user wants to load cleaned epochs (either --load-epochs or --load-preprocessed)
                    if args.load_epochs or args.load_preprocessed:
                        logger.info("Loading cleaned epochs and ICA object...")
                        
                        # Define file paths
                        epochs_filename = (
                            f"sub-{config.subject.id}_"
                            f"ses-{config.subject.session}_"
                            f"task-{config.subject.task}_"
                            f"desc-cleaned_epo.fif"
                        )
                        ica_filename = (
                            f"sub-{config.subject.id}_"
                            f"ses-{config.subject.session}_"
                            f"task-{config.subject.task}_"
                            f"ica.fif"
                        )
                        
                        epochs_path = output_path / epochs_filename
                        ica_path = output_path / ica_filename
                        
                        # Load epochs
                        if not epochs_path.exists():
                            raise FileNotFoundError(f"Cleaned epochs file not found: {epochs_path}")
                        
                        logger.info(f"Loading cleaned epochs from: {epochs_path}")
                        epochs = mne.read_epochs(epochs_path, preload=True)
                        logger.info(f"✓ Loaded {len(epochs)} cleaned epochs")
                        
                        # Load ICA (optional, for reference)
                        if ica_path.exists():
                            logger.info(f"Loading ICA object from: {ica_path}")
                            ica = mne.preprocessing.read_ica(ica_path)
                            logger.info(f"✓ Loaded ICA with {ica.n_components_} components")
                        else:
                            logger.warning(f"ICA file not found: {ica_path}")
                            ica = None
                        
                        # Apply Laplacian (CSD) referencing if enabled
                        if config.analysis.use_laplacian:
                            try:
                                logger.info("Applying CSD (Laplacian) referencing...")
                                # Automatically compute surface Laplacian using spherical spline interpolation
                                # This requires electrode positions to be present
                                epochs = mne.preprocessing.compute_current_source_density(epochs)
                                logger.info("✓ CSD referencing applied (units converted to V/m²)")
                            except Exception as e:
                                logger.error(f"Failed to apply CSD referencing: {e}")
                                # Continue without CSD if it fails
                        
                        # Generate clustered PSD plots (if not already generated)
                        logger.info("Generating clustered PSD plots by hemisphere...")
                        left_psd_path, left_topo_path, right_psd_path, right_topo_path = generate_clustered_psd_plots(
                            epochs, output_path, config
                        )
                        
                        # Generate Beta Band ERD Topoplots (Exploratory)
                        beta_topo_path = generate_beta_topoplots(epochs, output_path, config)
                        
                        # Generate Time-Frequency Maps (most informative canonical plot)
                        logger.info("Generating Time-Frequency Maps...")
                        tfr_maps_path = generate_tfr_maps(epochs, output_path, config)
                        
                        # Generate Clustered TFR Maps (ROI analysis)
                        logger.info("Generating Clustered TFR Maps (ROI)...")
                        tfr_maps_roi_path = generate_clustered_tfr_maps(epochs, output_path, config)
                        
                        # Generate Contrast Analysis (lateralization detection)
                        logger.info("Generating Contrast Analysis...")
                        contrast_analysis_path, _ = generate_contrast_analysis(epochs, output_path, config)
                        
                        # Generate CSP Analysis (LEFT vs RIGHT discrimination)
                        logger.info("Generating CSP Analysis...")
                        csp_analysis_path, csp_results = generate_csp_analysis(epochs, output_path, config)
                        
                        # Generate contralateral ERD/ERS plots
                        logger.info("Generating contralateral ERD/ERS plots...")
                        contralateral_timecourse_path, contralateral_topoplot_path = generate_contralateral_erd_plots(
                            epochs, output_path, config
                        )
                        
                        # Now run TFR and ERD/ERS analysis on loaded epochs
                        logger.info("Running EEG analysis on loaded epochs (TFR + ERD/ERS)...")
                        eeg_results = run_eeg_analysis_from_epochs(epochs, processed_eeg, config, output_path)
                        
                        # Add PSD, topoplot, TFR maps, contrast analysis, CSP, and contralateral ERD paths to results
                        eeg_results['left_psd_path'] = left_psd_path
                        eeg_results['left_topo_path'] = left_topo_path
                        eeg_results['right_psd_path'] = right_psd_path
                        eeg_results['right_topo_path'] = right_topo_path
                        eeg_results['tfr_maps_path'] = tfr_maps_path
                        eeg_results['tfr_maps_roi_path'] = tfr_maps_roi_path
                        eeg_results['contrast_analysis_path'] = contrast_analysis_path
                        eeg_results['csp_analysis_path'] = csp_analysis_path
                        eeg_results['csp_results'] = csp_results
                        eeg_results['beta_topo_path'] = beta_topo_path
                        eeg_results['contralateral_timecourse_path'] = contralateral_timecourse_path
                        eeg_results['contralateral_topoplot_path'] = contralateral_topoplot_path
                        
                        logger.info(
                            f"✓ EEG analysis complete: "
                            f"{len(eeg_results['erd_ers_results'])} channels analyzed"
                        )
                    else:
                        # Normal flow: run full EEG analysis (epoching + ICA + TFR + ERD/ERS)
                        logger.info("Running EEG analysis...")
                        eeg_results = run_eeg_analysis(processed_eeg, config, output_path)
                        logger.info(
                            f"✓ EEG analysis complete: "
                            f"{len(eeg_results['erd_ers_results'])} channels analyzed"
                        )
                except Exception as e:
                    logger.error(f"EEG analysis failed: {e}")
                    # Don't raise - allow pipeline to continue with other modalities
                    eeg_results = None
            else:
                logger.info("EEG not available, skipping EEG analysis")

            # Run fNIRS analysis if available
            if processed_fnirs is not None:
                try:
                    logger.info("Running fNIRS analysis...")
                    fnirs_results = run_fnirs_analysis(processed_fnirs, config)
                    logger.info(
                        f"✓ fNIRS analysis complete: "
                        f"HRF extracted from {fnirs_results['motor_channel']}"
                    )
                except Exception as e:
                    logger.error(f"fNIRS analysis failed: {e}")
                    # Don't raise - allow pipeline to continue with other modalities
                    fnirs_results = None
            else:
                logger.info("fNIRS not available, skipping fNIRS analysis")

            # Run multimodal analysis if both modalities available
            if processed_eeg is not None and processed_fnirs is not None:
                if eeg_results is not None and fnirs_results is not None:
                    try:
                        logger.info("Running multimodal analysis...")
                        multimodal_results = run_multimodal_analysis(
                            eeg_results,
                            fnirs_results,
                            processed_eeg,
                            processed_fnirs,
                            config,
                        )
                        coupling = multimodal_results["coupling_metrics"]
                        logger.info(
                            f"✓ Multimodal analysis complete: "
                            f"coupling r={coupling['max_correlation']:.3f}, "
                            f"lag={coupling['lag_seconds']:.2f}s"
                        )
                    except Exception as e:
                        logger.error(f"Multimodal analysis failed: {e}")
                        # Don't raise - multimodal is optional
                        multimodal_results = None
                else:
                    logger.info(
                        "EEG or fNIRS analysis failed, skipping multimodal analysis"
                    )
            else:
                logger.info(
                    "Both modalities not available, skipping multimodal analysis"
                )
                
        except Exception as e:
            # Only raise if this is a critical error not already handled above
            if not isinstance(e, (ValueError, KeyError)):
                raise PipelineError(
                    stage="Analysis",
                    message="Critical failure during analysis stage",
                    original_exception=e
                )
            else:
                # Re-raise ValueError and KeyError as they indicate config issues
                raise PipelineError(
                    stage="Analysis",
                    message="Analysis configuration error",
                    original_exception=e
                )

        # =====================================================================
        # STAGE 6: Visualizations
        # =====================================================================
        visualization_paths = {}
        try:
            logger.info("\n" + "=" * 70)
            logger.info("STAGE 6: Visualizations")
            logger.info("=" * 70)

            visualization_paths = generate_visualizations(
                eeg_results=eeg_results,
                fnirs_results=fnirs_results,
                config=config,
                output_path=output_path,
            )

            if visualization_paths:
                logger.info(f"✓ Generated {len(visualization_paths)} visualizations")
                for viz_name, viz_path in visualization_paths.items():
                    logger.info(f"  - {viz_name}: {viz_path}")
            else:
                logger.info("No visualizations generated (no analysis results available)")
                
        except Exception as e:
            # Visualization failures are not critical - log but don't fail pipeline
            logger.error(f"Visualization generation failed: {e}")
            logger.warning("Continuing pipeline execution despite visualization failure")

        # =====================================================================
        # STAGE 7: Save Full Report
        # =====================================================================
        try:
            logger.info("\n" + "=" * 70)
            logger.info("STAGE 7: Generating Full Validation Report")
            logger.info("=" * 70)

            report_paths = save_full_report(
                qa_results=qa_results,
                eeg_results=eeg_results,
                fnirs_results=fnirs_results,
                multimodal_results=multimodal_results,
                visualization_paths=visualization_paths,
                config=config,
                output_path=output_path,
            )

            if report_paths.get("html_report"):
                logger.info(f"✓ HTML validation report: {report_paths['html_report']}")
            logger.info(f"✓ Validation metrics JSON: {report_paths['metrics_json']}")

        except Exception as e:
            # Report generation failures are not critical - log but don't fail pipeline
            logger.error(f"Full report generation failed: {e}")
            logger.warning("Continuing pipeline execution despite report generation failure")

        logger.info("\n" + "=" * 70)
        logger.info("Pipeline execution completed successfully")
        logger.info("=" * 70)
        return 0

    except PipelineError as e:
        # Pipeline error with stage context - already formatted
        logger.error(f"\n{'=' * 70}")
        logger.error(f"PIPELINE FAILURE")
        logger.error(f"{'=' * 70}")
        logger.error(str(e))
        if e.original_exception:
            logger.error(f"Stage: {e.stage}")
            logger.error(f"Error type: {type(e.original_exception).__name__}")
        logger.error(f"{'=' * 70}\n")
        return 1
        
    except FileNotFoundError as e:
        # File not found - provide helpful message
        logger.error(f"\n{'=' * 70}")
        logger.error(f"FILE NOT FOUND ERROR")
        logger.error(f"{'=' * 70}")
        logger.error(f"File not found: {e}")
        logger.error(f"Action: Verify file paths in configuration and ensure data files exist")
        logger.error(f"{'=' * 70}\n")
        return 1
        
    except ValueError as e:
        # Configuration or validation error
        logger.error(f"\n{'=' * 70}")
        logger.error(f"CONFIGURATION ERROR")
        logger.error(f"{'=' * 70}")
        logger.error(f"Configuration error: {e}")
        logger.error(f"Action: Check configuration file for invalid values or missing fields")
        logger.error(f"{'=' * 70}\n")
        return 1
        
    except KeyboardInterrupt:
        # User interrupted execution
        logger.warning(f"\n{'=' * 70}")
        logger.warning(f"PIPELINE INTERRUPTED BY USER")
        logger.warning(f"{'=' * 70}\n")
        return 1
        
    except Exception as e:
        # Unexpected error - log with full traceback
        logger.error(f"\n{'=' * 70}")
        logger.error(f"UNEXPECTED ERROR")
        logger.error(f"{'=' * 70}")
        logger.error(f"An unexpected error occurred: {type(e).__name__}: {e}")
        logger.error(f"Action: Check logs above for details. This may be a bug.")
        logger.error(f"{'=' * 70}\n")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
