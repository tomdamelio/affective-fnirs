"""
Debug script to verify EEG cluster analysis is working correctly.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from affective_fnirs.config import PipelineConfig
from affective_fnirs.eeg_analysis import (
    compute_tfr_by_condition,
    define_motor_roi_clusters,
)
from affective_fnirs.eeg_processing import preprocess_eeg_pipeline
from affective_fnirs.ingestion import (
    extract_stream_data,
    identify_streams,
    load_xdf_file,
)
from affective_fnirs.mne_builder import build_eeg_raw, embed_events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
xdf_file = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
eeg_json = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_eeg.json")

logger.info(f"Loading XDF: {xdf_file}")
streams, header = load_xdf_file(xdf_file)
identified_streams = identify_streams(streams)

eeg_data, eeg_sfreq, eeg_timestamps = extract_stream_data(identified_streams["eeg"])
marker_stream = identified_streams["markers"]

logger.info("Building EEG Raw...")
raw_eeg = build_eeg_raw(
    eeg_data, eeg_sfreq, identified_streams["eeg"]["info"], eeg_timestamps
)

event_mapping = {"LEFT": 1, "RIGHT": 2, "task_start": 10, "task_end": 11}
raw_eeg = embed_events(raw_eeg, marker_stream, event_mapping)

logger.info("Preprocessing EEG...")
config = PipelineConfig.from_yaml(Path("configs/sub002_optimized.yml"))
raw_eeg_processed, _ = preprocess_eeg_pipeline(raw_eeg.copy(), config)

logger.info("Computing TFR by condition...")
tfr_by_condition = compute_tfr_by_condition(
    raw_eeg_processed,
    freqs=np.arange(3, 31, 1),
    n_cycles=7,
    tmin=config.epochs.eeg_tmin_sec,
    tmax=config.epochs.eeg_tmax_sec,
    baseline=(config.epochs.baseline_tmin_sec, config.epochs.baseline_tmax_sec),
    baseline_mode="percent",
)

# Define clusters
clusters = define_motor_roi_clusters()
logger.info(f"Clusters: {clusters}")

# Check what channels are available in TFR
for cond_name, tfr in tfr_by_condition.items():
    if tfr is not None:
        logger.info(f"\n{cond_name} condition:")
        logger.info(f"  Available channels: {tfr.ch_names}")
        logger.info(f"  TFR shape: {tfr.data.shape}")
        
        # Check if cluster channels exist
        for cluster_name, cluster_channels in clusters.items():
            available = [ch for ch in cluster_channels if ch in tfr.ch_names]
            logger.info(f"  {cluster_name}: {available} (requested: {cluster_channels})")
            
            if available:
                # Get indices and compute mean
                ch_indices = [tfr.ch_names.index(ch) for ch in available]
                logger.info(f"    Channel indices: {ch_indices}")
                
                # Extract data for these channels at a specific time/freq point
                # Let's check alpha band (8-13 Hz) at time 5s
                freq_mask = (tfr.freqs >= 8) & (tfr.freqs <= 13)
                time_idx = np.argmin(np.abs(tfr.times - 5.0))
                
                for ch_idx, ch_name in zip(ch_indices, available):
                    alpha_power_at_5s = np.mean(tfr.data[ch_idx, freq_mask, time_idx])
                    logger.info(f"    {ch_name}: alpha power at 5s = {alpha_power_at_5s:.2f}%")
                
                # Compute cluster average
                cluster_data = np.mean(tfr.data[ch_indices, :, :], axis=0)
                cluster_alpha_at_5s = np.mean(cluster_data[freq_mask, time_idx])
                logger.info(f"    {cluster_name} average: {cluster_alpha_at_5s:.2f}%")

logger.info("\nDebug complete!")
