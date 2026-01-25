#!/usr/bin/env python3
"""
Spatial Filtering Analysis Script for EEG Data.

This script implements spatial filtering techniques (Bipolar, CAR, CSP) to analyze
motor lateralization (Left vs Right hand) in EEG data. It is based on course materials
and adapted for the affective-fnirs project structure.

Usage:
    python scripts/spatial_filtering_analysis.py
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mne.decoding import CSP
from sklearn.model_selection import train_test_split

def plot_PSD(PSD, channels_list, colors_list, title, axes, ylim):
    """
    Plot Power Spectral Density (PSD) for multiple channels.
    Adapted from Practical_SpatialFiltering1.ipynb
    """
    # PSD.get_data returns (n_channels, n_freqs)
    # The 'picks' argument in get_data is deprecated or behaves differently in newer MNE versions for Spectrum objects
    # It's better to pick channels before computing PSD or index the result if it's already computed for all channels.
    # Assuming PSD is an mne.time_frequency.Spectrum object computed on epochs (average=False usually results in EpochsSpectrum)
    # or average=True (Spectrum).
    
    # Let's handle both EpochsSpectrum (from compute_psd on epochs) and Spectrum (from compute_psd on raw/evoked) 
    # But usually we compute average PSD across epochs for robust estimation.
    
    # NOTE: In recent MNE versions, compute_psd returns a Spectrum or EpochsSpectrum object.
    # usage: psd.get_data(picks=...)
    
    PSD_array, freqs = PSD.get_data(picks=channels_list, return_freqs=True)
    # PSD_array shape: (n_epochs, n_channels, n_freqs) if from epochs and average=False
    # or (n_channels, n_freqs) if average=True
    
    # We want average across epochs if it's not already averaged
    if PSD_array.ndim == 3:
        PSD_array = np.mean(PSD_array, axis=0)
        
    for c, channel in enumerate(channels_list):
        # 10*np.log10(np.mean(10e12*PSD_array[:, c], axis=0)) -> Scaling to uV^2/Hz and dB
        # MNE PSD data is usually in T^2/Hz (magnetometers) or V^2/Hz (EEG). 
        # For EEG it's V^2/Hz. 1e12 converts V^2 to uV^2 (since (1e6)^2 = 1e12).
        
        # We just plot the mean PSD profile
        psd_data = 10 * np.log10(PSD_array[c] * 1e12) # Convert to uV^2/Hz in dB
        axes.plot(freqs, psd_data, color=colors_list[c], label=channel, linewidth=1.2)
        
    axes.legend()
    axes.grid(True, linestyle=":")
    axes.set_ylabel("μV²/Hz (dB)")
    axes.set_title(title)
    axes.set_ylim(ylim)
    axes.set_xlabel("Frequency (Hz)")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Spatial Filtering Analysis")
    parser.add_argument("--subject_id", type=str, default="009", help="Subject ID (e.g., '009')")
    args = parser.parse_args()

    # --- Configuration ---
    subject_id = args.subject_id
    session_id = "001"
    task_name = "fingertapping"
    
    # Path to preprocessed data - Updated to the file found
    data_dir = Path("data/derivatives/validation-pipeline")
    sub_dir = data_dir / f"sub-{subject_id}" / f"ses-{session_id}"
    input_file = sub_dir / f"sub-{subject_id}_ses-{session_id}_task-{task_name}_desc-preprocessed_eeg.fif"
    
    output_dir = Path("data/derivatives/spatial_filtering") / f"sub-{subject_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing Subject: {subject_id}")
    print(f"Loading data from: {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    raw = mne.io.read_raw_fif(input_file, preload=True)
    
    # --- Preprocessing for Motor Imagery ---
    # Apply band-pass filter (8-30 Hz) as per course (Alpha/Beta band)
    print("Filtering data (8-30 Hz)...")
    raw.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    
    # Extract Events
    events, event_id = mne.events_from_annotations(raw)
    print(f"Events found: {event_id}")
    
    # Identify Left/Right event IDs
    # Assuming standard labelling from the validation pipeline or data
    # We need to map the string event IDs to the integer codes
    left_id = None
    right_id = None
    
    # Heuristic to find relevant event IDs
    for key, val in event_id.items():
        if "LEFT" in key.upper():
            left_id = val
        elif "RIGHT" in key.upper():
            right_id = val
            
    if left_id is None or right_id is None:
        print("Could not identify LEFT and RIGHT events automatically. Using all events for inspection.")
        print(f"Available keys: {event_id.keys()}")
        # If we can't find them, we abort or ask for specification. 
        # For now, let's assume they are there as per previous tasks.
        # Fallback to hardcoded if needed, but let's try to proceed carefully.
        if left_id is None: left_id = 999 
        if right_id is None: right_id = 999
    
    selected_event_ids = {k: v for k, v in event_id.items() if v in [left_id, right_id]}
    print(f"Selected events: {selected_event_ids}")

    # Epoching
    # Time window: 0.5s to 2.5s post-stimulus (as per course material for MI)
    tmin, tmax = 0.5, 2.5
    # Motor cortex channels of interest
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    
    epochs = mne.Epochs(
        raw,
        events,
        selected_event_ids,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    
    # --- Part 1: Spatial Filtering / Rereferencing ---
    print("Running Spatial Filtering (Rereferencing) Analysis...")
    
    # We focus on C3 (Left Motor), C4 (Right Motor), and Cz (Central)
    channels_list = ['C3', 'C4', 'Cz']
    # Filter channels list to only those present in data
    channels_list = [ch for ch in channels_list if ch in raw.ch_names]
    colors_list = ['red', 'blue', 'green'][:len(channels_list)]
    
    if not channels_list:
        print("Warning: C3/C4/Cz channels not found. Skipping specific channel plots.")
    
    # Get separate epochs for conditions
    conditions = list(selected_event_ids.keys())
    if len(conditions) >= 2:
        cond1 = conditions[0]
        cond2 = conditions[1]
        
        # Compute PSD for original reference
        psd_cond1 = epochs[cond1].compute_psd(fmin=8, fmax=30)
        psd_cond2 = epochs[cond2].compute_psd(fmin=8, fmax=30)
        
        # 1.1 Bipolar Rereferencing (simulated)
        # Try to use Pz as reference if available, or another central-parietal channel
        ref_ch = 'Pz' if 'Pz' in raw.ch_names else 'Cz' # Fallback
        if ref_ch in raw.ch_names and ref_ch not in channels_list:
             # Create new epochs with bipolar ref
             print(f"Computing Bipolar Reference (ref={ref_ch})...")
             epochs_bip = epochs.copy().set_eeg_reference(ref_channels=[ref_ch])
             # Note: set_eeg_reference on epochs is supported.
             psd_cond1_bip = epochs_bip[cond1].compute_psd(fmin=8, fmax=30)
             psd_cond2_bip = epochs_bip[cond2].compute_psd(fmin=8, fmax=30)
        else:
            epochs_bip = None
            psd_cond1_bip = None
            psd_cond2_bip = None
            print(f"Skipping Bipolar plot: Reference channel {ref_ch} issue or overlap.")

        # 1.2 CAR Rereferencing
        print("Computing CAR Reference...")
        epochs_car = epochs.copy().set_eeg_reference(ref_channels='average')
        psd_cond1_car = epochs_car[cond1].compute_psd(fmin=8, fmax=30)
        psd_cond2_car = epochs_car[cond2].compute_psd(fmin=8, fmax=30)
        
        # Prepare Plotting for PSD
        fig, axs = plt.subplots(3, 2, figsize=(16, 12))
        
        # Original
        plot_PSD(psd_cond1, channels_list, colors_list, f'{cond1} - Original', axs[0, 0], ylim=None)
        plot_PSD(psd_cond2, channels_list, colors_list, f'{cond2} - Original', axs[0, 1], ylim=None)
        
        # Bipolar
        if psd_cond1_bip:
            plot_PSD(psd_cond1_bip, channels_list, colors_list, f'{cond1} - Bipolar ({ref_ch})', axs[1, 0], ylim=None)
            plot_PSD(psd_cond2_bip, channels_list, colors_list, f'{cond2} - Bipolar ({ref_ch})', axs[1, 1], ylim=None)
        
        # CAR
        plot_PSD(psd_cond1_car, channels_list, colors_list, f'{cond1} - CAR', axs[2, 0], ylim=None)
        plot_PSD(psd_cond2_car, channels_list, colors_list, f'{cond2} - CAR', axs[2, 1], ylim=None)
        
        plt.tight_layout()
        psd_fig_path = output_dir / "psd_rereferencing_comparison.png"
        fig.savefig(psd_fig_path)
        plt.close(fig)
        print(f"Saved PSD comparison to {psd_fig_path}")
        
        
        # --- Part 2: Feature Learning (CSP) ---
        print("Running CSP Analysis...")
        
        # Get data and labels
        labels = epochs.events[:, -1]
        data = epochs.get_data(copy=True) # (n_epochs, n_channels, n_times)
        
        # Split train/test
        # Note: In a real analysis pipeline with multiple runs, would split by run. 
        # Here we split randomly for demonstration as in the notebook.
        if len(labels) > 10: # Ensure enough data
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
            
            # Define CSP
            n_components = 4 # Common choice
            csp = CSP(n_components=n_components, transform_into='csp_space', component_order='alternate', norm_trace=False)
            
            # Fit CSP
            print("Fitting CSP...")
            csp.fit(X_train, y_train)
            
            # Plot CSP patterns
            # plot_patterns returns a figure
            fig_patterns = csp.plot_patterns(epochs.info, show=False)
            patterns_path = output_dir / "csp_patterns.png"
            fig_patterns.savefig(patterns_path)
            plt.close(fig_patterns)
            print(f"Saved CSP patterns to {patterns_path}")
            
            # Transform test data
            X_test_csp = csp.transform(X_test)
            
            # Feature extraction (Log-variance of CSP components)
            # This is what's typically fed to classifiers
            # Option 1 from notebook: (X_test_csp**2).mean(axis=2) -> log -> standardize
            # Option 2: mne CSP transform_into='average_power' does this automatically if log=True
            
            # Let's visualize the features as in the notebook
            # Manually calculating for visualization purposes to match notebook logic
            csp_features = np.log((X_test_csp**2).mean(axis=2))
            
            # Plot first two components features
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Assuming just two classes
            classes = np.unique(y_test)
            colors = ['blue', 'red']
            for i, cls in enumerate(classes):
                # Retrieve label name
                label_name = [k for k, v in selected_event_ids.items() if v == cls][0]
                idx = y_test == cls
                ax.scatter(csp_features[idx, 0], csp_features[idx, 1], 
                           color=colors[i % len(colors)], label=label_name, alpha=0.7)
            
            ax.set_xlabel('CSP feature 0')
            ax.set_ylabel('CSP feature 1')
            ax.legend()
            ax.set_title('CSP Features Separation')
            
            features_path = output_dir / "csp_features_scatter.png"
            fig.savefig(features_path)
            plt.close(fig)
            print(f"Saved CSP features plot to {features_path}")
            
        else:
            print("Not enough epochs for CSP analysis.")

    else:
        print("Not enough conditions for comparison (Left vs Right needed).")

if __name__ == "__main__":
    main()
