
import mne
import numpy as np
from pathlib import Path
from affective_fnirs.config import SubjectConfig

def verify_csd_effect():
    # Load config to get paths
    config_path = Path("configs/sub-009.yml")
    config = SubjectConfig.from_yaml(config_path)
    
    # Path to cleaned epochs
    # Search in the entire data/derivatives directory recursively
    search_root = Path("data/derivatives/validation-pipeline")
    pattern = "**/*desc-cleaned_epo.fif"
    
    print(f"Searching for epochs file matching '{pattern}' in {search_root.absolute()}...")
    candidates = list(search_root.rglob("*desc-cleaned_epo.fif"))
    
    if candidates:
        # Prefer sub-009 if multiple found
        epochs_path = next((p for p in candidates if "sub-009" in p.name), candidates[0])
        print(f"Found epochs file: {epochs_path}")
    else:
        # Fallback to current directory search
        print("Not found in derivatives, searching current directory...")
        candidates = list(Path.cwd().rglob("*desc-cleaned_epo.fif"))
        if candidates:
             epochs_path = next((p for p in candidates if "sub-009" in p.name), candidates[0])
             print(f"Found epochs file: {epochs_path}")
        else:
            raise FileNotFoundError("Could not find any *desc-cleaned_epo.fif file")

    print(f"Loading epochs from: {epochs_path}")
    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    
    # 1. Measure properties BEFORE CSD
    data_pre = epochs.get_data(copy=True)
    mean_pre = np.mean(np.abs(data_pre))
    max_pre = np.max(np.abs(data_pre))
    print("\n--- PRE-CSD (Original Voltage) ---")
    print(f"Mean Abs Amplitude: {mean_pre:.2e} V")
    print(f"Max Abs Amplitude:  {max_pre:.2e} V")
    
    # Check correlations between neighbors (C3 vs FC5) to see spatial blurring
    if 'C3' in epochs.ch_names and 'FC5' in epochs.ch_names:
        c3_idx = epochs.ch_names.index('C3')
        fc5_idx = epochs.ch_names.index('FC5')
        corr_pre = np.corrcoef(data_pre[:, c3_idx, :].flatten(), data_pre[:, fc5_idx, :].flatten())[0,1]
        print(f"Correlation C3-FC5 (Pre): {corr_pre:.3f}")

    # 2. Apply CSD
    print("\nApplying CSD...")
    epochs_csd = mne.preprocessing.compute_current_source_density(epochs, verbose=False)
    
    # 3. Measure properties AFTER CSD
    data_post = epochs_csd.get_data(copy=True)
    mean_post = np.mean(np.abs(data_post))
    max_post = np.max(np.abs(data_post))
    
    print("\n--- POST-CSD (Current Source Density) ---")
    print(f"Mean Abs Amplitude: {mean_post:.2e} V/m²")
    print(f"Max Abs Amplitude:  {max_post:.2e} V/m²")
    
    # Check correlations again
    if 'C3' in epochs_csd.ch_names and 'FC5' in epochs_csd.ch_names:
        c3_idx = epochs_csd.ch_names.index('C3')
        fc5_idx = epochs_csd.ch_names.index('FC5')
        corr_post = np.corrcoef(data_post[:, c3_idx, :].flatten(), data_post[:, fc5_idx, :].flatten())[0,1]
        print(f"Correlation C3-FC5 (Post): {corr_post:.3f}")
        
    print("\n--- COMPARISON ---")
    print(f"Magnitude change factor: {mean_post/mean_pre:.1f}x")
    if 'C3' in epochs.ch_names and 'FC5' in epochs.ch_names:
        print(f"Spatial sharpening (Correlation drop): {corr_pre - corr_post:.3f}")

if __name__ == "__main__":
    verify_csd_effect()
