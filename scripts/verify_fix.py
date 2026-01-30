import mne
from pathlib import Path

def main():
    epo_path = Path(r"c:\Users\tdamelio\Desktop\fnirs\affective-fnirs\data\derivatives\validation-pipeline\sub-011\ses-001\sub-011_ses-001_task-fingertapping_desc-cleaned_epo.fif")
    if not epo_path.exists():
        print(f"File not found: {epo_path}")
        return

    print(f"Loading epochs from: {epo_path}")
    epochs = mne.read_epochs(epo_path, preload=True)
    print(f"Total epochs: {len(epochs)}")
    
    # Check by condition if possible
    if hasattr(epochs, 'event_id'):
        print(f"Event IDs: {epochs.event_id}")
        for cond in epochs.event_id:
            try:
                count = len(epochs[cond])
                print(f"  {cond}: {count}")
            except:
                pass

if __name__ == "__main__":
    main()
