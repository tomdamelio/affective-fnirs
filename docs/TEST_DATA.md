# Test Data Reference

## Pilot Subjects Overview

This document describes the available test data for the multimodal validation pipeline.

### Subject 002 (Primary Test Subject) ✅

**Location:** `data/raw/sub-002/`

**Files:**
- `sub-002_tomi_ses-001_task-fingertapping_recording.xdf`
- `sub-002_Tomi_ses-001_task-fingertapping_eeg.json`
- `sub-002_Tomi_ses-001_task-fingertapping_nirs.json`

**XDF Streams:**
| Stream Name | Type | Channels | Samples | Sampling Rate | Duration |
|-------------|------|----------|---------|---------------|----------|
| actiCHamp-24020270 | EEG | 32 | 330,045 | 500 Hz | ~660 sec |
| Photon_Cap_C2022044_RAW | NIRS | 42 | 10,476 | 7.88 Hz | ~1112 sec |
| eeg_markers | Markers | 1 | 36 | 0 Hz | ~1050 sec |

**Status:** ✅ **COMPLETE** - Contains all required streams (EEG + fNIRS + Markers)

**Use for:**
- Primary development and testing
- Full pipeline validation
- EEG preprocessing and ERD/ERS analysis
- fNIRS processing and HRF analysis
- Multimodal neurovascular coupling analysis

---

### Subject 001 (Secondary Test Subject) ⚠️

**Location:** `data/raw/sub-001/`

**Files:**
- `sub-001_tomi_ses-001_task-fingertapping_recording.xdf`
- `sub-001_Tomi_ses-001_task-fingertapping_eeg.json`
- `sub-001_Tomi_ses-001_task-fingertapping_nirs.json`

**XDF Streams:**
| Stream Name | Type | Channels | Samples | Sampling Rate | Duration |
|-------------|------|----------|---------|---------------|----------|
| ❌ **No EEG stream** | - | - | - | - | - |
| Photon_Cap_C2022044_RAW | NIRS | 42 | 10,079 | 7.88 Hz | ~1104 sec |
| eeg_markers | Markers | 1 | 36 | 0 Hz | ~1050 sec |

**Status:** ⚠️ **INCOMPLETE** - Missing EEG stream in XDF file

**Use for:**
- Testing error handling for missing streams
- fNIRS-only pipeline validation
- Verifying graceful degradation when EEG unavailable

---

## Testing Strategy

### Development Phase (Tasks 2-16)
**Use `sub-002` exclusively** to ensure all modules work with complete data.

### Integration Testing (Task 17)
Test both subjects:
1. **sub-002**: Verify full pipeline with all modalities
2. **sub-001**: Verify graceful handling of missing EEG stream

---

## LSL Timestamp Ranges

All timestamps are in LSL clock domain (seconds since LSL epoch):

**sub-002:**
- EEG: 604821.601 - 605481.702
- fNIRS: 604825.073 - 605936.716
- Markers: 604842.065 - 605892.059

**sub-001:**
- fNIRS: 602320.654 - 603424.240
- Markers: 602334.387 - 603384.381

**Note:** Different timestamp ranges between subjects are expected (different recording sessions).

---

## Montage Configuration

Both subjects use **identical montage configurations**:

**EEG:** 32 channels (actiCHamp)
- Standard 10-20 positions: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz
- Extended 10-10 positions: F9, F10, P9, P10
- Auxiliary: AUX_1, AUX_2, AUX_3

**fNIRS:** 42 channels (Cortivision Photon Cap C20)
- 18 source-detector pairs × 2 wavelengths (760nm, 850nm) = 36 long channels
- 3 short channel pairs × 2 wavelengths = 6 short channels
- Coverage: Bilateral sensorimotor cortex

See `data/raw/sub-00X/*.json` files for complete montage details.

---

## Quick Reference Commands

### Test ingestion module with sub-002:
```bash
micromamba run -n affective-fnirs python scripts/test_ingestion.py
```

### Load sub-002 data in Python:
```python
from pathlib import Path
from affective_fnirs.ingestion import load_xdf_file, identify_streams

xdf_path = Path("data/raw/sub-002/sub-002_tomi_ses-001_task-fingertapping_recording.xdf")
streams, header = load_xdf_file(xdf_path)
identified = identify_streams(streams)

# Access streams
eeg_stream = identified['eeg']
fnirs_stream = identified['fnirs']
markers_stream = identified['markers']
```

---

## Notes

- JSON sidecar files contain canonical montage information (identical for both subjects)
- XDF files contain actual recorded data with LSL timestamps
- Stream durations may differ due to different start/stop times during recording
- Temporal synchronization uses LSL timestamps, not fixed offsets
