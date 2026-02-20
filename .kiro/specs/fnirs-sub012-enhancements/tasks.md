# Implementation Plan: fNIRS Sub-012 Enhancements

## Overview

This implementation plan breaks down the 8 requirements into discrete coding tasks for `scripts/run_analysis_sub012.py` and supporting modules. Tasks are ordered to build incrementally, with property tests placed close to their implementations.

## Tasks

- [ ] 1. Add single-modality CLI flags
  - [x] 1.1 Add `--fnirs-only` and `--eeg-only` mutually exclusive arguments to argparse
    - Create `create_argument_parser()` function with mutually exclusive group
    - Add `ModalityMode` enum to track execution mode
    - Log active modality mode at pipeline start
    - _Requirements: 1.1, 1.2, 1.6, 1.8_
  
  - [x] 1.2 Add conditional execution logic based on modality flags
    - Wrap EEG loading/processing/analysis in `if not args.eeg_only` blocks
    - Wrap fNIRS loading/processing/analysis in `if not args.fnirs_only` blocks
    - Skip multimodal analysis when either flag is set
    - _Requirements: 1.3, 1.4, 1.5, 1.7_
  
  - [ ]* 1.3 Write property test for CLI flag mutual exclusivity
    - **Property 1: CLI Flag Mutual Exclusivity**
    - **Validates: Requirements 1.1, 1.2, 1.6**

- [ ] 2. Enhance montage loading with validation
  - [x] 2.1 Create `load_sub012_montage()` function with error handling
    - Load JSON from BIDS path `data/raw/sub-012/ses-001/montage_combined_EEG_fNIRS_with_3Dcoords_approx.json`
    - Validate ChMontage key exists
    - Log error and terminate if file missing
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [ ]* 2.2 Write property test for channel type assignment
    - **Property 4: Channel Type Assignment from Montage**
    - **Validates: Requirements 2.4**

- [ ] 3. Fix channel count in QA report
  - [x] 3.1 Create `count_fnirs_channels()` function
    - Use `mne.pick_types(raw.info, fnirs=True, exclude=[])` to count only fnirs_cw_amplitude
    - Update QA results post-processing to use this function
    - _Requirements: 3.1, 3.2_
  
  - [ ]* 3.2 Write property test for channel count excluding misc
    - **Property 3: Channel Count Excludes Misc Channels**
    - **Validates: Requirements 3.1, 3.2**

- [x] 4. Checkpoint - Verify montage and channel count
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Verify SCI comparison plot with new montage
  - [x] 5.1 Verify `generate_sci_comparison_plot()` handles 24 pairs
    - Confirm plot includes one bar per source-detector pair (up to 24)
    - Confirm threshold line drawn at configured value
    - Verify BIDS naming pattern in output filename
    - _Requirements: 4.1, 4.2, 4.3_

- [ ] 6. Implement channel filtering by SCI threshold
  - [x] 6.1 Create `filter_channels_by_sci()` function
    - Override config `sci_threshold` to 0.50 at pipeline start
    - Classify channels as good (SCI > 0.50) or bad (SCI ≤ 0.50)
    - Return tuple of (good_pairs, bad_pairs) lists
    - _Requirements: 5.1, 5.2_
  
  - [x] 6.2 Create `mark_bad_channels_in_info()` function
    - Mark both wavelength channels (760nm, 850nm) for each bad pair
    - Update `raw.info['bads']` for both Raw and Epochs objects
    - _Requirements: 5.3_
  
  - [ ]* 6.3 Write property test for SCI classification and marking
    - **Property 5: SCI-Based Channel Classification and Marking**
    - **Validates: Requirements 5.2, 5.3**

- [ ] 7. Enhance timeseries plot with good channel filtering
  - [x] 7.1 Update `generate_fnirs_timeseries_plot()` to filter by good_channels
    - Filter HbO/HbR channels to include only good_channels
    - Compute mean ± std across filtered channels
    - Handle edge case: zero good channels returns None with warning
    - _Requirements: 5.4, 6.1, 6.2, 6.6_
  
  - [x] 7.2 Add stimulus markers to timeseries plot
    - Draw vertical dashed lines at annotation onsets
    - Color-code by condition (LEFT=green, RIGHT=purple, NOTHING=gray)
    - _Requirements: 6.3, 6.4, 6.5_
  
  - [ ]* 7.3 Write property test for timeseries plot structure
    - **Property 6: Timeseries Plot Structure**
    - **Validates: Requirements 6.1, 6.2**

- [x] 8. Checkpoint - Verify channel filtering and timeseries
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Enhance 4-ROI HRF plot
  - [x] 9.1 Define ROI constants and `ROIDefinition` dataclass
    - Define `ROI_DEFINITIONS` dict mapping ROI names to source prefixes
    - Left Anterior: S1_, S9_; Right Anterior: S2_, S10_
    - Left Posterior: S3_, S4_, S5_; Right Posterior: S6_, S7_, S8_
    - _Requirements: 7.2_
  
  - [x] 9.2 Update `generate_fnirs_hrf_by_condition_4roi()` with good channel filtering
    - Filter channels by good_channels list before ROI assignment
    - Handle edge case: all ROI channels bad shows empty subplot with annotation
    - _Requirements: 5.5, 7.6_
  
  - [x] 9.3 Verify 4-ROI plot structure and condition coloring
    - Confirm 4 rows × 2 columns (8 subplots)
    - Confirm condition colors (LEFT=green, RIGHT=purple, NOTHING=gray)
    - Confirm vertical dashed line at time=0
    - _Requirements: 7.1, 7.3, 7.4, 7.5_
  
  - [ ]* 9.4 Write property test for ROI channel assignment
    - **Property 9: ROI Channel Assignment and Filtering**
    - **Validates: Requirements 7.2, 7.6**

- [ ] 10. Implement SCR verification and logging
  - [x] 10.1 Define `SHORT_CHANNEL_ROI_MAP` constant
    - S13_D1 → Left Anterior (S1, S9)
    - S14_D3 → Right Anterior (S2, S10)
    - S15_D6 → Left Posterior (S3, S4, S5)
    - S16_D9 → Right Posterior (S6, S7, S8)
    - _Requirements: 8.1_
  
  - [x] 10.2 Create `verify_scr_pairing()` function
    - Compare MNE-NIRS proximity-based pairing to expected ROI mapping
    - Log actual short-to-long channel pairing
    - Return verification result with mismatches if any
    - _Requirements: 8.2, 8.3_
  
  - [x] 10.3 Create `log_scr_noise_reduction()` function
    - Compute PSD in systemic band (0.1-0.4 Hz) before and after SCR
    - Log mean power reduction per ROI cluster
    - _Requirements: 8.4_
  
  - [x] 10.4 Add fallback for incorrect pairing
    - If proximity pairing mismatches expected, log warning
    - Fall back to explicit ROI-based pairing override
    - _Requirements: 8.5_
  
  - [x] 10.5 Handle missing or bad short channels
    - Log warning identifying which ROI lacks SCR
    - Proceed without regression for that ROI only
    - _Requirements: 8.6_
  
  - [ ]* 10.6 Write property test for SCR pairing
    - **Property 10: SCR Pairing Matches Expected ROI Mapping**
    - **Validates: Requirements 8.2**
  
  - [ ]* 10.7 Write property test for SCR processing order
    - **Property 11: SCR Processing Order**
    - **Validates: Requirements 8.7**

- [x] 11. Final checkpoint - Full integration verification
  - Ensure all tests pass, ask the user if questions arise.
  - Verify pipeline runs with `--fnirs-only` flag
  - Verify pipeline runs with `--eeg-only` flag
  - Verify full pipeline produces all expected outputs

## Notes

- Tasks marked with `*` are optional property-based tests and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests use Hypothesis library (already in environment.yml)
- All file outputs follow BIDS naming conventions
