# Implementation Plan: Unified Analysis Pipeline

## Overview

This implementation plan creates a unified analysis pipeline (`run_analysis.py`) that orchestrates existing modules with subject-specific YAML configuration. The approach is minimal: extend `config.py` with new dataclasses and create a single orchestration script.

## Tasks

- [x] 1. Extend configuration schema in config.py
  - [x] 1.1 Add SubjectInfo, ModalityConfig, ReportConfig, TrialsConfig dataclasses
    - Add frozen dataclasses with validation in `__post_init__`
    - Include default values as specified in design
    - _Requirements: 2.1, 2.2, 2.3, 2.5_
  - [x] 1.2 Add SubjectConfig dataclass that composes all config sections
    - Compose existing PipelineConfig sections with new sections
    - Add `from_yaml()` class method for loading
    - Add `eeg_channels_of_interest` field with default ["C3", "C4"]
    - _Requirements: 2.4, 2.6, 2.7_
  - [ ]* 1.3 Write property test for configuration defaults
    - **Property 2: Configuration Default Values**
    - **Validates: Requirements 2.2, 2.3, 2.4, 5.3**

- [x] 2. Create run_analysis.py CLI entry point
  - [x] 2.1 Implement argument parser with --config, --eeg, --fnirs, --qa-only
    - Use argparse with required --config and optional override flags
    - --eeg and --fnirs accept "true" or "false" string values
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  - [x] 2.2 Implement main() function with config loading and flag override logic
    - Load SubjectConfig from YAML
    - Apply CLI flag overrides to modality settings
    - Print summary of enabled modalities
    - _Requirements: 1.5, 1.6_
  - [ ]* 2.3 Write property test for CLI flag override behavior
    - **Property 1: CLI Flag Override Behavior**
    - **Validates: Requirements 1.2, 1.3, 1.5**

- [x] 3. Implement data loading stage
  - [x] 3.1 Implement load_and_identify_streams() function
    - Call existing `load_xdf_file()` and `identify_streams()`
    - Handle missing streams gracefully with warnings
    - _Requirements: 3.6_
  - [x] 3.2 Implement build_mne_objects() with conditional EEG/fNIRS building
    - Call `build_eeg_raw()` only if eeg_enabled and stream exists
    - Call `build_fnirs_raw()` only if fnirs_enabled and stream exists
    - Call `embed_events()` for each built Raw object
    - _Requirements: 3.1, 3.2, 3.4, 3.5_
  - [ ]* 3.3 Write property test for modality-conditional processing
    - **Property 3: Modality-Conditional Processing**
    - **Validates: Requirements 3.1, 3.2, 3.4, 3.5**

- [x] 4. Implement quality assessment stage
  - [x] 4.1 Implement run_quality_assessment() function
    - Call `compute_eeg_channel_quality()` if EEG enabled
    - Call fNIRS quality functions (`calculate_sci`, etc.) if fNIRS enabled
    - Return combined QA results
    - _Requirements: 4.2, 4.3_
  - [x] 4.2 Implement save_qa_report() function
    - Save QA results as JSON using existing format
    - Use BIDS-compliant filename
    - _Requirements: 4.5, 6.1, 6.2_

- [x] 5. Implement preprocessing stage
  - [x] 5.1 Implement run_preprocessing() function
    - Call `preprocess_eeg_pipeline()` if EEG enabled
    - Call `process_fnirs_pipeline()` if fNIRS enabled
    - Preserve annotations through preprocessing
    - _Requirements: 3.1, 3.2_

- [x] 6. Implement analysis stage
  - [x] 6.1 Implement run_eeg_analysis() function
    - Create epochs using `mne.events_from_annotations()` and `mne.Epochs()`
    - Call `compute_tfr()` for time-frequency analysis
    - Call `detect_erd_ers()` for each channel in `eeg_channels_of_interest`
    - _Requirements: 5.2, 5.4_
  - [x] 6.2 Implement run_fnirs_analysis() function
    - Call `create_fnirs_epochs()` and `extract_hrf()`
    - _Requirements: 3.2_
  - [x] 6.3 Implement run_multimodal_analysis() function
    - Call `compute_neurovascular_coupling()` if both modalities enabled
    - _Requirements: 3.3_
  - [ ]* 6.4 Write property test for ERD/ERS channel selection
    - **Property 5: ERD/ERS Computed for Specified Channels**
    - **Validates: Requirements 5.2**

- [x] 7. Implement visualization stage
  - [x] 7.1 Implement generate_visualizations() function
    - Call `plot_condition_contrast_spectrograms()` if EEG enabled
    - Call `plot_erd_timecourse_bilateral()` if EEG enabled
    - Call `plot_hrf_curves()` if fNIRS enabled
    - Save figures to BIDS-compliant paths
    - _Requirements: 6.3, 6.4, 6.5_
  - [ ]* 7.2 Write property test for QA-only mode skipping analysis
    - **Property 4: QA-Only Mode Skips Analysis**
    - **Validates: Requirements 4.1, 4.4**

- [x] 8. Implement error handling and logging
  - [x] 8.1 Add logging configuration matching pipeline.py format
    - Use Python logging module with timestamp format
    - Log progress for each stage
    - _Requirements: 7.1, 7.2_
  - [x] 8.2 Implement error handling with stage context
    - Catch exceptions from modules, add stage context, re-raise
    - Return exit code 0 on success, 1 on failure
    - _Requirements: 7.3, 7.4, 7.5, 7.6_
  - [ ]* 8.3 Write property test for exit code consistency
    - **Property 6: Exit Code Consistency**
    - **Validates: Requirements 7.6**

- [x] 9. Checkpoint - Verify implementation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Extend EEG preprocessing configuration options
  - [x] 10.1 Add EEGPreprocessingConfig dataclass to config.py
    - Add `reference_channel` field (default: "Cz") for initial reference
    - Add `apply_car` field (default: false) for Common Average Reference
    - Add `ica_enabled` field (default: false) for ICA artifact removal
    - Validate reference_channel is a valid EEG channel name
    - _Requirements: 2.2, 3.1_
  - [x] 10.2 Update SubjectConfig to include eeg_preprocessing section
    - Add `eeg_preprocessing: EEGPreprocessingConfig` field
    - Update `from_dict()` and `from_yaml()` to parse new section
    - Ensure backward compatibility with existing configs (use defaults)
    - _Requirements: 2.6, 2.7_
  - [x] 10.3 Update run_analysis.py preprocessing stage
    - Pass `ica_enabled` from eeg_preprocessing to control ICA execution
    - Pass `reference_channel` for initial reference (before preprocessing)
    - Pass `apply_car` to control CAR application (after preprocessing)
    - Log preprocessing options at start of EEG preprocessing
    - _Requirements: 3.1, 7.2_
  - [x] 10.4 Update eeg_processing.py to support configurable reference
    - Modify `preprocess_eeg_pipeline()` to accept reference options
    - Add parameter for initial reference channel
    - Add parameter to enable/disable CAR
    - Maintain backward compatibility with existing behavior
    - _Requirements: 3.1_
  - [x] 10.5 Update YAML configs with new eeg_preprocessing section
    - Add eeg_preprocessing section to test_unified.yml (defaults)
    - Add eeg_preprocessing section to sub010_optimized.yml with ica_enabled: true
    - _Requirements: 2.1-2.6_

- [x] 11. Create example subject configuration files
  - [x] 11.1 Create configs/sub-010.yml with EEG-only configuration
    - Set fnirs_enabled: false
    - Set eeg_channels_of_interest: ["C3", "C4"]
    - Set eeg_preprocessing with ica_enabled: true (sub-010 has enough channels)
    - Set trials for 12 per condition, 15s task, 15s rest
    - _Requirements: 2.1-2.6_
  - [x] 11.2 Update existing configs to new schema format
    - Add subject, modalities, report, trials sections to existing configs
    - Add eeg_preprocessing section with appropriate defaults
    - Preserve existing filter/quality/epoch/analysis settings
    - _Requirements: 2.1-2.6_

- [x] 12. Final checkpoint - End-to-end validation
  - Run `run_analysis.py --config configs/sub-010.yml` and verify outputs
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based tests
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- The implementation reuses existing modules - minimal new code required
