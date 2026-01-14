# Requirements Document

## Introduction

This document specifies requirements for a unified analysis pipeline (`run_analysis.py`) that processes EEG and/or fNIRS data from finger-tapping experiments. The pipeline consolidates multiple subject-specific scripts into a single, configurable entry point that reads all parameters from a per-subject YAML configuration file.

**Key Design Principle**: This pipeline reuses the existing modules in `src/affective_fnirs/` as the foundation. The new script is a thin orchestration layer that calls existing functions with subject-specific configuration.

## Existing Modules to Reuse

The following modules from `src/affective_fnirs/` provide the core functionality:

- **config.py**: `PipelineConfig` dataclass with all filter, quality, epoch, analysis, ICA, and motion correction parameters
- **ingestion.py**: `load_xdf_file()`, `identify_streams()`, `extract_stream_data()` for XDF loading
- **mne_builder.py**: `build_eeg_raw()`, `build_fnirs_raw()`, `embed_events()` for MNE object construction
- **eeg_processing.py**: `preprocess_eeg_pipeline()` for EEG preprocessing (filter, ICA, CAR)
- **eeg_analysis.py**: `compute_tfr()`, `detect_erd_ers()`, `compute_tfr_by_condition()`, `plot_condition_contrast_spectrograms()`, `plot_erd_timecourse_bilateral()` for ERD/ERS analysis
- **fnirs_processing.py**: `process_fnirs_pipeline()` for fNIRS preprocessing
- **fnirs_quality.py**: `calculate_sci()`, `detect_saturation()`, `assess_cardiac_power()`, `calculate_coefficient_of_variation()`, `mark_bad_channels()` for fNIRS QA
- **fnirs_analysis.py**: `create_fnirs_epochs()`, `extract_hrf()`, `plot_hrf_curves()` for HRF analysis
- **reporting.py**: `compute_eeg_channel_quality()`, `QualityReport`, `ValidationResults` for reporting
- **multimodal_analysis.py**: `compute_neurovascular_coupling()` for cross-modal analysis

## Glossary

- **Pipeline**: The unified analysis script (`run_analysis.py`) that orchestrates existing modules
- **Subject_Config**: Extended YAML configuration file with subject-specific parameters plus modality flags
- **QA_Report**: Quality Assessment report using existing `QualityReport` dataclass
- **Full_Report**: Complete analysis using existing `ValidationResults` dataclass
- **Modality_Flag**: Boolean flag (`eeg_enabled`, `fnirs_enabled`) controlling which processing stages run

## Requirements

### Requirement 1: Unified Command-Line Interface

**User Story:** As a researcher, I want a single script that can analyze any subject's data, so that I don't need separate scripts per subject or modality.

#### Acceptance Criteria

1. THE Pipeline SHALL accept a `--config` argument specifying the path to a subject's YAML configuration file
2. THE Pipeline SHALL accept an optional `--eeg` flag with values `true` or `false` to override the config's `eeg_enabled` setting (e.g., `--eeg=false`)
3. THE Pipeline SHALL accept an optional `--fnirs` flag with values `true` or `false` to override the config's `fnirs_enabled` setting (e.g., `--fnirs=false`)
4. THE Pipeline SHALL accept an optional `--qa-only` flag to generate only the QA report without full analysis
5. WHEN no override flags are provided, THE Pipeline SHALL use the modality settings from the YAML configuration file
6. THE Pipeline SHALL print a summary of enabled modalities and QA-only mode before starting

### Requirement 2: Extended Subject Configuration Schema

**User Story:** As a researcher, I want all subject-specific parameters in a single YAML file, so that I can easily configure and reproduce analyses.

#### Acceptance Criteria

1. THE Subject_Config SHALL extend the existing `PipelineConfig` schema with a new `subject` section containing `id`, `session`, and `task`
2. THE Subject_Config SHALL include a `modalities` section with `eeg_enabled` (boolean, default true) and `fnirs_enabled` (boolean, default true)
3. THE Subject_Config SHALL include a `report` section with `qa_only` (boolean, default false)
4. THE Subject_Config SHALL include an `eeg_channels_of_interest` list (default: ["C3", "C4"])
5. THE Subject_Config SHALL include a `trials` section with `count_per_condition`, `task_duration_sec`, and `rest_duration_sec`
6. THE Subject_Config SHALL include all existing `PipelineConfig` sections (filters, quality, epochs, analysis, ica, motion_correction)
7. THE Pipeline SHALL use `PipelineConfig.from_yaml()` to load the base configuration and extend it with new fields

### Requirement 3: Modality-Conditional Processing Using Existing Modules

**User Story:** As a researcher, I want to process only the modalities that have valid data, reusing the existing processing functions.

#### Acceptance Criteria

1. WHEN `eeg_enabled` is true, THE Pipeline SHALL call `build_eeg_raw()`, `embed_events()`, `preprocess_eeg_pipeline()`, and EEG analysis functions
2. WHEN `fnirs_enabled` is true, THE Pipeline SHALL call `build_fnirs_raw()`, `embed_events()`, `process_fnirs_pipeline()`, and fNIRS analysis functions
3. WHEN both modalities are enabled, THE Pipeline SHALL call `compute_neurovascular_coupling()` from multimodal_analysis
4. WHEN only EEG is enabled, THE Pipeline SHALL skip all fNIRS function calls
5. WHEN only fNIRS is enabled, THE Pipeline SHALL skip all EEG function calls
6. IF a modality is enabled but `identify_streams()` returns None for that stream, THEN THE Pipeline SHALL log a warning and disable that modality

### Requirement 4: QA-Only Mode Using Existing Quality Functions

**User Story:** As a researcher, I want to quickly check data quality using the existing quality assessment functions.

#### Acceptance Criteria

1. WHEN `qa_only` is true, THE Pipeline SHALL call only data loading and quality assessment functions
2. WHEN `qa_only` is true with EEG enabled, THE Pipeline SHALL call `compute_eeg_channel_quality()` from reporting.py
3. WHEN `qa_only` is true with fNIRS enabled, THE Pipeline SHALL call `calculate_sci()`, `detect_saturation()`, `assess_cardiac_power()`, `calculate_coefficient_of_variation()` from fnirs_quality.py
4. WHEN `qa_only` is true, THE Pipeline SHALL skip `compute_tfr()`, `detect_erd_ers()`, `extract_hrf()`, and visualization functions
5. THE Pipeline SHALL save QA results using the existing JSON serialization format

### Requirement 5: EEG Channel Selection for ERD/ERS Analysis

**User Story:** As a researcher, I want to specify which EEG channels to analyze, using the existing ERD/ERS detection functions.

#### Acceptance Criteria

1. THE Subject_Config SHALL specify `eeg_channels_of_interest` as a list of channel names
2. THE Pipeline SHALL call `detect_erd_ers()` for each channel in `eeg_channels_of_interest`
3. WHEN `eeg_channels_of_interest` is empty or not provided, THE Pipeline SHALL default to ["C3", "C4"]
4. IF a channel is not found in the data, THEN THE Pipeline SHALL log a warning and skip that channel
5. THE Pipeline SHALL pass the channel list to `plot_erd_timecourse_bilateral()` for visualization

### Requirement 6: Output File Generation Using Existing Reporting Functions

**User Story:** As a researcher, I want BIDS-compliant outputs using the existing reporting infrastructure.

#### Acceptance Criteria

1. THE Pipeline SHALL use `generate_derivative_path()` from bids_utils.py to create output directories
2. THE Pipeline SHALL save QA summary using the existing JSON format from reporting.py
3. WHEN full analysis runs with EEG, THE Pipeline SHALL call `plot_condition_contrast_spectrograms()` and save the figure
4. WHEN full analysis runs with EEG, THE Pipeline SHALL call `plot_erd_timecourse_bilateral()` and save the figure
5. WHEN full analysis runs with fNIRS, THE Pipeline SHALL call `plot_hrf_curves()` and save the figure
6. THE Pipeline SHALL log all generated file paths at completion

### Requirement 7: Error Handling and Logging

**User Story:** As a researcher, I want clear error messages and progress logging consistent with existing pipeline behavior.

#### Acceptance Criteria

1. THE Pipeline SHALL use Python's logging module with the same format as pipeline.py
2. THE Pipeline SHALL log progress messages for each major stage (loading, preprocessing, analysis, visualization)
3. IF the XDF file is not found, THEN THE Pipeline SHALL raise FileNotFoundError with the expected path
4. IF the configuration file is not found, THEN THE Pipeline SHALL raise FileNotFoundError with the path
5. THE Pipeline SHALL catch exceptions from existing modules and re-raise with context about which stage failed
6. THE Pipeline SHALL return exit code 0 on success and 1 on failure

### Requirement 8: Configurable EEG Preprocessing Options

**User Story:** As a researcher, I want to configure EEG preprocessing options (ICA, reference channel, CAR) per subject, so that I can optimize processing for subjects with different channel counts.

#### Acceptance Criteria

1. THE Subject_Config SHALL include an `eeg_preprocessing` section with configurable options
2. THE `eeg_preprocessing` section SHALL include `ica_enabled` (boolean, default: false) to control ICA artifact removal
3. THE `eeg_preprocessing` section SHALL include `reference_channel` (string, default: "Cz") for initial EEG reference
4. THE `eeg_preprocessing` section SHALL include `apply_car` (boolean, default: false) to control Common Average Reference application
5. WHEN `ica_enabled` is true, THE Pipeline SHALL apply ICA artifact removal during EEG preprocessing
6. WHEN `ica_enabled` is false, THE Pipeline SHALL skip ICA and proceed with filtering only
7. THE Pipeline SHALL apply the specified `reference_channel` before other preprocessing steps
8. WHEN `apply_car` is true, THE Pipeline SHALL apply Common Average Reference after preprocessing
9. WHEN `apply_car` is false, THE Pipeline SHALL keep the original reference (no re-referencing)
10. THE Pipeline SHALL log the EEG preprocessing configuration at the start of the preprocessing stage
