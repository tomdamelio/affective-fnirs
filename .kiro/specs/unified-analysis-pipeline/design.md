# Design Document: Unified Analysis Pipeline

## Overview

This document describes the design for a unified analysis pipeline (`run_analysis.py`) that consolidates EEG and fNIRS analysis into a single, configurable script. The design leverages existing modules in `src/affective_fnirs/` as the foundation, with a thin orchestration layer that reads subject-specific configuration from YAML files.

**Design Philosophy**: Minimal new code. The script orchestrates existing functions with conditional execution based on configuration flags.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        run_analysis.py                               │
│  (CLI entry point - orchestration layer)                            │
├─────────────────────────────────────────────────────────────────────┤
│  1. Parse CLI arguments (--config, --eeg, --fnirs, --qa-only)       │
│  2. Load SubjectConfig from YAML                                     │
│  3. Conditional execution based on modality flags                    │
│  4. Call existing module functions                                   │
│  5. Save outputs to BIDS-compliant paths                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    src/affective_fnirs/                              │
├──────────────────┬──────────────────┬───────────────────────────────┤
│   config.py      │   ingestion.py   │   mne_builder.py              │
│   (extended)     │   (unchanged)    │   (unchanged)                 │
├──────────────────┼──────────────────┼───────────────────────────────┤
│ eeg_processing   │ eeg_analysis.py  │ fnirs_processing.py           │
│ (unchanged)      │ (unchanged)      │ (unchanged)                   │
├──────────────────┼──────────────────┼───────────────────────────────┤
│ fnirs_quality.py │ fnirs_analysis   │ multimodal_analysis.py        │
│ (unchanged)      │ (unchanged)      │ (unchanged)                   │
├──────────────────┴──────────────────┴───────────────────────────────┤
│                       reporting.py (unchanged)                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Extended Configuration Schema (config.py modification)

The existing `PipelineConfig` will be extended with new dataclasses for subject-specific configuration.

```python
@dataclass(frozen=True)
class SubjectInfo:
    """Subject identification for BIDS compliance."""
    id: str           # e.g., "010" (without "sub-" prefix)
    session: str      # e.g., "001" (without "ses-" prefix)
    task: str         # e.g., "fingertapping"

@dataclass(frozen=True)
class ModalityConfig:
    """Modality enable/disable flags."""
    eeg_enabled: bool = True
    fnirs_enabled: bool = True

@dataclass(frozen=True)
class ReportConfig:
    """Report generation options."""
    qa_only: bool = False

@dataclass(frozen=True)
class TrialsConfig:
    """Trial structure information."""
    count_per_condition: int = 10
    task_duration_sec: float = 10.0
    rest_duration_sec: float = 10.0

@dataclass(frozen=True)
class EEGPreprocessingConfig:
    """
    EEG preprocessing options for subject-specific configuration.
    
    Allows control over ICA, reference channel, and CAR application
    based on subject's data quality and channel count.
    
    Attributes:
        ica_enabled: Whether to apply ICA artifact removal (default: False)
            - Set to True for subjects with sufficient channels (>20)
            - Set to False for subjects with few channels or clean data
        reference_channel: Initial reference channel (default: "Cz")
            - Standard motor cortex reference for ERD/ERS studies
            - Can be changed to other channels if Cz is noisy
        apply_car: Whether to apply Common Average Reference (default: False)
            - Set to True for dense arrays with many channels
            - Set to False to keep original reference
    """
    ica_enabled: bool = False
    reference_channel: str = "Cz"
    apply_car: bool = False

@dataclass
class SubjectConfig:
    """Complete subject-specific configuration."""
    subject: SubjectInfo
    modalities: ModalityConfig
    report: ReportConfig
    eeg_channels_of_interest: list[str]
    eeg_preprocessing: EEGPreprocessingConfig  # NEW: EEG preprocessing options
    trials: TrialsConfig
    # Inherited from PipelineConfig
    filters: FilterConfig
    quality: QualityThresholds
    epochs: EpochConfig
    analysis: AnalysisConfig
    ica: ICAConfig
    motion_correction: MotionCorrectionConfig
    data_root: Path
    output_root: Path
    random_seed: int
```

### 2. CLI Interface (run_analysis.py)

```python
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified EEG/fNIRS analysis pipeline"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to subject YAML configuration file"
    )
    parser.add_argument(
        "--eeg",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override EEG processing (true/false)"
    )
    parser.add_argument(
        "--fnirs",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override fNIRS processing (true/false)"
    )
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="Generate only QA report, skip full analysis"
    )
    return parser.parse_args()
```

### 3. Pipeline Orchestration Flow

```python
def run_analysis(config: SubjectConfig) -> int:
    """
    Main analysis orchestration.
    
    Returns:
        0 on success, 1 on failure
    """
    # Stage 1: Load XDF data
    streams = load_and_identify_streams(config)
    
    # Stage 2: Build MNE objects (conditional)
    raw_eeg, raw_fnirs = build_mne_objects(streams, config)
    
    # Stage 3: Quality Assessment (always runs)
    qa_results = run_quality_assessment(raw_eeg, raw_fnirs, config)
    
    if config.report.qa_only:
        save_qa_report(qa_results, config)
        return 0
    
    # Stage 4: Preprocessing (conditional)
    processed_eeg, processed_fnirs = run_preprocessing(raw_eeg, raw_fnirs, config)
    
    # Stage 5: Analysis (conditional)
    analysis_results = run_analysis_stage(processed_eeg, processed_fnirs, config)
    
    # Stage 6: Visualizations (conditional)
    generate_visualizations(analysis_results, config)
    
    # Stage 7: Save results
    save_full_report(qa_results, analysis_results, config)
    
    return 0
```

### 4. Conditional Execution Logic

```python
def build_mne_objects(streams: dict, config: SubjectConfig) -> tuple:
    """Build MNE Raw objects based on enabled modalities."""
    raw_eeg = None
    raw_fnirs = None
    
    if config.modalities.eeg_enabled:
        if streams.get("eeg") is None:
            logger.warning("EEG enabled but stream not found, disabling")
            # Modify config to disable EEG
        else:
            eeg_data, eeg_sfreq, eeg_ts = extract_stream_data(streams["eeg"])
            raw_eeg = build_eeg_raw(eeg_data, eeg_sfreq, streams["eeg"]["info"], eeg_ts)
            raw_eeg = embed_events(raw_eeg, streams["markers"], EVENT_MAPPING)
    
    if config.modalities.fnirs_enabled:
        if streams.get("fnirs") is None:
            logger.warning("fNIRS enabled but stream not found, disabling")
        else:
            fnirs_data, fnirs_sfreq, fnirs_ts = extract_stream_data(streams["fnirs"])
            raw_fnirs = build_fnirs_raw(fnirs_data, fnirs_sfreq, ...)
            raw_fnirs = embed_events(raw_fnirs, streams["markers"], EVENT_MAPPING)
    
    return raw_eeg, raw_fnirs
```

## Data Models

### YAML Configuration File Format

```yaml
# configs/sub-010.yml
# Subject-specific configuration for sub-010

subject:
  id: "010"
  session: "001"
  task: "fingertapping"

modalities:
  eeg_enabled: true
  fnirs_enabled: false  # fNIRS not recorded for this subject

report:
  qa_only: false

eeg_channels_of_interest:
  - "C3"
  - "C4"
  - "Cz"

# NEW: EEG preprocessing options
eeg_preprocessing:
  ica_enabled: true       # Enable ICA for sub-010 (has enough channels)
  reference_channel: "Cz" # Standard motor cortex reference
  apply_car: false        # Keep Cz reference, don't apply CAR

trials:
  count_per_condition: 12
  task_duration_sec: 15.0
  rest_duration_sec: 15.0

# Standard PipelineConfig sections below
filters:
  eeg_bandpass_low_hz: 1.0
  eeg_bandpass_high_hz: 40.0
  fnirs_bandpass_low_hz: 0.01
  fnirs_bandpass_high_hz: 0.5
  cardiac_band_low_hz: 0.5
  cardiac_band_high_hz: 2.5

quality:
  sci_threshold: 0.75
  cv_threshold_percent: 15.0
  saturation_percent: 5.0
  psp_threshold: 0.1
  short_channel_distance_mm: 15.0

epochs:
  eeg_tmin_sec: -3.0
  eeg_tmax_sec: 20.0
  fnirs_tmin_sec: -3.0
  fnirs_tmax_sec: 25.0
  baseline_tmin_sec: -3.0
  baseline_tmax_sec: -1.0

analysis:
  alpha_band_low_hz: 8.0
  alpha_band_high_hz: 13.0
  beta_band_low_hz: 13.0
  beta_band_high_hz: 30.0
  baseline_window_start_sec: -3.0
  baseline_window_end_sec: -1.0
  task_window_start_sec: 1.0
  task_window_end_sec: 14.0
  beta_rebound_window_start_sec: 15.0
  beta_rebound_window_end_sec: 20.0
  hrf_onset_window_start_sec: 2.0
  hrf_onset_window_end_sec: 3.0
  hrf_peak_window_start_sec: 4.0
  hrf_peak_window_end_sec: 8.0
  dpf: 6.0

ica:
  enabled: false
  n_components: 15
  random_state: 42
  max_iter: 200

motion_correction:
  method: tddr

data_root: data/raw
output_root: data/derivatives/validation-pipeline
random_seed: 42
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: CLI Flag Override Behavior

*For any* configuration file with `eeg_enabled=X` and `fnirs_enabled=Y`, when CLI flags `--eeg=A` and `--fnirs=B` are provided, the effective modality settings SHALL be `A` and `B` respectively, overriding the config values.

**Validates: Requirements 1.2, 1.3, 1.5**

### Property 2: Configuration Default Values

*For any* YAML configuration file missing optional fields (`modalities`, `report`, `eeg_channels_of_interest`), the loaded `SubjectConfig` SHALL have default values: `eeg_enabled=true`, `fnirs_enabled=true`, `qa_only=false`, `eeg_channels_of_interest=["C3", "C4"]`.

**Validates: Requirements 2.2, 2.3, 2.4, 5.3**

### Property 3: Modality-Conditional Processing

*For any* configuration where `eeg_enabled=false`, the pipeline SHALL NOT call any EEG-specific functions (`build_eeg_raw`, `preprocess_eeg_pipeline`, `compute_tfr`, `detect_erd_ers`). Similarly, *for any* configuration where `fnirs_enabled=false`, the pipeline SHALL NOT call any fNIRS-specific functions.

**Validates: Requirements 3.1, 3.2, 3.4, 3.5**

### Property 4: QA-Only Mode Skips Analysis

*For any* configuration with `qa_only=true`, the pipeline SHALL NOT call analysis functions (`compute_tfr`, `detect_erd_ers`, `extract_hrf`) or visualization functions (`plot_condition_contrast_spectrograms`, `plot_erd_timecourse_bilateral`, `plot_hrf_curves`).

**Validates: Requirements 4.1, 4.4**

### Property 5: ERD/ERS Computed for Specified Channels

*For any* configuration with `eeg_channels_of_interest=[ch1, ch2, ...]` where all channels exist in the data, the pipeline SHALL call `detect_erd_ers()` exactly once for each channel in the list.

**Validates: Requirements 5.2**

### Property 6: Exit Code Consistency

*For any* successful pipeline execution (no exceptions raised), the exit code SHALL be 0. *For any* failed execution (exception raised), the exit code SHALL be 1.

**Validates: Requirements 7.6**

### Property 7: EEG Preprocessing Configuration Defaults

*For any* YAML configuration file missing the `eeg_preprocessing` section, the loaded `SubjectConfig` SHALL have default values: `ica_enabled=false`, `reference_channel="Cz"`, `apply_car=false`.

**Validates: Requirements 8.2, 8.3, 8.4**

### Property 8: ICA Conditional Execution

*For any* configuration where `eeg_preprocessing.ica_enabled=false`, the pipeline SHALL NOT call ICA-related functions during EEG preprocessing. *For any* configuration where `eeg_preprocessing.ica_enabled=true`, the pipeline SHALL apply ICA artifact removal.

**Validates: Requirements 8.5, 8.6**

## Error Handling

### Error Categories

1. **Configuration Errors**: Invalid YAML, missing required fields, invalid values
   - Raise `ValueError` with descriptive message
   - Exit code: 1

2. **File Not Found Errors**: Missing XDF, JSON, or config files
   - Raise `FileNotFoundError` with expected path
   - Exit code: 1

3. **Data Stream Errors**: Enabled modality but stream not found in XDF
   - Log warning, disable modality, continue processing
   - Exit code: 0 (graceful degradation)

4. **Processing Errors**: Exceptions from existing modules
   - Catch, log with stage context, re-raise
   - Exit code: 1

### Error Handling Pattern

```python
try:
    # Stage N processing
    result = existing_module_function(...)
except Exception as e:
    logger.error(f"Stage N failed: {e}")
    raise PipelineError(f"Stage N ({stage_name}) failed: {e}") from e
```

## Testing Strategy

### Unit Tests

Unit tests verify specific examples and edge cases:

1. **Config Loading**: Test YAML parsing with various field combinations
2. **CLI Parsing**: Test argument parsing with different flag combinations
3. **Error Handling**: Test specific error conditions (missing files, invalid config)

### Property-Based Tests

Property-based tests verify universal properties across many generated inputs:

1. **Property 1 (CLI Override)**: Generate random config values and CLI flags, verify override behavior
2. **Property 2 (Defaults)**: Generate configs with missing fields, verify defaults applied
3. **Property 3 (Conditional Processing)**: Mock module functions, verify call patterns based on flags
4. **Property 4 (QA-Only)**: Mock analysis functions, verify they're not called in QA mode
5. **Property 5 (Channel Selection)**: Generate channel lists, verify ERD/ERS called for each
6. **Property 6 (Exit Codes)**: Generate success/failure scenarios, verify exit codes

### Test Configuration

- Property-based testing library: `hypothesis`
- Minimum iterations per property: 100
- Test file location: `tests/test_run_analysis.py`
- Tag format: `# Feature: unified-analysis-pipeline, Property N: description`
