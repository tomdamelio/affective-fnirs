# Requirements Document: Multimodal Validation Pipeline

## Introduction

This document specifies the requirements for a validation and analysis pipeline for simultaneous EEG + fNIRS recordings during a finger tapping task. The system validates signal quality and neurovascular coupling in pilot data, following BIDS standards and neuroscience best practices.

## Glossary

- **System**: The multimodal validation pipeline software
- **XDF_File**: Extensible Data Format file containing synchronized multi-stream recordings
- **Stream**: A single data source within an XDF file (EEG, fNIRS, or Markers)
- **SCI**: Scalp Coupling Index - metric quantifying optical coupling quality (range 0-1)
- **CV**: Coefficient of Variation - metric detecting noise or saturation (percentage)
- **ERD**: Event-Related Desynchronization - decrease in oscillatory power during motor activity
- **HRF**: Hemodynamic Response Function - blood oxygenation change following neural activity
- **HbO**: Oxygenated hemoglobin concentration
- **HbR**: Deoxygenated hemoglobin concentration
- **Epoch**: Time-locked segment of continuous data around an event marker
- **TFR**: Time-Frequency Representation - spectral power across time
- **MNE_Object**: Data structure from MNE-Python library (Raw, Epochs, Evoked)
- **Channel_Montage**: Spatial arrangement of sensors with anatomical labels
- **BIDS**: Brain Imaging Data Structure - standardized neuroimaging data organization
- **Optical_Density**: Logarithmic transformation of light intensity (Beer-Lambert intermediate)
- **Beer_Lambert_Law**: Physical principle converting optical density to hemoglobin concentration

## Requirements

### Requirement 1: XDF Data Ingestion

**User Story:** As a neuroscientist, I want to load synchronized multimodal data from XDF files, so that I can analyze temporally aligned EEG and fNIRS signals.

#### Acceptance Criteria

1. WHEN an XDF file path is provided, THE System SHALL identify all available streams within the file
2. WHEN multiple streams exist, THE System SHALL extract EEG stream, fNIRS stream, and Markers stream separately
3. WHEN stream identification fails, THE System SHALL return a descriptive error listing available stream names
4. WHEN streams are extracted, THE System SHALL preserve original sampling rates and timestamps for each stream
5. THE System SHALL validate that extracted streams contain non-empty data arrays before proceeding

### Requirement 2: MNE Object Construction

**User Story:** As a researcher, I want raw data converted to MNE-Python objects with proper metadata, so that I can use standard neuroscience analysis tools.

#### Acceptance Criteria

1. WHEN EEG stream data is extracted, THE System SHALL construct an MNE Raw object with channel names from the JSON sidecar
2. WHEN constructing EEG Raw objects, THE System SHALL apply the standard 10-20 montage with anatomical positions
3. WHEN fNIRS stream data is extracted, THE System SHALL construct an MNE Raw object with wavelength information (760nm, 850nm)
4. WHEN constructing fNIRS Raw objects, THE System SHALL map channel indices from JSON metadata to data matrix columns
5. IF channel index ordering mismatches between JSON and data matrix, THEN THE System SHALL raise a validation error with specific channel details
6. WHEN MNE objects are created, THE System SHALL embed event markers with accurate timestamps synchronized to the data

### Requirement 3: fNIRS Quality Assessment

**User Story:** As a data quality analyst, I want automated quality metrics for fNIRS channels, so that I can identify and exclude poor-quality optodes before analysis.

#### Acceptance Criteria

1. WHEN calculating SCI, THE System SHALL apply a bandpass filter between 0.5 Hz and 2.5 Hz to isolate cardiac pulsation
2. WHEN calculating SCI for a channel pair, THE System SHALL compute Pearson correlation between 760nm and 850nm filtered signals
3. IF SCI value is less than 0.5, THEN THE System SHALL mark the channel as BAD and exclude it from further analysis
4. WHEN calculating CV, THE System SHALL compute (standard_deviation / mean) × 100 on unfiltered raw signal
5. IF CV value exceeds 10 percent, THEN THE System SHALL mark the channel as BAD and exclude it from further analysis
6. THE System SHALL generate a quality report listing all channels with their SCI and CV values
7. WHEN quality assessment completes, THE System SHALL create a visual heatmap showing spatial distribution of good and bad channels

### Requirement 4: EEG Preprocessing and ERD Analysis

**User Story:** As a motor neuroscience researcher, I want to detect event-related desynchronization in motor cortex, so that I can validate task-related neural activity.

#### Acceptance Criteria

1. WHEN preprocessing EEG data, THE System SHALL apply a bandpass filter between 1 Hz and 40 Hz
2. WHEN re-referencing EEG, THE System SHALL apply average reference across all scalp electrodes
3. WHEN creating epochs, THE System SHALL extract time windows from -5 seconds to +20 seconds around task onset markers
4. WHEN baseline correcting epochs, THE System SHALL use the -5 to 0 second pre-stimulus interval as baseline
5. WHEN computing time-frequency representations, THE System SHALL use Morlet wavelets across 3 to 30 Hz frequency range
6. WHEN analyzing motor tasks with right hand movement, THE System SHALL focus analysis on channel C3 (left motor cortex)
7. THE System SHALL generate TFR plots showing power changes in Alpha (8-13 Hz) and Beta (15-25 Hz) bands
8. WHEN ERD is detected, THE System SHALL verify power decrease begins at task onset (t=0s) and persists during the 15-second task block

### Requirement 5: fNIRS Hemodynamic Response Analysis

**User Story:** As a hemodynamics researcher, I want to extract and visualize hemoglobin concentration changes, so that I can validate neurovascular coupling during motor tasks.

#### Acceptance Criteria

1. WHEN converting fNIRS data, THE System SHALL transform raw intensity to optical density
2. WHEN applying quality filtering, THE System SHALL exclude channels marked as BAD from SCI and CV assessment
3. WHEN applying Beer-Lambert Law, THE System SHALL convert optical density to HbO and HbR concentrations
4. WHEN filtering hemodynamic signals, THE System SHALL apply a bandpass filter between 0.01 Hz and 0.2 Hz
5. WHEN creating fNIRS epochs, THE System SHALL extract time windows from -5 seconds to +25 seconds around task onset
6. WHEN analyzing motor cortex activation, THE System SHALL select channels anatomically closest to C3 location
7. THE System SHALL generate averaged hemodynamic response plots showing HbO increase and HbR decrease
8. WHEN validating hemodynamic lag, THE System SHALL verify HbO peak occurs between 6 and 8 seconds post-stimulus, not at t=0

### Requirement 6: Multimodal Neurovascular Coupling Analysis

**User Story:** As a systems neuroscientist, I want to quantify temporal relationships between neural and vascular signals, so that I can validate neurovascular coupling mechanisms.

#### Acceptance Criteria

1. WHEN extracting EEG envelope, THE System SHALL filter EEG signal in Alpha band (8-12 Hz) and apply Hilbert transform
2. WHEN preparing signals for correlation, THE System SHALL resample EEG envelope to match fNIRS sampling rate
3. WHEN computing cross-correlation, THE System SHALL correlate HbO time series with inverted Alpha envelope at C3
4. THE System SHALL identify the lag value producing maximum correlation between EEG and fNIRS signals
5. WHEN neurovascular coupling is present, THE System SHALL verify negative lag indicating EEG changes precede HbO changes
6. THE System SHALL generate overlay plots showing temporal alignment of Alpha power and HbO concentration

### Requirement 7: Validation Report Generation

**User Story:** As a principal investigator, I want a comprehensive quality report for pilot data, so that I can make informed decisions about experimental protocols.

#### Acceptance Criteria

1. THE System SHALL generate a report containing a table of fNIRS channel quality with SCI values
2. THE System SHALL include EEG spectrograms for channel C3 showing clear ERD patterns
3. THE System SHALL include hemodynamic response curves with HbO and HbR traces and standard deviation shading
4. THE System SHALL include neurovascular coupling plots with superimposed EEG and fNIRS time series
5. WHEN generating reports, THE System SHALL output results in HTML or Jupyter Notebook format
6. THE System SHALL include a conclusions section summarizing whether validation criteria were met
7. WHERE initial dip is detected in HbO signal, THE System SHALL flag this as high-quality indicator in the report

### Requirement 8: BIDS Compliance and Data Integrity

**User Story:** As a data manager, I want all processing to respect BIDS standards and data immutability, so that raw data remains protected and outputs are standardized.

#### Acceptance Criteria

1. THE System SHALL never open files in data/raw directory with write permissions
2. WHEN generating derivative data, THE System SHALL write outputs to data/derivatives/validation-pipeline directory
3. WHEN creating output filenames, THE System SHALL follow BIDS naming conventions with key-value pairs
4. WHEN generating tabular outputs, THE System SHALL create accompanying JSON data dictionaries describing columns
5. THE System SHALL validate that all input file paths follow BIDS entity ordering (sub-XX_ses-XX_task-XX)
6. IF BIDS validation fails, THEN THE System SHALL provide specific guidance on correct naming format

### Requirement 9: Configuration and Reproducibility

**User Story:** As a computational researcher, I want configurable analysis parameters with deterministic outputs, so that results are reproducible across runs.

#### Acceptance Criteria

1. THE System SHALL accept configuration files specifying filter parameters, epoch windows, and frequency bands
2. WHEN stochastic operations are performed, THE System SHALL use a configurable random seed
3. THE System SHALL log the random seed value in output metadata for reproducibility
4. WHEN configuration parameters are used, THE System SHALL save a copy of the configuration alongside results
5. THE System SHALL validate that all required dependencies match versions specified in environment.yml
6. IF a required library is missing from environment.yml, THEN THE System SHALL raise an error before execution

### Requirement 10: Error Handling and Diagnostics

**User Story:** As a pipeline user, I want clear error messages and diagnostic information, so that I can troubleshoot issues efficiently.

#### Acceptance Criteria

1. WHEN file loading fails, THE System SHALL report the specific file path and reason for failure
2. WHEN stream identification fails, THE System SHALL list all available stream names in the error message
3. WHEN channel mapping fails, THE System SHALL report which channels have mismatched indices
4. WHEN quality thresholds eliminate all channels, THE System SHALL warn the user and suggest threshold adjustments
5. WHEN marker events are missing, THE System SHALL report expected marker names and what was found
6. THE System SHALL provide progress indicators for long-running operations (filtering, epoching, TFR computation)
