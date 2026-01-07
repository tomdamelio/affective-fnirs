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
- **ICA**: Independent Component Analysis - blind source separation technique for artifact removal
- **EOG**: Electrooculography - electrical signals from eye movements
- **EMG**: Electromyography - electrical signals from muscle activity

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
2. WHEN calculating SCI for a channel pair, THE System SHALL compute zero-lag cross-correlation between 760nm and 850nm filtered signals within each 15-second task block
3. IF mean SCI value across all task blocks is less than a configurable threshold (default: 0.5), THEN THE System SHALL mark the channel as BAD and exclude it from further analysis
4. WHEN detecting signal saturation, THE System SHALL identify samples where raw intensity exceeds 95 percent of maximum ADC range
5. IF more than 5 percent of samples show saturation within any task block, THEN THE System SHALL mark the channel as BAD
6. WHEN assessing cardiac pulsation strength, THE System SHALL compute power spectral density in the 0.5-2.5 Hz band for each task block
7. IF mean cardiac peak power across blocks is below a configurable threshold (default: 0.1 normalized power), THEN THE System SHALL flag the channel as potentially poor quality
8. WHERE Coefficient of Variation is calculated, THE System SHALL restrict this calculation to baseline periods only to avoid flagging strong brain activation as noise
9. THE System SHALL generate a quality report listing all channels with their SCI values, saturation percentage, and cardiac power
10. WHEN quality assessment completes, THE System SHALL create a visual heatmap showing spatial distribution of good and bad channels

### Requirement 4: fNIRS Motion Artifact Correction and Short Channel Regression

**User Story:** As a signal processing specialist, I want to detect and correct motion artifacts and systemic interference, so that cortical hemodynamic signals are not contaminated by superficial physiology.

#### Acceptance Criteria

1. WHEN correcting motion artifacts, THE System SHALL support two methods: Temporal Derivative Distribution Repair (TDDR) or wavelet-based filtering
2. WHEN using TDDR method, THE System SHALL apply the complete TDDR algorithm which detects outliers in temporal derivatives and repairs them using distribution-based correction
3. WHEN using wavelet method, THE System SHALL decompose signals into wavelet coefficients, identify artifact components, and reconstruct cleaned signals
4. THE System SHALL provide a configuration option to select motion correction method (TDDR, wavelet, or none)
5. WHEN short channels are available in the montage, THE System SHALL identify them by source-detector separation less than 15 mm
6. WHEN performing short channel regression, THE System SHALL include short-channel signals as nuisance regressors within the GLM design matrix rather than performing separate subtraction
7. THE System SHALL apply short channel regression separately for each wavelength (760nm and 850nm)
8. WHEN short channel regression is applied, THE System SHALL verify that residual signals show reduced systemic oscillations in 0.1-0.4 Hz band (Mayer waves and respiration)
9. THE System SHALL provide a configuration option to enable or disable short channel regression independently from motion correction
10. WHEN motion correction is applied, THE System SHALL log the correction method used and the number of samples corrected per channel

### Requirement 5: EEG Preprocessing and ERD Analysis

**User Story:** As a motor neuroscience researcher, I want to detect event-related desynchronization (ERD) in the motor cortex and subsequent synchronization (ERS), so that I can validate task-related neural activity against established motor patterns.

#### Acceptance Criteria

1. WHEN preprocessing EEG data, THE System SHALL apply a bandpass filter between 1 Hz and 40 Hz to remove drift and line noise
2. WHEN removing artifacts, THE System SHALL apply Independent Component Analysis (ICA) to identify and remove ocular (EOG) and muscular (EMG) artifacts
3. WHEN performing ICA, THE System SHALL use a minimum of 15 components and fit the ICA on filtered continuous data before epoching
4. WHEN identifying artifact components, THE System SHALL detect EOG components by correlation with frontal channels (Fp1, Fp2) or dedicated EOG channels if available
5. WHEN identifying artifact components, THE System SHALL detect EMG components by high-frequency power concentration above 20 Hz
6. THE System SHALL provide visualization of ICA components with topographic maps and time courses to support manual review if needed
7. WHEN re-referencing EEG, THE System SHALL exclude or interpolate channels marked as BAD before applying the Common Average Reference (CAR) to prevent noise propagation
8. WHEN creating epochs, THE System SHALL extract time windows from -5 seconds to +20 seconds around task onset markers to accommodate edge artifacts in wavelet convolution
9. WHEN computing Time-Frequency Representations (TFR), THE System SHALL use Morlet wavelets across the 3 to 30 Hz frequency range
10. WHEN generating TFR plots, THE System SHALL apply baseline correction (mode: percent or logratio) using the -5 to -1 second pre-stimulus interval to normalize power spectra across frequencies
11. WHEN analyzing motor tasks with right hand movement, THE System SHALL focus analysis on channel C3 (left motor cortex) and optionally C1 or CP3 if available
12. THE System SHALL generate TFR plots visualizing Mu-Alpha Band (8-13 Hz) sustained power decrease (ERD) during the task and Beta Band (15-30 Hz) power decrease (ERD) during movement potentially followed by power increase (ERS-Rebound) after task cessation
13. WHEN validating the signal, THE System SHALL verify that mean power in the Alpha band (8-13 Hz) during the task block (t=1s to t=14s) is statistically lower than the baseline power (negative change)

### Requirement 6: fNIRS Hemodynamic Response Analysis

**User Story:** As a hemodynamics researcher, I want to extract and visualize hemoglobin concentration changes with proper temporal dynamics, so that I can validate neurovascular coupling patterns (HbO increase with correct onset latency, HbR decrease) during motor tasks.

#### Acceptance Criteria

1. WHEN converting fNIRS data, THE System SHALL transform raw intensity to optical density (OD)
2. WHEN applying quality filtering, THE System SHALL exclude channels marked as BAD from previous SCI and saturation assessments
3. WHEN converting to concentration, THE System SHALL apply the Modified Beer-Lambert Law (MBLL) using standard Differential Pathlength Factors (DPF) (default: 6.0 or age-dependent tables) to calculate Delta HbO and Delta HbR
4. WHEN filtering hemodynamic signals, THE System SHALL apply a bandpass filter between 0.01 Hz and 0.5 Hz to remove slow drift and high-frequency noise after concentration conversion
5. WHEN creating fNIRS epochs, THE System SHALL extract time windows from -5 seconds to +30 seconds around task onset to allow sufficient time for signal return to baseline
6. WHEN processing epochs, THE System SHALL apply baseline correction by subtracting the mean concentration of the pre-stimulus interval (-5 to 0 seconds) from the entire epoch
7. WHEN analyzing motor cortex activation, THE System SHALL select channels anatomically closest to C3 location
8. THE System SHALL generate averaged hemodynamic response plots showing the canonical response: HbO concentration increase and HbR concentration decrease relative to baseline
9. WHEN validating HRF temporal dynamics, THE System SHALL verify that HbO concentration begins to increase within 2 to 3 seconds after task onset (onset latency check)
10. WHEN validating the response shape for the 15-second task, THE System SHALL verify that mean HbO amplitude during the task plateau (t=5s to t=15s) is significantly positive compared to baseline rather than looking for a single peak at 6 seconds
11. WHEN assessing HRF quality, THE System SHALL compute the time-to-peak for HbO response and verify it falls within the physiologically plausible range of 4 to 8 seconds post-stimulus
12. THE System SHALL provide diagnostic plots showing individual trial HRF curves overlaid with the grand average to assess response consistency across trials

### Requirement 7: Multimodal Neurovascular Coupling Analysis

**User Story:** As a systems neuroscientist, I want to quantify temporal relationships between neural and vascular signals, so that I can validate neurovascular coupling mechanisms.

#### Acceptance Criteria

1. WHEN extracting EEG envelope, THE System SHALL filter EEG signal in Alpha band (8-12 Hz) and apply Hilbert transform
2. WHEN preparing signals for correlation, THE System SHALL apply a low-pass filter (less than 0.5 Hz) to the EEG envelope to match the hemodynamic frequency content and then resample it to match the fNIRS sampling rate
3. WHEN computing cross-correlation, THE System SHALL correlate HbO time series with inverted Alpha envelope at C3
4. THE System SHALL identify the lag value producing maximum correlation between EEG and fNIRS signals
5. WHEN neurovascular coupling is present, THE System SHALL verify negative lag indicating EEG changes precede HbO changes
6. THE System SHALL generate overlay plots showing temporal alignment of Alpha power and HbO concentration

### Requirement 8: Validation Report Generation

**User Story:** As a principal investigator, I want a comprehensive quality report for pilot data, so that I can make informed decisions about experimental protocols.

#### Acceptance Criteria

1. THE System SHALL generate a report containing a table of fNIRS channel quality with SCI values
2. THE System SHALL include EEG spectrograms for channel C3 showing clear ERD patterns
3. THE System SHALL include hemodynamic response curves with HbO and HbR traces and standard deviation shading
4. THE System SHALL include neurovascular coupling plots with superimposed EEG and fNIRS time series
5. WHEN generating reports, THE System SHALL output results in HTML or Jupyter Notebook format
6. THE System SHALL include a conclusions section summarizing whether validation criteria were met
7. THE System MAY optionally annotate the presence of an initial dip (HbO decrease in t=0 to t=2s) but lack thereof shall not penalize the quality score

### Requirement 9: BIDS Compliance and Data Integrity

**User Story:** As a data manager, I want all processing to respect BIDS standards and data immutability, so that raw data remains protected and outputs are standardized.

#### Acceptance Criteria

1. THE System SHALL never open files in data/raw directory with write permissions
2. WHEN generating derivative data, THE System SHALL write outputs to data/derivatives/validation-pipeline directory
3. WHEN creating output filenames, THE System SHALL follow BIDS naming conventions with key-value pairs
4. WHEN generating tabular outputs, THE System SHALL create accompanying JSON data dictionaries describing columns
5. THE System SHALL validate that all input file paths follow BIDS entity ordering (sub-XX_ses-XX_task-XX)
6. IF BIDS validation fails, THEN THE System SHALL provide specific guidance on correct naming format

### Requirement 10: Configuration and Reproducibility

**User Story:** As a computational researcher, I want configurable analysis parameters with deterministic outputs, so that results are reproducible across runs.

#### Acceptance Criteria

1. THE System SHALL accept configuration files specifying filter parameters, epoch windows, frequency bands, and quality assessment thresholds (SCI, saturation, cardiac power, correlation limits)
2. WHEN stochastic operations are performed, THE System SHALL use a configurable random seed
3. THE System SHALL log the random seed value in output metadata for reproducibility
4. WHEN configuration parameters are used, THE System SHALL save a copy of the configuration alongside results
5. THE System SHALL validate that all required dependencies match versions specified in environment.yml
6. IF a required library is missing from environment.yml, THEN THE System SHALL raise an error before execution

### Requirement 11: Error Handling and Diagnostics

**User Story:** As a pipeline user, I want clear error messages and diagnostic information, so that I can troubleshoot issues efficiently.

#### Acceptance Criteria

1. WHEN file loading fails, THE System SHALL report the specific file path and reason for failure
2. WHEN stream identification fails, THE System SHALL list all available stream names in the error message
3. WHEN channel mapping fails, THE System SHALL report which channels have mismatched indices
4. WHEN quality thresholds eliminate all channels, THE System SHALL warn the user and suggest threshold adjustments
5. WHEN marker events are missing, THE System SHALL report expected marker names and what was found
6. THE System SHALL provide progress indicators for long-running operations (filtering, epoching, TFR computation)
