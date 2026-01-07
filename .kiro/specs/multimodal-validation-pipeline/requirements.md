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
8. WHEN ERD is detected, THE System SHALL verify power d