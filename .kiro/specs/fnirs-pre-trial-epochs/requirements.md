# Requirements Document

## Introduction

This feature modifies fNIRS epoch extraction for block average and contrast analysis in the sub-012 analysis pipeline. Instead of extracting epochs from stimulus onset with fixed duration windows, the system will extract epochs capturing the "stable state" just BEFORE the next trial marker. This approach provides cleaner hemodynamic baselines by capturing the return-to-baseline period after motor execution (MOV) or true resting state (NO MOV).

## Glossary

- **Epoch_Extractor**: The component responsible for extracting time-windowed fNIRS data segments relative to trial markers.
- **Pre_Trial_Window**: A 3-second time window immediately preceding the next trial marker onset.
- **MOV_Condition**: Motor execution trials (LEFT or RIGHT hand finger tapping).
- **NO_MOV_Condition**: Rest/NOTHING trials where no motor task is performed.
- **Trial_Marker**: Annotation event indicating trial onset (LEFT, RIGHT, or NOTHING).
- **Hemodynamic_Baseline**: The stable fNIRS signal state representing return-to-baseline after hemodynamic response.
- **Block_Average_Visualizer**: The component that generates averaged HRF plots comparing MOV vs NO MOV conditions.
- **Contrast_Analyzer**: The component that computes hemispheric lateralization and motor vs rest contrasts.

## Requirements

### Requirement 1: Pre-Trial Epoch Extraction for MOV Condition

**User Story:** As a neuroscience researcher, I want to extract fNIRS epochs from the last 3 seconds before the next trial marker for MOV (LEFT/RIGHT) conditions, so that I can capture the stable hemodynamic return-to-baseline state after motor execution.

#### Acceptance Criteria

1. WHEN extracting epochs for MOV (LEFT or RIGHT) trials, THE Epoch_Extractor SHALL identify the onset time of the next trial marker.
2. WHEN the next trial marker is identified, THE Epoch_Extractor SHALL extract a 3-second window ending at the next marker onset (equivalent to -3000ms to 0ms relative to next marker).
3. WHEN a MOV trial has 8 seconds of rest before the next marker, THE Epoch_Extractor SHALL extract data from 5000ms to 8000ms post-current-marker onset.
4. IF a MOV trial is the last trial in the recording, THEN THE Epoch_Extractor SHALL skip that trial or use available data up to recording end.

### Requirement 2: Pre-Trial Epoch Extraction for NO MOV Condition

**User Story:** As a neuroscience researcher, I want to extract fNIRS epochs from the last 3 seconds before the next LEFT/RIGHT marker for NO MOV (NOTHING) conditions, so that I can capture the true resting state just before motor task onset.

#### Acceptance Criteria

1. WHEN extracting epochs for NO MOV (NOTHING) trials, THE Epoch_Extractor SHALL find the next LEFT or RIGHT trial marker (not another NOTHING marker).
2. WHEN the next motor marker is found, THE Epoch_Extractor SHALL extract a 3-second window ending at that marker onset.
3. WHEN a NOTHING trial has variable rest duration (8-16 seconds), THE Epoch_Extractor SHALL correctly locate the next motor marker regardless of rest duration.
4. IF no subsequent motor marker exists after a NOTHING trial, THEN THE Epoch_Extractor SHALL skip that trial.

### Requirement 3: Epoch Data Structure

**User Story:** As a developer, I want the pre-trial epochs to be returned in a standard MNE Epochs format, so that existing visualization and analysis functions can consume them without modification.

#### Acceptance Criteria

1. THE Epoch_Extractor SHALL return pre-trial epochs as an MNE Epochs object with shape (n_epochs, n_channels, n_times).
2. THE Epoch_Extractor SHALL preserve channel metadata (HbO/HbR types, channel names) from the source Raw object.
3. THE Epoch_Extractor SHALL set the epoch time axis to represent the 3-second window (e.g., 0 to 3 seconds or -3 to 0 seconds).
4. THE Epoch_Extractor SHALL include event metadata indicating the original trial condition (LEFT, RIGHT, or NOTHING).

### Requirement 4: Block Average Visualization Update

**User Story:** As a neuroscience researcher, I want the block average visualization to use pre-trial epochs, so that I can compare stable hemodynamic states between MOV and NO MOV conditions.

#### Acceptance Criteria

1. WHEN generating block average plots, THE Block_Average_Visualizer SHALL use pre-trial epochs instead of stimulus-onset epochs.
2. THE Block_Average_Visualizer SHALL compute mean and SEM across pre-trial epochs for each condition.
3. THE Block_Average_Visualizer SHALL display MOV (contralateral) and NO MOV traces with appropriate labels indicating pre-trial extraction.
4. THE Block_Average_Visualizer SHALL maintain the existing channel grid layout and color scheme.

### Requirement 5: Contrast Analysis Update

**User Story:** As a neuroscience researcher, I want the contrast analysis to use pre-trial epochs, so that I can compute hemispheric lateralization from stable baseline states.

#### Acceptance Criteria

1. WHEN computing contrast maps, THE Contrast_Analyzer SHALL use pre-trial epochs for amplitude calculations.
2. THE Contrast_Analyzer SHALL compute mean amplitude across the 3-second pre-trial window for each condition and hemisphere.
3. THE Contrast_Analyzer SHALL maintain the existing lateralization index formula: (R-L)/(|R|+|L|).
4. THE Contrast_Analyzer SHALL preserve the 2x3 figure layout (HbO/HbR rows, amplitude/lateralization/contrast columns).

### Requirement 6: Backward Compatibility

**User Story:** As a developer, I want to maintain backward compatibility with existing epoch extraction, so that other analyses (TFR, ERP, HRF validation) continue to work unchanged.

#### Acceptance Criteria

1. THE Epoch_Extractor SHALL provide a new function for pre-trial extraction without modifying the existing `create_fnirs_epochs()` function.
2. WHEN the existing `create_fnirs_epochs()` function is called, THE Epoch_Extractor SHALL continue to extract stimulus-onset epochs with the original tmin/tmax parameters.
3. THE Epoch_Extractor SHALL allow selection between stimulus-onset and pre-trial extraction modes via a parameter or separate function.

### Requirement 7: Edge Case Handling

**User Story:** As a developer, I want robust handling of edge cases, so that the epoch extraction does not fail on incomplete or unusual trial sequences.

#### Acceptance Criteria

1. IF a trial has insufficient data before the next marker (less than 3 seconds), THEN THE Epoch_Extractor SHALL skip that trial and log a warning.
2. IF the marker stream contains unexpected marker types, THEN THE Epoch_Extractor SHALL ignore non-trial markers and continue processing.
3. IF the recording ends before a complete pre-trial window can be extracted, THEN THE Epoch_Extractor SHALL skip the affected trial.
4. THE Epoch_Extractor SHALL log the number of trials extracted and skipped for each condition.
