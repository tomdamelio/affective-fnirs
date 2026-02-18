# Requirements Document

## Introduction

This feature integrates the NOTHING (REST) condition as a full third condition across all EEG and fNIRS analyses for sub-012 in the affective-fnirs pipeline. Currently, NOTHING annotations are synthesized from post-trial rest periods, but support is inconsistent: some visualizations include NOTHING (TFR maps, ERP), others have it commented out (contralateral ERD), and the fNIRS block average does not separate by condition. Additionally, the epoch timing parameters in `synthesize_nothing_annotations()` need verification against the protocol. This feature ensures NOTHING is a first-class condition everywhere, enables MOV vs NO MOV classification, and adds validation checks.

## Glossary

- **NOTHING_Condition**: A synthesized experimental condition representing the post-trial REST period, where no motor task is performed. Used as a neural baseline control.
- **NOTHING_Epoch**: A 6-second data segment extracted from the REST period, starting 1 second after LEFT/RIGHT stimulus offset. The 1-second gap serves as baseline for the NOTHING epoch itself.
- **MOV_Class**: A binary classification label combining LEFT and RIGHT conditions into a single "movement" class for predictive modeling.
- **NO_MOV_Class**: A binary classification label corresponding to the NOTHING condition, representing absence of motor execution.
- **Synthesizer**: The `synthesize_nothing_annotations()` function in `scripts/run_analysis_sub012.py` that creates NOTHING annotations from existing LEFT/RIGHT trial timing.
- **Pipeline**: The complete analysis workflow orchestrated by `scripts/run_analysis.py` and `scripts/run_analysis_sub012.py`.
- **ERD_ERS_Plot**: Contralateral Event-Related Desynchronization/Synchronization timecourse visualization in `generate_contralateral_erd_plots()`.
- **Block_Average_Plot**: The fNIRS block-averaged HRF visualization in `generate_fnirs_block_average()`.
- **Condition_Validator**: A validation module that checks NOTHING epoch counts, timing, and presence across all analysis outputs.

## Requirements

### Requirement 1: NOTHING Epoch Synthesis Timing

**User Story:** As a neuroscience researcher, I want NOTHING epochs to have correct timing parameters matching the protocol, so that the REST condition data is scientifically valid.

#### Acceptance Criteria

1. WHEN the Synthesizer creates NOTHING annotations, THE Synthesizer SHALL set the virtual onset at exactly 1.0 second after the LEFT/RIGHT stimulus offset (task_end + 1.0s).
2. WHEN the Synthesizer creates NOTHING annotations, THE Synthesizer SHALL produce epochs of 6.0 seconds duration (from 1s to 7s after stimulus offset), matching LEFT and RIGHT epoch duration.
3. WHEN the available rest period between trials is shorter than 7.0 seconds (1s baseline gap + 6s epoch), THE Synthesizer SHALL skip that trial and not create a NOTHING annotation.
4. WHEN the Synthesizer completes annotation synthesis, THE Synthesizer SHALL log the count of NOTHING annotations created and the count of trials skipped due to insufficient rest.
5. THE Synthesizer SHALL use `task_duration=7.0` and `rest_duration_cap=6.0` as parameters, consistent with the 6-second epoch window defined in `configs/sub-012.yml`.

### Requirement 2: NOTHING in EEG Contralateral ERD/ERS Plots

**User Story:** As a neuroscience researcher, I want NOTHING displayed alongside LEFT and RIGHT in contralateral ERD/ERS timecourse plots, so that I can visually compare motor desynchronization against the resting baseline.

#### Acceptance Criteria

1. WHEN the ERD_ERS_Plot generates alpha-band timecourses for C3 and C4, THE ERD_ERS_Plot SHALL include the NOTHING condition as a dashed green line alongside LEFT and RIGHT.
2. WHEN the ERD_ERS_Plot generates beta-band timecourses for C3 and C4, THE ERD_ERS_Plot SHALL include the NOTHING condition as a dashed green line alongside LEFT and RIGHT.
3. WHEN NOTHING epochs are not available in the input data, THE ERD_ERS_Plot SHALL generate plots with only LEFT and RIGHT conditions without error.

### Requirement 3: NOTHING in fNIRS Block Average

**User Story:** As a neuroscience researcher, I want the fNIRS block average plot to show separate HRF curves per condition (LEFT, RIGHT, NOTHING), so that I can compare hemodynamic responses across all conditions per channel.

#### Acceptance Criteria

1. WHEN the Block_Average_Plot generates per-channel HRF curves, THE Block_Average_Plot SHALL display separate colored lines for LEFT (blue), RIGHT (red), and NOTHING (green) conditions.
2. WHEN the Block_Average_Plot generates per-channel HRF curves, THE Block_Average_Plot SHALL include standard deviation shading for each condition.
3. WHEN a condition has zero epochs for a given channel, THE Block_Average_Plot SHALL omit that condition line for that channel without error.

### Requirement 4: MOV vs NO MOV Predictive Model

**User Story:** As a neuroscience researcher, I want a CSP-based classifier comparing MOV (LEFT + RIGHT combined) vs NO_MOV (NOTHING), so that I can quantify the discriminability of motor execution versus rest.

#### Acceptance Criteria

1. WHEN the Pipeline runs CSP analysis for MOV vs NO_MOV, THE Pipeline SHALL combine LEFT and RIGHT epochs into the MOV_Class and use NOTHING epochs as the NO_MOV_Class.
2. WHEN the Pipeline reports classification results, THE Pipeline SHALL report balanced accuracy with standard deviation from stratified k-fold cross-validation.
3. WHEN the class imbalance ratio between MOV_Class and NO_MOV_Class exceeds 2:1, THE Pipeline SHALL log a warning about class imbalance.
4. IF NOTHING epochs are absent from the input data, THEN THE Pipeline SHALL skip MOV vs NO_MOV analysis and log a descriptive warning.

### Requirement 5: NOTHING Condition Validation

**User Story:** As a neuroscience researcher, I want automated validation checks confirming NOTHING is correctly generated and included in all analyses, so that I can trust the integrity of the three-condition comparison.

#### Acceptance Criteria

1. WHEN the Condition_Validator runs after epoch synthesis, THE Condition_Validator SHALL verify that the count of NOTHING epochs is within 10% of the count of LEFT epochs and within 10% of the count of RIGHT epochs.
2. WHEN the Condition_Validator runs after epoch synthesis, THE Condition_Validator SHALL verify that all NOTHING epoch durations equal 6.0 seconds (within 0.1s tolerance).
3. WHEN the Condition_Validator runs after analysis completion, THE Condition_Validator SHALL verify that NOTHING appears as a condition key in both EEG and fNIRS epoch objects.
4. WHEN any validation check fails, THE Condition_Validator SHALL log a warning with the specific check name, expected value, and actual value.
5. WHEN all validation checks pass, THE Condition_Validator SHALL log a summary confirming NOTHING condition integrity.

### Requirement 6: Consistent Three-Condition Support Across All Analyses

**User Story:** As a neuroscience researcher, I want all existing analysis functions to consistently handle three conditions (LEFT, RIGHT, NOTHING), so that no analysis silently drops the NOTHING condition.

#### Acceptance Criteria

1. WHEN `generate_tfr_maps()` receives epochs with NOTHING, THE Pipeline SHALL render NOTHING as a third column in the TFR grid (already implemented, verify no regression).
2. WHEN `generate_erp_analysis()` receives epochs with NOTHING, THE Pipeline SHALL render NOTHING as a green dashed line with SEM shading (already implemented, verify no regression).
3. WHEN `generate_contrast_analysis()` receives epochs with NOTHING, THE Pipeline SHALL include the Motor Execution contrast row: (LEFT+RIGHT)/2 vs NOTHING (already implemented, verify no regression).
4. WHEN `generate_fnirs_hrf_by_condition()` receives epochs with NOTHING, THE Pipeline SHALL render NOTHING HRF curves in green (already implemented, verify no regression).
5. WHEN `generate_fnirs_contrast_map()` receives epochs with NOTHING, THE Pipeline SHALL include the Motor vs Rest bar comparison (already implemented, verify no regression).
6. WHEN any analysis function receives epochs without NOTHING, THE Pipeline SHALL degrade gracefully to two-condition (LEFT vs RIGHT) mode without error.
