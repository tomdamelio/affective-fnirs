# Implementation Plan: NOTHING (REST) Condition Integration

## Overview

Integrate NOTHING as a consistent third condition across all sub-012 analyses. Changes span the synthesizer timing fix, uncommenting ERD plots, extending fNIRS block average, adding validation, and property-based tests.

## Tasks

- [x] 1. Fix NOTHING epoch synthesis timing and return stats
  - [x] 1.1 Update `synthesize_nothing_annotations()` in `scripts/run_analysis_sub012.py`
    - Change default `rest_duration_cap` parameter from 7.0 to 6.0
    - Add `SynthesisStats` dataclass (or dict) tracking `n_created`, `n_skipped`, `n_source_trials`
    - Return `(raw, synthesis_stats)` tuple instead of just `raw`
    - Add logging of skipped trials count and created count
    - Update both call sites in `main()` (EEG and fNIRS) to unpack the tuple
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 1.2 Write property test: Synthesis output correctness
    - **Property 1: Synthesis output correctness**
    - Use Hypothesis to generate random annotation sets with valid spacing
    - Verify each NOTHING onset = source_onset + task_duration + 1.0
    - Verify each NOTHING duration = rest_duration_cap
    - **Validates: Requirements 1.1, 1.2**

  - [ ]* 1.3 Write property test: Synthesis skip logic
    - **Property 2: Synthesis skip logic**
    - Use Hypothesis to generate annotation sets with mixed inter-trial intervals
    - Verify NOTHING only created for trials with rest >= rest_duration_cap + 1.0
    - Verify n_created + n_skipped = n_source_trials
    - **Validates: Requirements 1.3**

- [x] 2. Enable NOTHING in contralateral ERD/ERS plots
  - [x] 2.1 Uncomment NOTHING plotting in `generate_contralateral_erd_plots()` in `scripts/run_analysis.py`
    - Uncomment the 4 commented-out blocks for C3 alpha, C4 alpha, C3 beta, C4 beta
    - Each block plots `extract_band_power(tfr_nothing, channel, band)` as green dashed line
    - Ensure the `if tfr_nothing is not None:` guard wraps each block
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 3. Add per-condition lines to fNIRS block average
  - [x] 3.1 Modify `generate_fnirs_block_average()` in `scripts/run_analysis.py`
    - Extract condition names from `fnirs_epochs.event_id`
    - For each HbO channel subplot, plot separate mean ± std lines per condition (LEFT=blue, RIGHT=red, NOTHING=green)
    - Handle missing conditions gracefully (skip condition if zero epochs)
    - Add legend to each subplot
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 4. Checkpoint - Verify visualization changes
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Create NOTHING condition validation module
  - [x] 5.1 Create `src/affective_fnirs/validation.py` with `validate_nothing_condition()`
    - Implement `NothingValidationResult` dataclass
    - Implement epoch count parity check (within 10% tolerance)
    - Implement epoch duration check (6.0s ± 0.1s tolerance)
    - Implement EEG/fNIRS presence checks
    - Log warnings for failures, summary for success
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 5.2 Write property test: Validator correctness
    - **Property 3: Validator correctness**
    - Use Hypothesis to generate random (n_left, n_right, n_nothing) triples and duration lists
    - Verify epoch_count_ok matches mathematical condition
    - Verify epoch_duration_ok matches range check
    - **Validates: Requirements 5.1, 5.2**

- [x] 6. Integrate validation into sub-012 pipeline
  - [x] 6.1 Add validation call in `run_analysis_sub012.py` `main()`
    - Import `validate_nothing_condition` from `affective_fnirs.validation`
    - Call after EEG and fNIRS epoch creation
    - Log validation results
    - _Requirements: 5.3, 5.4, 5.5_

- [x] 7. Write regression and integration tests
  - [ ]* 7.1 Write unit tests for existing NOTHING support (no-regression)
    - Test `generate_tfr_maps()` with 3-condition mock epochs doesn't crash
    - Test `generate_erp_analysis()` with 3-condition mock epochs doesn't crash
    - Test `generate_contralateral_erd_plots()` with and without NOTHING
    - Test `generate_fnirs_block_average()` with 3-condition mock epochs
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 7.2 Write property test: MOV/NO_MOV class composition
    - **Property 4: MOV/NO_MOV class composition**
    - Use Hypothesis to generate random epoch counts for LEFT, RIGHT, NOTHING
    - Verify label array has len(LEFT)+len(RIGHT) MOV labels and len(NOTHING) NO_MOV labels
    - **Validates: Requirements 4.1**

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The project uses `hypothesis` (already in `environment.yml`) for property-based testing
- Run tests with: `micromamba run -n affective-fnirs pytest tests/test_nothing_condition.py -v`
- All changes preserve backward compatibility: functions degrade gracefully when NOTHING is absent
