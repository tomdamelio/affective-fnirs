# Implementation Plan: Multimodal Validation Pipeline

## Overview

This implementation plan transforms the design document into actionable coding tasks for building a validation pipeline that processes simultaneous EEG + fNIRS recordings during finger tapping tasks. The pipeline validates signal quality, detects neural patterns (ERD/ERS), hemodynamic responses (HRF), and quantifies neurovascular coupling.

**Key Implementation Principles:**
- Use LSL timestamps from PyXDF for temporal synchronization (never assume fixed offsets)
- Maintain separate MNE Raw objects for EEG and fNIRS (never mix processing)
- Use validated MNE/MNE-NIRS functions instead of implementing from scratch
- Calculate CV on raw intensities, not optical density
- Apply motion correction on OD before Beer-Lambert conversion
- Validate empirically that events precede expected ERD patterns

## Tasks

- [ ] 1. Update environment and configuration infrastructure
  - [ ] 1.1 Update environment.yml with pinned dependencies
    - Add mne, mne-nirs, pyxdf, hypothesis, scipy, pandas, matplotlib
    - Pin versions for reproducibility (e.g., mne==1.6.0, mne-nirs==0.6.0)
    - Include pytest, black, ruff for development
    - _Requirements: 10.5_

  - [ ] 1.2 Replace config.py with comprehensive PipelineConfig
    - Create FilterConfig: eeg_bandpass (1-40 Hz), fnirs_bandpass (0.01-0.5 Hz), cardiac_band (0.5-2.5 Hz)
    - Create QualityThresholds: sci_threshold (0.75-0.80), cv_threshold (10-15%), saturation_percent (5%)
    - Create EpochConfig: eeg_tmin/tmax (-5 to +20s), fnirs_tmin/tmax (-5 to +30s)
    - Create AnalysisConfig: alpha_band (8-13 Hz), beta_band (13-30 Hz), task/baseline windows
    - Create ICAConfig: n_components (0.99 variance), random_state, eog/emg thresholds
    - Implement YAML serialization/deserialization with validation
    - _Requirements: 10.1, 10.4_

- [ ] 2. Implement data ingestion module (ingestion.py)
  - [ ] 2.1 Create XDF file loading with LSL timestamp extraction
    - Implement load_xdf_file() using pyxdf.load_xdf() in read-only mode
    - Extract time_stamps array from each stream (LSL clock domain)
    - Preserve original sampling rates without modification
    - **CRITICAL**: Use LSL timestamps for synchronization, not fixed offsets
    - _Requirements: 1.1, 1.4, 9.1_

  - [ ] 2.2 Implement stream identification by name patterns
    - Implement identify_streams() to detect EEG, fNIRS, Markers by name/type
    - Support common patterns: 'EEG', 'BioSemi', 'fNIRS', 'NIRS', 'Markers', 'Events'
    - Return dictionary with 'eeg', 'fnirs', 'markers' keys
    - Raise descriptive error listing available streams if required stream missing
    - _Requirements: 1.2, 1.3, 11.2_

  - [ ] 2.3 Implement stream data extraction with validation
    - Implement extract_stream_data() returning (data, sfreq, timestamps)
    - Validate data array is non-empty before returning
    - Preserve original data types and shapes
    - _Requirements: 1.4, 1.5_

  - [ ]* 2.4 Write property tests for data ingestion
    - **Property 1: Stream extraction preserves data integrity**
    - **Property 2: Stream validation rejects empty data**
    - **Property 3: Error messages contain diagnostic information**
    - **Validates: Requirements 1.1-1.5, 11.1-11.3**

- [ ] 3. Implement MNE object construction module (mne_builder.py)
  - [ ] 3.1 Create EEG Raw object with proper channel types
    - Implement build_eeg_raw() with channel names from JSON sidecar
    - Set channel types: 'eeg' for standard channels, 'misc' for AUX_1, AUX_2, AUX_3
    - Apply standard_1020 montage (covers 10-20 and 10-10 positions)
    - Validate all EEG channels have 3D positions after montage
    - **NOTE**: Keep EEG Raw separate from fNIRS Raw throughout pipeline
    - _Requirements: 2.1, 2.2_

  - [ ] 3.2 Create fNIRS Raw object with wavelength metadata
    - Implement build_fnirs_raw() with channel types 'fnirs_cw_amplitude'
    - Store wavelength info (760nm, 850nm) in channel metadata loc[9]
    - Load optode positions from JSON montage config
    - Compute and store source-detector distances in loc[10] for short channel detection
    - Validate channel count matches between JSON and data matrix
    - **CRITICAL**: Use proper naming (S1_D1 760, S1_D1 850) for MNE-NIRS compatibility
    - _Requirements: 2.3, 2.4, 2.5_

  - [ ] 3.3 Implement event marker synchronization with LSL timestamps
    - Implement embed_events() converting LSL timestamps to MNE Annotations
    - Calculate onset relative to recording start: onset = marker_time_lsl - raw.times[0]
    - Create mne.Annotations with onset, duration=0, description
    - **VALIDATION**: Verify events precede expected ERD (movement onset before ERD)
    - If temporal drift >1ms, raise detailed error with drift magnitude
    - Map event names to integer codes for epochs extraction
    - _Requirements: 2.6, 11.5_

  - [ ]* 3.4 Write property tests for MNE construction
    - **Property 4: Channel names round-trip through MNE objects**
    - **Property 5: Montage application adds spatial information**
    - **Property 6: Wavelength metadata is preserved**
    - **Property 7: Event timestamps synchronized within 1ms tolerance**
    - **Validates: Requirements 2.1-2.6**

- [ ] 4. Checkpoint - Verify data ingestion and MNE construction
  - Ensure all tests pass, ask the user if questions arise.
  - Verify EEG and fNIRS are in separate Raw objects
  - Confirm LSL timestamps are used for synchronization

- [ ] 5. Implement fNIRS quality assessment module (fnirs_quality.py)
  - [ ] 5.1 Implement Scalp Coupling Index (SCI) calculation
    - Implement calculate_sci() using mne_nirs.preprocessing.scalp_coupling_index()
    - Filter in cardiac band (0.5-2.5 Hz) using FIR filter
    - Compute zero-lag Pearson correlation between 760nm and 850nm per optode pair
    - **THRESHOLD**: SCI > 0.75-0.80 indicates good coupling (Pollonini 2016)
    - Return dict mapping channel pairs to SCI values
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ] 5.2 Implement Coefficient of Variation (CV) on raw intensities
    - Implement calculate_cv() on RAW INTENSITY data (NOT optical density)
    - Extract baseline segments only (e.g., 5s before each task block)
    - Calculate CV = (std / mean) * 100% per channel
    - **THRESHOLD**: CV < 10-15% indicates acceptable stability
    - **CRITICAL**: CV must be calculated BEFORE OD conversion
    - _Requirements: 3.8_

  - [ ] 5.3 Implement saturation and signal loss detection
    - Implement detect_saturation() for ADC overflow (>95% of max range)
    - Implement detect_flat_signal() for contact loss (near-zero variance)
    - Count saturated samples per channel, flag if >5%
    - _Requirements: 3.4, 3.5_

  - [ ] 5.4 Implement cardiac pulsation power assessment
    - Implement assess_cardiac_power() using Welch PSD
    - Identify peak power in 0.5-2.5 Hz band (~1 Hz heart rate)
    - Normalize: PSP = peak_power / total_power_in_band
    - **THRESHOLD**: PSP > 0.1 indicates clear cardiac signal (PHOEBE framework)
    - Absence of cardiac peak suggests poor optode contact
    - _Requirements: 3.6, 3.7_

  - [ ] 5.5 Implement comprehensive bad channel marking
    - Implement mark_bad_channels() evaluating all quality metrics
    - Mark as BAD if: SCI < 0.75 OR saturation > 5% OR PSP < 0.1 OR CV > 15%
    - Record specific failure reason(s) for each bad channel
    - **WARNING**: If ALL long channels marked bad, emit warning with threshold suggestions
    - _Requirements: 3.3, 3.5, 3.7, 11.4_

  - [ ] 5.6 Implement quality visualization
    - Implement generate_quality_heatmap() with spatial channel layout
    - Color-code by SCI value: green (>0.8), yellow (0.6-0.8), red (<0.6)
    - Annotate bad channels with failure reasons
    - Generate quality summary table (TSV) with all metrics per channel
    - _Requirements: 3.9, 3.10_

  - [ ]* 5.7 Write property tests for quality assessment
    - **Property 8: SCI uses cardiac band filtering (0.5-2.5 Hz)**
    - **Property 9: Threshold-based marking is consistent**
    - **Property 10: Quality reports contain all metrics**
    - **Property 11: CV calculated on baseline periods only**
    - **Validates: Requirements 3.1-3.10**

- [ ] 6. Implement fNIRS processing module (fnirs_processing.py)
  - [ ] 6.1 Implement optical density conversion (FIRST STEP)
    - Implement convert_to_optical_density() using mne.preprocessing.nirs.optical_density()
    - **CRITICAL ORDER**: Intensity → OD → Motion Correction → SCR → Beer-Lambert → Filter
    - Verify channel types change to 'fnirs_od'
    - **NOTE**: Never filter raw intensities; CV already calculated in quality step
    - _Requirements: 6.1_

  - [ ] 6.2 Implement motion artifact correction on OD
    - Implement correct_motion_artifacts() with TDDR as default method
    - Use mne.preprocessing.nirs.temporal_derivative_distribution_repair() for TDDR
    - Support wavelet method as alternative (Molavi & Dumont 2012)
    - **CRITICAL**: Apply on OD data, BEFORE Beer-Lambert conversion
    - Log number of corrected artifacts per channel
    - Generate before/after plots for visual verification
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.10_

  - [ ] 6.3 Implement short channel identification
    - Implement identify_short_channels() using source-detector distance
    - Use mne.preprocessing.nirs.short_channels() or compute from montage
    - **THRESHOLD**: < 15mm = short channel, >= 15mm = long channel
    - Return separate lists: short_channels, long_channels
    - _Requirements: 4.5_

  - [ ] 6.4 Implement GLM-based short channel regression
    - Implement apply_short_channel_regression() using mne_nirs.signal_enhancement.short_channel_regression()
    - Pair short/long channels by wavelength AND spatial proximity
    - **CRITICAL**: Apply on OD data, BEFORE Beer-Lambert conversion
    - Log regression statistics (β coefficients, R² per channel)
    - _Requirements: 4.6, 4.7_

  - [ ] 6.5 Verify systemic noise reduction after SCR
    - Implement verify_systemic_noise_reduction() comparing PSD before/after
    - Compute power in systemic band (0.1-0.4 Hz): Mayer waves (~0.1 Hz), respiration (~0.2-0.4 Hz)
    - **EXPECTED**: 20-50% power reduction in long channels
    - If reduction < 10%, log warning about SCR effectiveness
    - _Requirements: 4.8, 4.9_

  - [ ] 6.6 Implement Beer-Lambert hemoglobin conversion
    - Implement convert_to_hemoglobin() using mne.preprocessing.nirs.beer_lambert_law()
    - Use configurable DPF (default 6.0 for adults)
    - **CRITICAL**: Apply AFTER motion correction and SCR (on corrected OD)
    - Verify output has 'hbo' and 'hbr' channel types
    - Channel naming: 'S1_D1 hbo', 'S1_D1 hbr'
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ] 6.7 Implement hemoglobin bandpass filtering
    - Implement filter_hemoglobin_data() with 0.01-0.5 Hz bandpass
    - Use FIR filter for linear phase (no latency distortion)
    - **CRITICAL**: Apply ONLY to HbO/HbR channels, not short channels
    - Preserves hemodynamic response (~0.05-0.2 Hz) while removing drift and cardiac
    - _Requirements: 6.4_

  - [ ] 6.8 Create complete fNIRS processing pipeline function
    - Implement process_fnirs_pipeline() orchestrating all steps in correct order:
      1. Quality assessment (on raw intensity) - already done
      2. Intensity → Optical Density
      3. Motion correction (TDDR on OD)
      4. Short channel regression (on OD)
      5. Verify noise reduction
      6. OD → Hemoglobin (Beer-Lambert)
      7. Bandpass filter (0.01-0.5 Hz on Hb)
    - _Requirements: 4.1-4.10, 6.1-6.4_

  - [ ]* 6.9 Write property tests for fNIRS processing
    - **Property 12: Filter preserves signal length**
    - **Property 13: Short channels identified by distance < 15mm**
    - **Property 14: SCR reduces systemic noise (0.1-0.4 Hz)**
    - **Property 17: Beer-Lambert produces HbO and HbR channels**
    - **Validates: Requirements 4.5-4.8, 6.1-6.4**

- [ ] 7. Checkpoint - Verify fNIRS quality and processing
  - Ensure all tests pass, ask the user if questions arise.
  - Verify processing order: Intensity → OD → TDDR → SCR → Hb → Filter
  - Confirm CV was calculated on raw intensities (not OD)
  - Check systemic noise reduction metrics

- [ ] 8. Implement EEG processing module (eeg_processing.py)
  - [ ] 8.1 Implement EEG bandpass filtering
    - Implement preprocess_eeg() with 1-40 Hz bandpass (or 0.5-45 Hz)
    - Use FIR filter with Hamming window (MNE default) for linear phase
    - **CRITICAL**: Filter EEG separately from fNIRS (different Raw objects)
    - Add progress indicator for long recordings
    - _Requirements: 5.1_

  - [ ] 8.2 Implement bad EEG channel detection
    - Implement detect_bad_eeg_channels() with multiple criteria:
      - Saturation: peak amplitude > 500 μV
      - Excessive noise: RMS > mean + 5*std across channels
      - Flat signal: RMS near zero (disconnected electrode)
    - Mark bad channels in raw.info['bads'] BEFORE ICA
    - _Requirements: 5.7_

  - [ ] 8.3 Implement ICA artifact removal
    - Implement apply_ica_artifact_removal() using mne.preprocessing.ICA
    - Fit ICA on filtered continuous data (1-40 Hz), BEFORE epoching
    - Use n_components=0.99 (99% variance) or n_channels for full decomposition
    - **MINIMUM**: 15 components, but more is better for 32-channel data
    - Set random_state for reproducibility, log seed value
    - Return cleaned Raw and fitted ICA object (save for reuse)
    - _Requirements: 5.2, 5.3, 10.2, 10.3_

  - [ ] 8.4 Implement EOG component identification
    - Implement identify_eog_components() using frontal channel correlation
    - Use ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2']) or manual correlation
    - **THRESHOLD**: |correlation| > 0.8 with frontal channels
    - EOG components show: frontal topography, spike-like time series during blinks
    - Generate topography plots for visual verification
    - _Requirements: 5.4, 5.6_

  - [ ] 8.5 Implement EMG component identification
    - Implement identify_emg_components() using high-frequency power ratio
    - Compute PSD per component, calculate power ratio: (20-40 Hz) / (1-20 Hz)
    - **THRESHOLD**: ratio > 2.0 indicates muscle artifact
    - EMG components show: spiky time series, high power above 20 Hz
    - _Requirements: 5.5, 5.6_

  - [ ] 8.6 Implement bad channel interpolation
    - Implement interpolate_bad_channels() using raw.interpolate_bads(reset_bads=True)
    - **CRITICAL**: Interpolate BEFORE Common Average Reference
    - Requires 10-20 montage with 3D positions
    - If >20% channels bad, warn about interpolation reliability
    - Log interpolated channel names
    - _Requirements: 5.7_

  - [ ] 8.7 Implement Common Average Reference
    - Implement rereference_eeg() using raw.set_eeg_reference('average')
    - **CRITICAL**: Apply AFTER interpolating bad channels
    - MNE automatically excludes channels in info['bads'] from average
    - Document original reference in processing log
    - _Requirements: 5.7_

  - [ ] 8.8 Create complete EEG preprocessing pipeline function
    - Implement preprocess_eeg_pipeline() orchestrating all steps:
      1. Bandpass filter (1-40 Hz)
      2. Detect bad channels
      3. Fit ICA on filtered data
      4. Identify EOG components (frontal correlation)
      5. Identify EMG components (high-freq power)
      6. Apply ICA (exclude artifact components)
      7. Interpolate bad channels
      8. Common Average Reference
    - Save ICA object to derivatives for reproducibility
    - _Requirements: 5.1-5.7_

  - [ ]* 8.9 Write property tests for EEG processing
    - **Property 15: ICA uses at least minimum components**
    - **Property 16: Bad channels excluded/interpolated before CAR**
    - **Validates: Requirements 5.3, 5.7**

- [ ] 9. Implement EEG analysis module (eeg_analysis.py)
  - [ ] 9.1 Implement epoch extraction
    - Implement create_epochs() using mne.Epochs
    - Time window: -5 to +20s around task markers
    - Baseline: -5 to -1s (exclude -1 to 0s to avoid anticipatory ERD)
    - Reject epochs with excessive artifacts (if not caught by ICA)
    - _Requirements: 5.8_

  - [ ] 9.2 Implement time-frequency representation (TFR)
    - Implement compute_tfr() using mne.time_frequency.tfr_morlet()
    - Frequency range: 3-30 Hz with 1 Hz steps
    - Wavelet cycles: 7 (constant) or freqs/2 (frequency-dependent)
    - Apply baseline correction: mode='percent' using -5 to -1s
    - **OUTPUT**: Negative values = ERD (power decrease), Positive = ERS (increase)
    - _Requirements: 5.9, 5.10_

  - [ ] 9.3 Implement ERD/ERS detection with statistical validation
    - Implement detect_erd_ers() for motor cortex channel C3
    - Extract power in mu band (8-13 Hz) and beta band (13-30 Hz)
    - Compare task window (1-14s) vs baseline (-5 to -1s)
    - **EXPECTED PATTERNS** (right-hand tapping):
      - Mu ERD: -20% to -40% during task
      - Beta ERD: -30% to -50% during movement
      - Beta rebound (ERS): +10% to +30% after task (16-20s)
    - Perform paired t-test (task vs baseline) for statistical significance
    - Return metrics: erd_percent, p_value, significant (p<0.05)
    - _Requirements: 5.11, 5.12, 5.13_

  - [ ] 9.4 Implement channel selection with fallback
    - Primary channel: C3 (left motor cortex, contralateral to right hand)
    - Fallback channels: CP3, C1 if C3 is bad/interpolated
    - Verify selected channel not in raw.info['bads']
    - Log channel selection rationale
    - _Requirements: 5.11_

  - [ ] 9.5 Implement EEG visualization functions
    - Implement plot_eeg_spectrogram() with diverging colormap (RdBu_r)
    - Center colormap at 0% change, symmetric vmin/vmax (e.g., -50 to +50%)
    - Annotate: task onset (0s), task offset (15s), frequency bands
    - Implement plot_erd_timecourse() showing alpha and beta power over time
    - Include error bands (±1 SEM across trials)
    - _Requirements: 8.2_

  - [ ]* 9.6 Write property tests for EEG analysis
    - **Property 18: Epoch windows match configuration**
    - **Property 19: Baseline correction subtracts baseline mean**
    - **Property 20: TFR covers specified frequency range**
    - **Property 21: Motor cortex channel C3 selected for right-hand task**
    - **Property 22: ERD compares task to baseline statistically**
    - **Validates: Requirements 5.8-5.13**

- [ ] 10. Checkpoint - Verify EEG processing and analysis
  - Ensure all tests pass, ask the user if questions arise.
  - Verify ICA was fit on filtered continuous data (not epochs)
  - Confirm bad channels interpolated before CAR
  - Check ERD patterns match expected motor cortex activation

- [ ] 11. Implement fNIRS analysis module (fnirs_analysis.py)
  - [ ] 11.1 Implement fNIRS epoch extraction
    - Implement create_fnirs_epochs() with extended window
    - Time window: -5 to +30s (longer than EEG to capture HRF return)
    - Baseline: -5 to 0s (safe because HRF onset delayed ~2s)
    - Include both HbO and HbR channels in same Epochs object
    - _Requirements: 6.5, 6.6_

  - [ ] 11.2 Implement motor ROI channel identification
    - Implement identify_motor_roi_channel() finding channel closest to C3
    - Compute Euclidean distance from fNIRS channel positions to C3 (10-20)
    - **CANDIDATES**: CCP3h-CP3, FCC3h-FC3, CCP1h-CP1 (near motor cortex)
    - Verify selected channel not marked as bad
    - Return HbO channel name (e.g., 'CCP3h-CP3 hbo')
    - _Requirements: 6.7_

  - [ ] 11.3 Implement HRF extraction and averaging
    - Implement extract_hrf() using epochs.average()
    - Extract HbO and HbR time series for motor ROI channel
    - **EXPECTED PATTERN**:
      - HbO: rises ~2s post-stimulus, plateaus 5-15s, returns ~20-30s
      - HbR: inverse pattern (decreases during task)
    - Return times, hrf_hbo, hrf_hbr arrays
    - _Requirements: 6.8_

  - [ ] 11.4 Implement HRF temporal dynamics validation
    - Implement validate_hrf_temporal_dynamics() with statistical testing
    - **Onset detection**: Find first time HbO > threshold (~0.1 μM) after 0s
      - Expected: 2-3s post-stimulus (neurovascular delay)
    - **Time-to-peak**: Find maximum HbO in 0-15s window
      - Expected: 4-8s for brief stimuli, may plateau for sustained tasks
    - **Plateau amplitude**: Mean HbO during 5-15s vs baseline
      - Paired t-test for significance (p < 0.05)
    - **Trial consistency**: Correlation between individual trials and grand average
      - r > 0.7 indicates highly consistent response
    - _Requirements: 6.9, 6.10, 6.11_

  - [ ] 11.5 Implement HRF quality metrics
    - Implement compute_hrf_quality_metrics() for reliability assessment
    - **Trial-to-trial consistency**: Mean pairwise correlation (Fisher z-transform)
    - **SNR**: (mean_plateau - mean_baseline) / std_baseline
      - SNR > 2 indicates good response, SNR < 1 indicates poor
    - Optional: Fit canonical double-gamma HRF model, compute R²
    - _Requirements: 6.12_

  - [ ] 11.6 Implement HRF visualization functions
    - Implement plot_hrf_curves() with HbO (red) and HbR (blue) traces
    - Add ±1 SD shading across trials for variability
    - Annotate: task window (0-15s), onset time, peak time
    - Option to overlay individual trial curves (semi-transparent)
    - Implement plot_hrf_spatial_map() showing activation across all channels
    - _Requirements: 8.3, 6.12_

  - [ ]* 11.7 Write property tests for fNIRS analysis
    - **Property 23: HRF onset detected within 2-3s window**
    - **Property 24: Time-to-peak within 4-8s range**
    - **Property 25: Plateau amplitude significantly positive**
    - **Validates: Requirements 6.9-6.11**

- [ ] 12. Implement multimodal analysis module (multimodal_analysis.py)
  - [ ] 12.1 Implement EEG alpha envelope extraction
    - Implement extract_eeg_envelope() for neurovascular coupling
    - Bandpass filter EEG in alpha band (8-12 Hz)
    - Apply Hilbert transform to get analytic signal
    - Extract envelope (magnitude of analytic signal)
    - Low-pass filter envelope (<0.5 Hz) to match hemodynamic frequency
    - _Requirements: 7.1, 7.2_

  - [ ] 12.2 Implement signal resampling for cross-correlation
    - Resample EEG envelope to match fNIRS sampling rate
    - Use scipy.signal.resample or mne.filter.resample
    - Verify time alignment after resampling
    - _Requirements: 7.2_

  - [ ] 12.3 Implement neurovascular coupling computation
    - Implement compute_neurovascular_coupling() with cross-correlation
    - **INVERT** alpha envelope (alpha decreases during activation → ERD)
    - Compute cross-correlation with HbO time series
    - Find lag with maximum correlation
    - **EXPECTED**: Negative lag (EEG precedes HbO by ~2-5s)
    - **INTERPRETATION**: Negative correlation between ERD and HbO increase
    - Return: max_correlation, lag_seconds, lag_negative (bool)
    - _Requirements: 7.3, 7.4, 7.5_

  - [ ] 12.4 Implement coupling visualization
    - Implement plot_coupling_overlay() showing EEG and fNIRS time series
    - Plot inverted alpha envelope and HbO on same axes (dual y-axis)
    - Annotate optimal lag and correlation strength
    - Show cross-correlation function with peak marked
    - _Requirements: 7.6, 8.4_

  - [ ]* 12.5 Write property tests for multimodal analysis
    - **Property 26: Envelope extraction uses Hilbert transform**
    - **Property 27: Resampling matches fNIRS sampling rate**
    - **Property 28: Cross-correlation identifies maximum lag**
    - **Property 29: Coupling lag is negative (EEG precedes fNIRS)**
    - **Validates: Requirements 7.1-7.5**

- [ ] 13. Implement reporting module (reporting.py)
  - [ ] 13.1 Implement quality report generation (BIDS-compliant)
    - Implement generate_quality_report() with TSV and JSON outputs
    - **TSV file**: sub-{sub}_ses-{ses}_task-{task}_desc-quality_channels.tsv
      - Columns: channel_name, sci, saturation_percent, cardiac_power, cv, is_bad, reason
    - **JSON data dictionary**: Describe each column (description, units, range, threshold)
    - **PNG heatmap**: Spatial quality visualization
    - _Requirements: 3.9, 8.1, 9.2, 9.3, 9.4_

  - [ ] 13.2 Implement HTML validation report
    - Implement generate_validation_report_html() using mne.Report()
    - **Sections**:
      1. Header: subject, session, task, timestamp, software versions
      2. Quality Assessment: tables, heatmaps
      3. EEG Analysis: spectrograms, ERD metrics
      4. fNIRS Analysis: HRF curves, temporal validation
      5. Multimodal Coupling: overlay plots, correlation metrics
      6. Validation Summary: pass/fail criteria, conclusions
      7. Configuration: pipeline parameters used
    - _Requirements: 8.1-8.6_

  - [ ] 13.3 Implement validation conclusions generator
    - Implement generate_validation_conclusions() with pass/fail evaluation
    - **EEG Criteria**:
      - Alpha ERD significant? (p < 0.05, magnitude > 20%)
      - Beta ERD significant? (p < 0.05, magnitude > 30%)
      - Beta rebound observed? (positive change post-task)
    - **fNIRS Criteria**:
      - HbO onset detected? (within 2-3s)
      - Time-to-peak plausible? (within 4-8s)
      - Plateau amplitude significant? (p < 0.05)
      - Trial consistency acceptable? (r > 0.5)
    - **Coupling Criteria**:
      - EEG precedes fNIRS? (negative lag)
      - Correlation strength adequate? (r > 0.4)
    - Provide diagnostic suggestions for failures
    - _Requirements: 8.6_

  - [ ] 13.4 Implement numerical results saving
    - Implement save_numerical_results() to JSON
    - Include: all metrics, software versions, random seed, config parameters
    - Enable reproducibility verification
    - Filename: sub-{sub}_ses-{ses}_task-{task}_desc-validation_metrics.json
    - _Requirements: 10.2, 10.3_

  - [ ]* 13.5 Write property tests for reporting
    - **Property 32: Tabular outputs have JSON data dictionaries**
    - **Property 37: Report contains all required sections**
    - **Property 38: Report output is HTML format**
    - **Validates: Requirements 8.1-8.6, 9.4**

- [ ] 14. Checkpoint - Verify reporting functionality
  - Ensure all tests pass, ask the user if questions arise.
  - Verify BIDS-compliant filenames and paths
  - Check HTML report contains all sections
  - Confirm JSON data dictionaries accompany TSV files

- [ ] 15. Implement BIDS compliance utilities (bids_utils.py)
  - [ ] 15.1 Implement BIDS path validation
    - Implement validate_bids_path() with regex for entity ordering
    - Pattern: sub-XX_ses-XX_task-XX (correct order)
    - Provide specific guidance for incorrect paths
    - _Requirements: 9.5, 9.6_

  - [ ] 15.2 Implement derivative output path generation
    - Implement generate_derivative_path() following BIDS structure
    - Output to: data/derivatives/validation-pipeline/sub-{sub}/
    - Use BIDS naming: key-value pairs separated by underscores
    - _Requirements: 9.2, 9.3_

  - [ ] 15.3 Implement read-only enforcement for raw data
    - Verify all file operations on data/raw/ use read-only mode
    - Raise error if write/append mode attempted on raw data
    - _Requirements: 9.1_

  - [ ]* 15.4 Write property tests for BIDS compliance
    - **Property 30: Raw data accessed in read-only mode**
    - **Property 31: Derivative outputs follow BIDS structure**
    - **Property 33: BIDS path validation catches incorrect ordering**
    - **Validates: Requirements 9.1-9.6**

- [ ] 16. Implement main pipeline orchestration (pipeline.py)
  - [ ] 16.1 Create pipeline runner function
    - Implement run_validation_pipeline() orchestrating all stages:
      1. Load XDF and JSON metadata
      2. Build MNE objects (EEG and fNIRS separately)
      3. fNIRS quality assessment (on raw intensity)
      4. fNIRS processing (OD → TDDR → SCR → Hb → Filter)
      5. EEG preprocessing (Filter → ICA → Interpolate → CAR)
      6. EEG analysis (Epochs → TFR → ERD/ERS)
      7. fNIRS analysis (Epochs → HRF → Validation)
      8. Multimodal coupling (Envelope → Cross-correlation)
      9. Generate reports (Quality, HTML, JSON)
    - Handle errors gracefully with informative messages
    - Log progress for long-running operations (Req. 11.6)
    - _Requirements: All_

  - [ ] 16.2 Create command-line interface
    - Implement CLI with argparse
    - Arguments: --xdf-file, --eeg-json, --fnirs-json, --config, --output
    - Validate input paths before execution
    - Support both CLI and library usage
    - _Requirements: 10.1_

  - [ ]* 16.3 Write property tests for reproducibility
    - **Property 34: Configuration round-trip (YAML save/load)**
    - **Property 35: Random seed produces deterministic results**
    - **Property 36: Dependency versions validated against environment.yml**
    - **Validates: Requirements 10.1-10.5**

- [ ] 17. Integration testing and validation
  - [ ] 17.1 Test complete pipeline on pilot data
    - Run pipeline on sub-001 and sub-002 data
    - Verify all outputs generated in data/derivatives/validation-pipeline/
    - Check BIDS compliance of all output filenames
    - Validate HTML report completeness and correctness
    - _Requirements: All_

  - [ ] 17.2 Verify reproducibility
    - Run pipeline twice with same random seed
    - Verify identical numerical results (JSON metrics)
    - Check configuration round-trip (save → load → compare)
    - _Requirements: 10.2, 10.3, 10.4_

  - [ ] 17.3 Generate intermediate visualization plots
    - Create control plots for quality verification:
      - EEG spectrograms before/after ICA
      - fNIRS signals before/after motion correction
      - fNIRS signals before/after short channel regression
      - HRF individual trials overlaid with grand average
    - Use for detecting anomalies early
    - _Requirements: 5.6, 6.12_

  - [ ]* 17.4 Run full property-based test suite
    - Execute all 38 property tests with 100+ iterations each
    - Verify all properties pass
    - Document any edge cases discovered
    - _Requirements: All_

- [ ] 18. Final checkpoint - Complete pipeline validation
  - Ensure all tests pass, ask the user if questions arise.
  - Verify HTML reports generated for both subjects
  - Confirm all validation criteria evaluated correctly
  - Review conclusions section for scientific accuracy

## Notes

- Tasks marked with `*` are optional property-based tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation throughout implementation
- Property tests validate universal correctness properties across randomized inputs

## Critical Implementation Reminders

**Data Separation:**
- Keep EEG and fNIRS in separate MNE Raw objects throughout
- Never apply ICA to fNIRS data
- Never mix EEG and fNIRS before analysis

**Processing Order (fNIRS):**
1. Quality assessment on RAW INTENSITY (CV calculation)
2. Intensity → Optical Density
3. Motion correction (TDDR) on OD
4. Short channel regression on OD
5. OD → Hemoglobin (Beer-Lambert)
6. Bandpass filter on Hb

**Processing Order (EEG):**
1. Bandpass filter (1-40 Hz)
2. Detect bad channels
3. Fit ICA on filtered continuous data
4. Identify and exclude artifact components
5. Interpolate bad channels
6. Common Average Reference

**Temporal Synchronization:**
- Use LSL timestamps from PyXDF (never assume fixed offsets)
- Validate events precede expected ERD patterns
- If drift > 1ms, raise error with diagnostic info

**Quality Thresholds:**
- SCI > 0.75-0.80 (good optode coupling)
- CV < 10-15% (stable baseline)
- PSP > 0.1 (clear cardiac signal)
- Saturation < 5% (no ADC overflow)

**Expected Patterns:**
- Mu ERD: -20% to -40% during task (C3)
- Beta ERD: -30% to -50% during movement
- Beta rebound: +10% to +30% post-task
- HbO onset: 2-3s post-stimulus
- HbO peak/plateau: 4-8s post-stimulus
- Neurovascular lag: negative (EEG precedes fNIRS)
