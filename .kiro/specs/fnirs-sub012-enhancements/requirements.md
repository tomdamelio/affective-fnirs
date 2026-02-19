# Requirements Document

## Introduction

This feature enhances the fNIRS processing, quality assessment, and visualization pipeline for subject 012 in the affective-fnirs project. The eight modifications target `scripts/run_analysis_sub012.py` and supporting modules to: (1) add `--fnirs-only` and `--eeg-only` CLI flags for single-modality runs, (2) load a new extended montage with 24 fNIRS source-detector pairs (48 wavelength columns at 760nm/850nm, plus 6 AUX columns = 54 raw XDF columns), (3) correct the total channel count reported by QA, (4) verify the existing SCI comparison plot works with the new montage, (5) filter bad channels using a lowered SCI threshold of 0.50, (6) add a full-experiment HbO/HbR time series visualization, (7) add a 4-ROI HRF-by-condition plot with anatomically defined regions, and (8) verify and enforce ROI-specific short channel regression so that each cluster of long channels is corrected using its spatially nearest short channel.

**Montage summary (from sub-012 recording session):**
- 10 long-channel sources (S1–S10) + 4 short-channel sources (S13, S14, S15, S16)
- 10 detectors (D1–D10)
- 24 source-detector pairs total (20 long + 4 short), each measured at 760nm and 850nm → 48 wavelength-specific data columns in the XDF stream, plus 6 AUX columns → 54 raw columns total.

**Short channel to ROI mapping (from montage spatial layout):**
- S13→D1 (Left Anterior): regresses long channels from S1, S9 cluster
- S14→D3 (Right Anterior): regresses long channels from S2, S10 cluster
- S15→D6 (Left Posterior): regresses long channels from S3, S4, S5 cluster
- S16→D9 (Right Posterior): regresses long channels from S6, S7, S8 cluster

## Glossary

- **Analysis_Script**: The `scripts/run_analysis_sub012.py` module that orchestrates the sub-012 fNIRS+EEG pipeline.
- **Modality_Flag**: A mutually exclusive CLI argument (`--fnirs-only`, `--eeg-only`) that restricts the pipeline to a single modality.
- **Montage_Loader**: The component responsible for loading the new JSON montage file (`montage_combined_EEG_fNIRS_with_3Dcoords_approx.json`) and passing its `ChMontage` to `build_fnirs_raw()`.
- **Channel_Counter**: The component that reports the total fNIRS channels in the quality assessment summary, filtering out AUX/misc channels.
- **SCI_Comparator**: The `generate_sci_comparison_plot()` function that produces a grouped barplot comparing SCI values between the first and second halves of the recording.
- **Channel_Filter**: The component that identifies good channels (SCI > threshold) and propagates a good-channels list to all downstream fNIRS visualization functions.
- **Timeseries_Plotter**: The `generate_fnirs_timeseries_plot()` function that renders full-experiment HbO and HbR concentration traces with stimulus markers.
- **ROI_HRF_Plotter**: The `generate_fnirs_hrf_by_condition_4roi()` function that renders HRF waveforms for 4 anatomical ROIs across 3 conditions.
- **SCR_Verifier**: The component that verifies short channel regression is correctly applied, with each short channel regressing its spatially corresponding cluster of long channels.
- **SCI**: Scalp Coupling Index, a 0–1 metric of optode-scalp contact quality (Pollonini et al., 2016).
- **ROI**: Region of Interest, a group of fNIRS channels sharing an anatomical area.
- **HbO**: Oxyhemoglobin concentration (μM), derived via modified Beer-Lambert law.
- **HbR**: Deoxyhemoglobin concentration (μM), derived via modified Beer-Lambert law.
- **Good_Channel**: A source-detector pair whose SCI exceeds the configured threshold.
- **Bad_Channel**: A source-detector pair whose SCI falls at or below the configured threshold.
- **Source_Detector_Pair**: A unique combination of one source and one detector, measured at two wavelengths (760nm, 850nm). The montage defines 24 such pairs.
- **Long_Channel**: A source-detector pair with ~30mm separation, measuring cortical hemodynamics.
- **Short_Channel**: A source-detector pair with ~8–10mm separation (S13, S14, S15, S16), measuring superficial/systemic signals for regression.
- **SCR**: Short Channel Regression — a GLM-based technique that removes superficial physiological noise (scalp blood flow, Mayer waves, respiration artifacts) from long channels by regressing out the signal captured by nearby short channels (Saager & Berger, 2005).

## Requirements

### Requirement 1: Single-Modality CLI Flags

**User Story:** As a researcher, I want `--fnirs-only` and `--eeg-only` flags on the sub-012 script, so that I can run only the modality I am interested in without waiting for the full multimodal pipeline.

#### Acceptance Criteria

1. THE Analysis_Script SHALL accept an optional `--fnirs-only` flag that, when provided, restricts the pipeline to fNIRS data loading, preprocessing, quality assessment, and visualization only.
2. THE Analysis_Script SHALL accept an optional `--eeg-only` flag that, when provided, restricts the pipeline to EEG data loading, preprocessing, analysis, and visualization only.
3. WHEN `--fnirs-only` is provided, THE Analysis_Script SHALL skip all EEG stream loading, EEG preprocessing, EEG analysis (TFR, ERD/ERS, ERP, CSP), and EEG visualization steps.
4. WHEN `--eeg-only` is provided, THE Analysis_Script SHALL skip all fNIRS stream loading, fNIRS preprocessing, fNIRS quality assessment (SCI, saturation), and fNIRS visualization steps.
5. WHEN either `--fnirs-only` or `--eeg-only` is provided, THE Analysis_Script SHALL skip multimodal analysis (`run_multimodal_analysis` / neurovascular coupling), since cross-modal correlation requires both modalities.
6. THE Analysis_Script SHALL treat `--fnirs-only` and `--eeg-only` as mutually exclusive: if both are provided simultaneously, THE Analysis_Script SHALL exit with an error message indicating the flags are incompatible.
7. WHEN neither `--fnirs-only` nor `--eeg-only` is provided, THE Analysis_Script SHALL run the full multimodal pipeline (both EEG and fNIRS) as it does currently, including multimodal analysis if both streams are available.
8. THE Analysis_Script SHALL log which modality mode is active (EEG-only, fNIRS-only, or full multimodal) at the start of execution.

### Requirement 2: Load New Extended Montage for Sub-012

**User Story:** As a researcher, I want the sub-012 pipeline to load the new 24-pair montage JSON directly, so that channel definitions match the actual optode placement used in this recording session.

#### Acceptance Criteria

1. WHEN the Analysis_Script starts for sub-012, THE Montage_Loader SHALL load the montage file `montage_combined_EEG_fNIRS_with_3Dcoords_approx.json` from the subject's BIDS session directory (`data/raw/sub-012/ses-001/`).
2. WHEN the montage file is loaded, THE Montage_Loader SHALL extract the `ChMontage` key and pass it to `build_fnirs_raw()` as the `montage_config` parameter.
3. IF the montage file does not exist at the expected path, THEN THE Montage_Loader SHALL log an error message containing the attempted file path and terminate the pipeline.
4. WHEN `build_fnirs_raw()` receives the new montage config, THE Montage_Loader SHALL produce an MNE Raw object where all 48 wavelength channels (24 pairs × 2 wavelengths) are typed as `fnirs_cw_amplitude` and any extra XDF columns (6 AUX) are typed as `misc`.

### Requirement 3: Correct Total fNIRS Channel Count in QA Report

**User Story:** As a researcher, I want the quality assessment report to show the correct number of fNIRS channels (48 wavelength channels from 24 source-detector pairs), so that the report accurately reflects the montage rather than including AUX columns from the XDF stream.

#### Acceptance Criteria

1. WHEN the Channel_Counter reports total fNIRS channels, THE Channel_Counter SHALL count only channels of type `fnirs_cw_amplitude`, excluding channels of type `misc`.
2. WHEN the new 24-pair montage is used with an XDF stream containing 54 raw columns, THE Channel_Counter SHALL report 48 as the total fNIRS wavelength channel count (24 pairs × 2 wavelengths), not 54.

### Requirement 4: Verify SCI Comparison Plot with New Montage

**User Story:** As a researcher, I want the SCI comparison plot to work correctly with the new 24-pair montage, so that I can assess signal quality stability across the recording.

#### Acceptance Criteria

1. WHEN the SCI_Comparator runs on fNIRS intensity data built from the new montage, THE SCI_Comparator SHALL produce a PNG file matching the BIDS naming pattern `sub-012_ses-001_task-fingertapping_desc-sci_comparison.png`.
2. WHEN the SCI_Comparator generates the plot, THE SCI_Comparator SHALL include one bar per source-detector pair (up to 24 pairs) for each recording segment (Initial and Final).
3. WHEN the SCI_Comparator generates the plot, THE SCI_Comparator SHALL draw a horizontal threshold line at the configured SCI threshold value.

### Requirement 5: Filter Bad Channels Using SCI Threshold 0.50

**User Story:** As a researcher, I want channels with SCI ≤ 0.50 excluded from fNIRS visualizations, so that plots reflect only channels with adequate scalp coupling.

#### Acceptance Criteria

1. WHEN the Analysis_Script initializes for sub-012, THE Channel_Filter SHALL override the config `sci_threshold` to 0.50 before quality assessment runs.
2. WHEN `calculate_sci()` returns SCI values, THE Channel_Filter SHALL classify each source-detector pair with SCI > 0.50 as a Good_Channel and each pair with SCI ≤ 0.50 as a Bad_Channel.
3. WHEN the Channel_Filter has identified Bad_Channels, THE Channel_Filter SHALL mark all wavelength channels (both 760nm and 850nm) belonging to Bad_Channel pairs as `bads` in the MNE info structure of both the processed fNIRS Raw and the fNIRS Epochs objects.
4. WHEN the Timeseries_Plotter, ROI_HRF_Plotter, or any fNIRS visualization function receives a `good_channels` list, THE visualization function SHALL include only Good_Channel data in the rendered plot.
5. WHEN all channels in a given ROI are Bad_Channels, THE ROI_HRF_Plotter SHALL display an empty subplot with a text annotation indicating no good channels are available for that ROI.

### Requirement 6: Full-Experiment HbO/HbR Time Series Plot

**User Story:** As a researcher, I want a time series plot showing mean HbO and HbR concentration across good channels for the entire experiment, so that I can visually inspect hemodynamic trends and stimulus-locked responses.

#### Acceptance Criteria

1. WHEN the Timeseries_Plotter receives preprocessed haemoglobin data, THE Timeseries_Plotter SHALL produce a figure with two vertically stacked subplots: HbO on top, HbR on bottom.
2. WHEN the Timeseries_Plotter renders each subplot, THE Timeseries_Plotter SHALL plot the mean concentration trace across good channels as a solid line and the ±1 standard deviation band as a shaded region.
3. WHEN the Timeseries_Plotter renders the plot, THE Timeseries_Plotter SHALL label the X-axis as time in seconds and the Y-axis as concentration in μM.
4. WHEN annotations are present in the haemoglobin data, THE Timeseries_Plotter SHALL draw vertical dashed lines at each stimulus onset, color-coded by condition (LEFT=green, RIGHT=purple, NOTHING=gray).
5. WHEN the Timeseries_Plotter completes rendering, THE Timeseries_Plotter SHALL save the figure as a PNG file matching the BIDS naming pattern `sub-012_ses-001_task-fingertapping_desc-fnirs_timeseries.png`.
6. IF the haemoglobin data contains zero good HbO or HbR channels, THEN THE Timeseries_Plotter SHALL log a warning and return None without saving a file.

### Requirement 7: 4-ROI HRF-by-Condition Plot

**User Story:** As a researcher, I want an HRF plot organized by 4 anatomical ROIs (Left Anterior, Right Anterior, Left Posterior, Right Posterior), so that I can compare hemodynamic responses across brain regions and experimental conditions.

#### Acceptance Criteria

1. WHEN the ROI_HRF_Plotter receives fNIRS epochs, THE ROI_HRF_Plotter SHALL produce a figure with 4 rows (one per ROI) and 2 columns (HbO, HbR), totaling 8 subplots.
2. WHEN the ROI_HRF_Plotter assigns channels to ROIs, THE ROI_HRF_Plotter SHALL use source-label prefixes from the sub-012 montage: Left Anterior = S1_, S9_; Right Anterior = S2_, S10_; Left Posterior = S3_, S4_, S5_; Right Posterior = S6_, S7_, S8_.
3. WHEN the ROI_HRF_Plotter renders each subplot, THE ROI_HRF_Plotter SHALL plot separate colored traces for each condition present in the epochs (LEFT=green, RIGHT=purple, NOTHING=gray), with the number of averaged trials shown in the legend.
4. WHEN the ROI_HRF_Plotter renders each subplot, THE ROI_HRF_Plotter SHALL draw a vertical dashed line at time = 0 (stimulus onset).
5. WHEN the ROI_HRF_Plotter completes rendering, THE ROI_HRF_Plotter SHALL save the figure as a PNG file matching the BIDS naming pattern `sub-012_ses-001_task-fingertapping_desc-fnirs_hrf_4roi.png`.
6. WHEN the ROI_HRF_Plotter filters channels, THE ROI_HRF_Plotter SHALL exclude Bad_Channels from the ROI averages using the `good_channels` list.

### Requirement 8: ROI-Specific Short Channel Regression

**User Story:** As a researcher, I want each cluster of long fNIRS channels to have its superficial physiological noise removed by regressing the signal from the spatially nearest short channel, so that the hemodynamic data reflects cortical activity rather than systemic contamination (scalp blood flow, Mayer waves, respiration artifacts).

#### Acceptance Criteria

1. WHEN the fNIRS preprocessing pipeline runs for sub-012, THE SCR_Verifier SHALL confirm that `process_fnirs_pipeline()` is called with `apply_scr=True` and that the montage JSON contains correct 3D coordinates for all 4 short channel sources (S13, S14, S15, S16) and their detectors (D1, D3, D6, D9).
2. WHEN `apply_short_channel_regression()` executes, THE SCR_Verifier SHALL verify that MNE-NIRS proximity-based pairing assigns each short channel to its spatially nearest long channels, resulting in the following mapping: S13→D1 regresses Left Anterior long channels (S1, S9 cluster), S14→D3 regresses Right Anterior long channels (S2, S10 cluster), S15→D6 regresses Left Posterior long channels (S3, S4, S5 cluster), S16→D9 regresses Right Posterior long channels (S6, S7, S8 cluster).
3. WHEN short channel regression completes, THE SCR_Verifier SHALL log the actual short-to-long channel pairing used by MNE-NIRS, listing which short channel was used as regressor for each long channel.
4. WHEN short channel regression completes, THE SCR_Verifier SHALL log the mean power reduction in the systemic band (0.1–0.4 Hz) for each ROI cluster, confirming that Mayer wave and respiration artifacts are attenuated.
5. IF the montage 3D coordinates cause incorrect short-to-long pairing (a short channel is paired with long channels from a different ROI), THEN THE Analysis_Script SHALL log a warning identifying the mismatch and fall back to an explicit ROI-based pairing that overrides the proximity-based assignment.
6. IF any of the 4 short channels is missing from the montage or has been marked as a Bad_Channel, THEN THE Analysis_Script SHALL log a warning identifying which ROI lacks short channel regression and proceed without regression for that ROI's long channels only.
7. WHEN the short channel regression is applied, THE Analysis_Script SHALL ensure it occurs in optical density (OD) space before the modified Beer-Lambert law conversion, following the standard MNE-NIRS processing order: Intensity → OD → Motion Correction (TDDR) → Short Channel Regression → Beer-Lambert → Bandpass Filter.
