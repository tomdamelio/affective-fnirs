#!/usr/bin/env python3
"""
Dedicated Analysis Script for Sub-012 (Modified Fingertapping Protocol).

This script implements the specific requirements for sub-012:
1. Handles 'sub-12' XDF filename discrepancy (vs 'sub-012')
2. Syncs NOTHING epochs from post-trial rest periods (10s task + 8s rest)
3. Runs standard validation pipeline (TFR, ERP, CSP)
4. Uses 'Agg' backend to prevent interactive plots from blocking execution.
"""
import matplotlib
# Force non-interactive backend BEFORE importing run_analysis or mne
matplotlib.use('Agg')

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import mne

# Add src to path for imports
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent / "src"))

from affective_fnirs.config import SubjectConfig
from affective_fnirs.ingestion import (
    load_xdf_file,
    identify_streams,
    extract_stream_data,
)
from affective_fnirs.mne_builder import (
    build_eeg_raw,
    embed_events,
)
from affective_fnirs.validation import validate_nothing_condition
import run_analysis as main_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class SynthesisStats:
    """Tracks NOTHING annotation synthesis outcomes.

    Attributes:
        n_created: Number of NOTHING annotations successfully created.
        n_skipped: Number of trials skipped due to insufficient rest.
        n_source_trials: Total LEFT/RIGHT trials found as synthesis sources.
    """

    n_created: int
    n_skipped: int
    n_source_trials: int


def synthesize_nothing_annotations(
    raw: mne.io.Raw,
    task_duration: float = 7.0,
    rest_duration_cap: float = 6.0,
) -> tuple[mne.io.Raw, dict[str, int]]:
    """Synthesize NOTHING annotations from post-trial rest periods.

    Creates a NOTHING annotation for each LEFT/RIGHT trial whose subsequent
    rest period is long enough.  The virtual onset is placed 1.0 s after
    stimulus offset (task_end + 1.0) to allow a 1-s baseline gap, and the
    annotation duration equals ``rest_duration_cap`` (6.0 s by default),
    matching LEFT/RIGHT epoch length.

    Args:
        raw: MNE Raw object containing LEFT/RIGHT annotations.
        task_duration: Duration of the motor task in seconds.
        rest_duration_cap: Duration assigned to each NOTHING annotation
            (epoch length).  The minimum required rest between trials is
            ``rest_duration_cap + 1.0`` (baseline gap + epoch).

    Returns:
        A tuple of (modified raw, synthesis_stats) where synthesis_stats is
        a dict with keys ``n_created``, ``n_skipped``, ``n_source_trials``.
    """
    logger.info("Synthesizing NOTHING annotations from post-trial rest...")

    onsets = raw.annotations.onset
    descriptions = raw.annotations.description
    sort_idx = np.argsort(onsets)
    sorted_onsets = onsets[sort_idx]
    sorted_descriptions = descriptions[sort_idx]

    new_onsets: list[float] = []
    new_durations: list[float] = []
    new_descriptions: list[str] = []

    n_source_trials = 0
    n_skipped = 0

    for trial_idx in range(len(sorted_onsets)):
        description = sorted_descriptions[trial_idx]
        onset = sorted_onsets[trial_idx]

        is_left = description == "LEFT" or description.startswith("LEFT/")
        is_right = description == "RIGHT" or description.startswith("RIGHT/")

        if not (is_left or is_right):
            continue

        n_source_trials += 1
        task_end = onset + task_duration
        nothing_virtual_onset = task_end + 1.0

        # Find next LEFT/RIGHT trial onset to bound available rest
        next_trial_onset: Optional[float] = None
        for search_idx in range(trial_idx + 1, len(sorted_onsets)):
            next_desc = sorted_descriptions[search_idx]
            if (
                next_desc == "LEFT"
                or next_desc.startswith("LEFT/")
                or next_desc == "RIGHT"
                or next_desc.startswith("RIGHT/")
            ):
                next_trial_onset = sorted_onsets[search_idx]
                break

        if next_trial_onset is not None:
            available_rest_duration = next_trial_onset - task_end
            if available_rest_duration >= (rest_duration_cap + 1.0):
                duration = rest_duration_cap
            else:
                n_skipped += 1
                continue
        else:
            # Last trial — assume sufficient rest
            duration = rest_duration_cap

        new_onsets.append(nothing_virtual_onset)
        new_durations.append(duration)
        new_descriptions.append("NOTHING")

    n_created = len(new_onsets)
    synthesis_stats: dict[str, int] = {
        "n_created": n_created,
        "n_skipped": n_skipped,
        "n_source_trials": n_source_trials,
    }

    if not new_onsets:
        logger.warning(
            "No NOTHING annotations created "
            f"(source trials: {n_source_trials}, skipped: {n_skipped})"
        )
        return raw, synthesis_stats

    nothing_annotations = mne.Annotations(
        onset=new_onsets,
        duration=new_durations,
        description=new_descriptions,
        orig_time=raw.annotations.orig_time,
    )

    raw.set_annotations(raw.annotations + nothing_annotations)
    logger.info(
        f"NOTHING synthesis complete — "
        f"created: {n_created}, skipped: {n_skipped}, "
        f"source trials: {n_source_trials}"
    )
    return raw, synthesis_stats



def main():
    parser = argparse.ArgumentParser(description="Sub-012 Analysis")
    parser.add_argument("--config", type=Path, default=Path("configs/sub-012.yml"), help="Path to config file")
    args = parser.parse_args()
    
    # 1. Load Configuration
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
        
    config = SubjectConfig.from_yaml(args.config)
    main_pipeline.print_configuration_summary(config)
    
    # 2. Custom Data Loading (Handle sub-12 vs sub-012)
    # Explicitly check for the 'sub-12' filename variant
    xdf_filename = f"sub-12_ses-{config.subject.session}_task-{config.subject.task}_recording.xdf"
    xdf_path = config.data_root / f"sub-{config.subject.id}" / f"ses-{config.subject.session}" / xdf_filename
    
    if not xdf_path.exists():
        # Try flat structure
        xdf_path = config.data_root / f"sub-{config.subject.id}" / xdf_filename
        
    # If still not found, try the standard 'sub-012' name using the main pipeline's logic
    if not xdf_path.exists():
        logger.warning(f"Custom sub-12 file not found at {xdf_path}, trying standard naming...")
        try:
             # This might fail if load_and_identify_streams is too strict, but worth a try as fallback
             streams = main_pipeline.load_and_identify_streams(config)
        except Exception:
             logger.error(f"XDF file not found. Checked custom path: {xdf_path}")
             sys.exit(1)
    else:
        logger.info(f"Loading XDF from custom path: {xdf_path}")
        streams, header = load_xdf_file(xdf_path)
        streams = identify_streams(streams)

    # 3. Build MNE Object
    # Use main pipeline's build function to get both EEG and fNIRS
    try:
        raw_eeg, raw_fnirs = main_pipeline.build_mne_objects(streams, config)
    except Exception as e:
        logger.error(f"Failed to build MNE objects: {e}")
        sys.exit(1)

    if raw_eeg is None:
        logger.error("No EEG stream found")
        sys.exit(1)
        
    if raw_fnirs is not None:
        logger.info(f"fNIRS stream found: {len(raw_fnirs.ch_names)} channels")
    else:
        logger.warning("No fNIRS stream found (unexpected for sub-012)")
        
    # 4. Synthesize NOTHING conditions
    # Task is 7s, NOTHING epochs are 6s (matching LEFT/RIGHT epoch duration)
    raw_eeg, eeg_synthesis_stats = synthesize_nothing_annotations(
        raw_eeg, task_duration=7.0, rest_duration_cap=6.0
    )
    
    if raw_fnirs is not None:
        # Run synthesis on fNIRS too — it has its own copy of annotations
        raw_fnirs, fnirs_synthesis_stats = synthesize_nothing_annotations(
            raw_fnirs, task_duration=7.0, rest_duration_cap=6.0
        )
    
    # 5. Preprocessing (Reuse main pipeline)
    # Output path
    output_path = config.output_root / f"sub-{config.subject.id}" / f"ses-{config.subject.session}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Pass both EEG and fNIRS
    processed_eeg, processed_fnirs = main_pipeline.run_preprocessing(
        raw_eeg=raw_eeg,
        raw_fnirs=raw_fnirs, 
        config=config,
        output_path=output_path
    )
    
    if processed_eeg is None:
        logger.error("EEG Preprocessing failed")
        sys.exit(1)
        
    # 6. Analysis and Visualization
    # Run Standard EEG Analysis (Epoching, TFR, ERD/ERS)
    logger.info("Running EEG Analysis...")
    eeg_results = main_pipeline.run_eeg_analysis(processed_eeg, config, output_path)
    
    # Run fNIRS Analysis
    fnirs_results = None
    if processed_fnirs is not None:
        logger.info("Running fNIRS Analysis...")
        # Check if function exists
        if hasattr(main_pipeline, 'run_fnirs_analysis'):
            fnirs_results = main_pipeline.run_fnirs_analysis(processed_fnirs, config)
        else:
            logger.warning("run_fnirs_analysis function not found in main_pipeline")

    # 6.5 Validate NOTHING condition integrity (advisory, non-blocking)
    logger.info("Validating NOTHING condition...")
    fnirs_epochs_for_validation = (
        fnirs_results["epochs"] if fnirs_results and "epochs" in fnirs_results else None
    )
    eeg_epochs_for_validation = (
        eeg_results["epochs"] if eeg_results and "epochs" in eeg_results else None
    )
    validation_result = validate_nothing_condition(
        eeg_epochs=eeg_epochs_for_validation,
        fnirs_epochs=fnirs_epochs_for_validation,
    )
    if validation_result.all_passed:
        logger.info(
            "NOTHING validation: all checks passed "
            "(%d NOTHING, %d LEFT, %d RIGHT epochs)",
            validation_result.n_nothing_epochs,
            validation_result.n_left_epochs,
            validation_result.n_right_epochs,
        )
    else:
        logger.warning(
            "NOTHING validation: %d issue(s) detected — %s",
            len(validation_result.warnings),
            "; ".join(validation_result.warnings),
        )

    # Process EEG Results (CSP, Viz)
    viz_paths = {}
    if eeg_results:
        # Run CSP (LEFT vs RIGHT)
        # Note: This function requires 'epochs' in eeg_results
        logger.info("Running CSP Analysis (LEFT vs RIGHT)...")
        csp_path, csp_results = main_pipeline.generate_csp_analysis(eeg_results['epochs'], output_path, config)
        eeg_results['csp_analysis_path'] = csp_path
        eeg_results['csp_results'] = csp_results
        
        # Run CSP (MOV vs REST)
        # Verify module path and function existence
        logger.info(f"Loaded main_pipeline from: {main_pipeline.__file__}")
        if not hasattr(main_pipeline, 'generate_csp_movement_vs_rest'):
             logger.error("generate_csp_movement_vs_rest NOT FOUND in main_pipeline!")
        
        logger.info(f"Event IDs in epochs: {eeg_results['epochs'].event_id}")

        logger.info("Running CSP Analysis (MOV vs NO MOV)...")
        csp_mov_path, csp_mov_results = main_pipeline.generate_csp_movement_vs_rest(eeg_results['epochs'], output_path, config)
        eeg_results['csp_mov_vs_rest_path'] = csp_mov_path
        eeg_results['csp_mov_results'] = csp_mov_results
        
        # Generate Individual Visualizations with all 3 conditions (LEFT, RIGHT, NOTHING)
        logger.info("Generating individual visualizations with all conditions...")
        epochs = eeg_results['epochs']
        
        # Add CSP paths to viz_paths for report generation
        if 'csp_analysis_path' in eeg_results and eeg_results['csp_analysis_path']:
             viz_paths['eeg_csp_analysis'] = eeg_results['csp_analysis_path']
        if 'csp_mov_vs_rest_path' in eeg_results and eeg_results['csp_mov_vs_rest_path']:
             viz_paths['eeg_csp_mov_vs_rest'] = eeg_results['csp_mov_vs_rest_path']

        # TFR Maps (shows LEFT, RIGHT, NOTHING in separate columns)
        logger.info("Generating TFR maps...")
        tfr_path = main_pipeline.generate_tfr_maps(epochs, output_path, config)
        if tfr_path:
            viz_paths['eeg_tfr_maps'] = tfr_path
        
        # ERP Analysis (shows all 3 conditions)
        logger.info("Generating ERP analysis...")
        erp_path = main_pipeline.generate_erp_analysis(epochs, output_path, config)
        if erp_path:
            viz_paths['eeg_erp_analysis'] = erp_path
        
        # Clustered TFR Maps
        logger.info("Generating clustered TFR maps...")
        clustered_tfr_path = main_pipeline.generate_clustered_tfr_maps(epochs, output_path, config)
        if clustered_tfr_path:
            viz_paths['eeg_tfr_maps_roi'] = clustered_tfr_path
        
        # Beta Topoplots
        logger.info("Generating beta topoplots...")
        beta_topo_path = main_pipeline.generate_beta_topoplots(epochs, output_path, config)
        if beta_topo_path:
            viz_paths['eeg_beta_topoplot'] = beta_topo_path
        
        # Contralateral ERD plots
        logger.info("Generating contralateral ERD plots...")
        contralat_timecourse, contralat_topo = main_pipeline.generate_contralateral_erd_plots(epochs, output_path, config)
        if contralat_topo:
            viz_paths['eeg_contralateral_topoplot'] = contralat_topo
        if contralat_timecourse:
            viz_paths['eeg_contralateral_timecourse'] = contralat_timecourse
            
        # Contrast Analysis
        logger.info("Generating contrast analysis...")
        contrast_path, lat_index_path = main_pipeline.generate_contrast_analysis(epochs, output_path, config)
        if contrast_path:
            viz_paths['eeg_contrast_analysis'] = contrast_path
        
        # fNIRS Visualizations
        if fnirs_results and 'epochs' in fnirs_results:
            logger.info("Generating fNIRS visualizations with all conditions...")
            fnirs_epochs = fnirs_results['epochs']
            
            # HRF by Condition (shows LEFT, RIGHT, NOTHING)
            logger.info("Generating fNIRS HRF by condition...")
            hrf_path = main_pipeline.generate_fnirs_hrf_by_condition(fnirs_epochs, output_path, config)
            if hrf_path:
                viz_paths['fnirs_hrf_by_condition'] = hrf_path
            
            # Block Average
            logger.info("Generating fNIRS block average...")
            block_avg_path = main_pipeline.generate_fnirs_block_average(fnirs_epochs, output_path, config)
            if block_avg_path:
                viz_paths['fnirs_block_average'] = block_avg_path
            
            # Contrast Map
            logger.info("Generating fNIRS contrast map...")
            contrast_path = main_pipeline.generate_fnirs_contrast_map(fnirs_epochs, output_path, config)
            if contrast_path:
                viz_paths['fnirs_contrast'] = contrast_path

    else:
        logger.error("EEG Analysis failed to produce results.")


    # 7. Quality Assessment (Reuse main pipeline)
    qa_results = main_pipeline.run_quality_assessment(raw_eeg, raw_fnirs, config)

    # 8. Save Full Report
    logger.info("Saving Full Report...")
    main_pipeline.save_full_report(
        qa_results=qa_results,
        eeg_results=eeg_results,
        fnirs_results=fnirs_results,
        multimodal_results=None,
        visualization_paths=viz_paths,
        config=config,
        output_path=output_path
    )
    
    logger.info(f"Sub-012 Analysis Complete. Results available in {output_path}")

if __name__ == "__main__":
    main()
