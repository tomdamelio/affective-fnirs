"""
Test suite for Checkpoint 14: Verify reporting functionality.

This test validates:
- BIDS-compliant filenames and paths
- HTML report contains all required sections
- JSON data dictionaries accompany TSV files

Requirements:
    - Req. 8.1-8.6: Validation report generation
    - Req. 9.2-9.4: BIDS compliance
    - Req. 10.2-10.3: Reproducibility
"""

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pytest

from affective_fnirs.reporting import (
    ChannelQuality,
    CouplingMetrics,
    ERDMetrics,
    HRFValidation,
    QualityReport,
    ValidationResults,
    generate_quality_report,
    generate_validation_conclusions,
    generate_validation_report_html,
    save_numerical_results,
)


@pytest.fixture
def sample_quality_report():
    """Create sample quality report for testing."""
    channels = [
        ChannelQuality(
            channel_name="S1_D1 760",
            sci=0.85,
            saturation_percent=2.1,
            cardiac_power=0.15,
            cv=8.5,
            is_bad=False,
            reason="",
        ),
        ChannelQuality(
            channel_name="S1_D1 850",
            sci=0.82,
            saturation_percent=2.3,
            cardiac_power=0.14,
            cv=9.2,
            is_bad=False,
            reason="",
        ),
        ChannelQuality(
            channel_name="S2_D2 760",
            sci=0.45,
            saturation_percent=8.5,
            cardiac_power=0.08,
            cv=18.3,
            is_bad=True,
            reason="Low SCI (0.45 < 0.8), High saturation (8.5% > 5%)",
        ),
    ]

    return QualityReport(
        channels=channels,
        n_total_channels=3,
        n_bad_channels=1,
        mean_sci=0.71,
        mean_saturation=4.3,
        mean_cardiac_power=0.12,
        mean_cv=12.0,
    )


@pytest.fixture
def sample_erd_metrics():
    """Create sample ERD metrics for testing."""
    return ERDMetrics(
        channel="C3",
        alpha_erd_percent=-32.5,
        alpha_p_value=0.003,
        alpha_significant=True,
        beta_erd_percent=-45.2,
        beta_p_value=0.001,
        beta_significant=True,
        beta_rebound_percent=18.3,
        beta_rebound_observed=True,
    )


@pytest.fixture
def sample_hrf_validation():
    """Create sample HRF validation metrics for testing."""
    return HRFValidation(
        channel="CCP3h-CP3 hbo",
        onset_time_sec=2.4,
        onset_detected=True,
        time_to_peak_sec=6.2,
        peak_plausible=True,
        plateau_amplitude_um=0.85,
        plateau_p_value=0.002,
        plateau_significant=True,
        trial_consistency_r=0.72,
        consistency_acceptable=True,
    )


@pytest.fixture
def sample_coupling_metrics():
    """Create sample coupling metrics for testing."""
    return CouplingMetrics(
        max_correlation=0.68,
        lag_seconds=-3.2,
        eeg_precedes_fnirs=True,
        correlation_adequate=True,
    )


@pytest.fixture
def sample_validation_results(
    sample_quality_report, sample_erd_metrics, sample_hrf_validation, sample_coupling_metrics
):
    """Create complete validation results for testing."""
    return ValidationResults(
        subject_id="002",
        session_id="001",
        task="fingertapping",
        timestamp="2024-01-15T14:30:00",
        software_versions={
            "python": "3.11.0",
            "mne": "1.6.0",
            "mne-nirs": "0.6.0",
            "numpy": "1.26.4",
        },
        config={
            "filter": {"eeg_bandpass": [1, 40], "fnirs_bandpass": [0.01, 0.5]},
            "quality": {"sci_threshold": 0.8, "cv_threshold": 15.0},
        },
        quality_report=sample_quality_report,
        erd_metrics=sample_erd_metrics,
        hrf_validation=sample_hrf_validation,
        coupling_metrics=sample_coupling_metrics,
    )


class TestBIDSCompliance:
    """Test BIDS-compliant filenames and paths."""

    def test_quality_report_bids_filenames(self, sample_quality_report):
        """Verify quality report generates BIDS-compliant filenames."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "data" / "derivatives" / "validation-pipeline" / "sub-002"

            tsv_path, json_path, _ = generate_quality_report(
                quality_report=sample_quality_report,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
                heatmap_figure=None,
            )

            # Check BIDS entity ordering: sub-XX_ses-XX_task-XX_desc-XX
            expected_pattern = r"sub-002_ses-001_task-fingertapping_desc-quality_channels\.(tsv|json)"

            assert re.match(expected_pattern, tsv_path.name), f"TSV filename not BIDS-compliant: {tsv_path.name}"
            assert re.match(expected_pattern, json_path.name), f"JSON filename not BIDS-compliant: {json_path.name}"

            # Verify files exist
            assert tsv_path.exists(), "TSV file not created"
            assert json_path.exists(), "JSON file not created"

    def test_numerical_results_bids_filename(self, sample_validation_results):
        """Verify numerical results use BIDS-compliant filename."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "data" / "derivatives" / "validation-pipeline" / "sub-002"

            json_path = save_numerical_results(
                validation_results=sample_validation_results,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            # Check BIDS entity ordering
            expected_pattern = r"sub-002_ses-001_task-fingertapping_desc-validation_metrics\.json"
            assert re.match(expected_pattern, json_path.name), f"Filename not BIDS-compliant: {json_path.name}"

            # Verify file exists
            assert json_path.exists(), "JSON file not created"

    def test_html_report_bids_filename(self, sample_validation_results):
        """Verify HTML report uses BIDS-compliant filename."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "data" / "derivatives" / "validation-pipeline" / "sub-002"

            # Create dummy figures
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            figures = {"quality_heatmap": fig}

            html_path = generate_validation_report_html(
                validation_results=sample_validation_results,
                figures=figures,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            plt.close(fig)

            # Check BIDS entity ordering
            expected_pattern = r"sub-002_ses-001_task-fingertapping_desc-validation_report\.html"
            assert re.match(expected_pattern, html_path.name), f"Filename not BIDS-compliant: {html_path.name}"

            # Verify file exists
            assert html_path.exists(), "HTML file not created"

    def test_derivatives_directory_structure(self, sample_quality_report):
        """Verify outputs go to correct derivatives directory."""
        with TemporaryDirectory() as tmpdir:
            # Correct BIDS derivatives path
            output_path = Path(tmpdir) / "data" / "derivatives" / "validation-pipeline" / "sub-002"

            tsv_path, json_path, _ = generate_quality_report(
                quality_report=sample_quality_report,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            # Verify path structure
            assert "derivatives" in str(tsv_path), "Output not in derivatives directory"
            assert "validation-pipeline" in str(tsv_path), "Output not in pipeline-specific directory"
            assert "sub-002" in str(tsv_path), "Output not in subject-specific directory"


class TestJSONDataDictionaries:
    """Test JSON data dictionaries accompany TSV files."""

    def test_quality_report_has_data_dictionary(self, sample_quality_report):
        """Verify quality report TSV has accompanying JSON data dictionary."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            tsv_path, json_path, _ = generate_quality_report(
                quality_report=sample_quality_report,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            # Verify JSON exists
            assert json_path.exists(), "JSON data dictionary not created"

            # Load and validate JSON structure
            with open(json_path) as f:
                data_dict = json.load(f)

            # Check all TSV columns have descriptions
            expected_columns = [
                "channel_name",
                "sci",
                "saturation_percent",
                "cardiac_power",
                "cv",
                "is_bad",
                "reason",
            ]

            for col in expected_columns:
                assert col in data_dict, f"Column '{col}' missing from data dictionary"
                assert "Description" in data_dict[col], f"Column '{col}' missing Description field"

    def test_data_dictionary_contains_metadata(self, sample_quality_report):
        """Verify data dictionary contains required metadata fields."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            _, json_path, _ = generate_quality_report(
                quality_report=sample_quality_report,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            with open(json_path) as f:
                data_dict = json.load(f)

            # Check SCI has all required metadata
            sci_metadata = data_dict["sci"]
            assert "Description" in sci_metadata, "SCI missing Description"
            assert "Units" in sci_metadata, "SCI missing Units"
            assert "Range" in sci_metadata, "SCI missing Range"
            assert "Threshold" in sci_metadata, "SCI missing Threshold"
            assert "Reference" in sci_metadata, "SCI missing Reference (PMC citation)"

            # Verify reference to scientific literature
            assert "PMC" in sci_metadata["Reference"], "SCI missing PMC reference"

    def test_data_dictionary_matches_tsv_columns(self, sample_quality_report):
        """Verify data dictionary columns match TSV columns exactly."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            tsv_path, json_path, _ = generate_quality_report(
                quality_report=sample_quality_report,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            # Load TSV columns
            import pandas as pd

            df = pd.read_csv(tsv_path, sep="\t")
            tsv_columns = set(df.columns)

            # Load JSON columns
            with open(json_path) as f:
                data_dict = json.load(f)
            json_columns = set(data_dict.keys())

            # Verify exact match
            assert tsv_columns == json_columns, f"Column mismatch: TSV={tsv_columns}, JSON={json_columns}"


class TestHTMLReportSections:
    """Test HTML report contains all required sections."""

    def test_html_report_contains_header_section(self, sample_validation_results):
        """Verify HTML report contains header with metadata."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            figures = {"quality_heatmap": fig}

            html_path = generate_validation_report_html(
                validation_results=sample_validation_results,
                figures=figures,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            plt.close(fig)

            # Read HTML content
            with open(html_path) as f:
                html_content = f.read()

            # Check header elements
            assert "sub-002" in html_content, "Subject ID missing from header"
            assert "ses-001" in html_content, "Session ID missing from header"
            assert "fingertapping" in html_content, "Task missing from header"
            assert "2024-01-15" in html_content, "Timestamp missing from header"

            # Check software versions
            assert "python" in html_content.lower(), "Python version missing"
            assert "mne" in html_content.lower(), "MNE version missing"

    def test_html_report_contains_quality_section(self, sample_validation_results):
        """Verify HTML report contains quality assessment section."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            fig, ax = plt.subplots()
            figures = {"quality_heatmap": fig}

            html_path = generate_validation_report_html(
                validation_results=sample_validation_results,
                figures=figures,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            plt.close(fig)

            with open(html_path) as f:
                html_content = f.read()

            # Check quality metrics present
            assert "Quality Assessment" in html_content or "quality" in html_content.lower()
            assert "SCI" in html_content or "Scalp Coupling" in html_content
            assert "Bad Channels" in html_content or "bad" in html_content.lower()

    def test_html_report_contains_eeg_section(self, sample_validation_results):
        """Verify HTML report contains EEG analysis section."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            fig, ax = plt.subplots()
            figures = {"eeg_spectrogram": fig}

            html_path = generate_validation_report_html(
                validation_results=sample_validation_results,
                figures=figures,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            plt.close(fig)

            with open(html_path) as f:
                html_content = f.read()

            # Check EEG elements
            assert "EEG" in html_content, "EEG section missing"
            assert "ERD" in html_content or "Desynchronization" in html_content, "ERD analysis missing"
            assert "Alpha" in html_content, "Alpha band missing"
            assert "Beta" in html_content, "Beta band missing"
            assert "C3" in html_content, "Motor cortex channel missing"

    def test_html_report_contains_fnirs_section(self, sample_validation_results):
        """Verify HTML report contains fNIRS analysis section."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            fig, ax = plt.subplots()
            figures = {"hrf_curves": fig}

            html_path = generate_validation_report_html(
                validation_results=sample_validation_results,
                figures=figures,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            plt.close(fig)

            with open(html_path) as f:
                html_content = f.read()

            # Check fNIRS elements
            assert "fNIRS" in html_content or "NIRS" in html_content, "fNIRS section missing"
            assert "HRF" in html_content or "Hemodynamic" in html_content, "HRF analysis missing"
            assert "HbO" in html_content, "HbO missing"
            assert "onset" in html_content.lower(), "Onset time missing"

    def test_html_report_contains_coupling_section(self, sample_validation_results):
        """Verify HTML report contains multimodal coupling section."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            fig, ax = plt.subplots()
            figures = {"coupling_overlay": fig}

            html_path = generate_validation_report_html(
                validation_results=sample_validation_results,
                figures=figures,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            plt.close(fig)

            with open(html_path) as f:
                html_content = f.read()

            # Check coupling elements
            assert "Coupling" in html_content or "coupling" in html_content.lower(), "Coupling section missing"
            assert "correlation" in html_content.lower(), "Correlation missing"
            assert "lag" in html_content.lower(), "Lag missing"

    def test_html_report_contains_validation_summary(self, sample_validation_results):
        """Verify HTML report contains validation summary/conclusions."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            fig, ax = plt.subplots()
            figures = {"quality_heatmap": fig}

            html_path = generate_validation_report_html(
                validation_results=sample_validation_results,
                figures=figures,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            plt.close(fig)

            with open(html_path) as f:
                html_content = f.read()

            # Check validation summary
            assert "Validation" in html_content or "Summary" in html_content, "Validation summary missing"
            assert "PASS" in html_content or "FAIL" in html_content, "Pass/fail criteria missing"

    def test_html_report_contains_configuration_section(self, sample_validation_results):
        """Verify HTML report contains configuration section."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            fig, ax = plt.subplots()
            figures = {"quality_heatmap": fig}

            html_path = generate_validation_report_html(
                validation_results=sample_validation_results,
                figures=figures,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            plt.close(fig)

            with open(html_path) as f:
                html_content = f.read()

            # Check configuration present
            assert "Configuration" in html_content or "config" in html_content.lower(), "Configuration section missing"
            assert "filter" in html_content.lower(), "Filter config missing"
            assert "quality" in html_content.lower(), "Quality config missing"


class TestValidationConclusions:
    """Test validation conclusions generation."""

    def test_conclusions_evaluate_all_criteria(self, sample_validation_results):
        """Verify conclusions evaluate all validation criteria."""
        conclusions = generate_validation_conclusions(sample_validation_results)

        # Check all criteria categories present
        assert "EEG" in conclusions, "EEG criteria missing"
        assert "fNIRS" in conclusions, "fNIRS criteria missing"
        assert "Coupling" in conclusions, "Coupling criteria missing"

        # Check specific criteria
        assert "Alpha ERD" in conclusions, "Alpha ERD criterion missing"
        assert "Beta ERD" in conclusions, "Beta ERD criterion missing"
        assert "HbO onset" in conclusions, "HbO onset criterion missing"
        assert "correlation" in conclusions.lower(), "Correlation criterion missing"

    def test_conclusions_show_pass_fail_status(self, sample_validation_results):
        """Verify conclusions show pass/fail status for each criterion."""
        conclusions = generate_validation_conclusions(sample_validation_results)

        # Check for pass/fail indicators
        assert "[PASS]" in conclusions or "[FAIL]" in conclusions, "Pass/fail status missing"

        # Check for summary counts
        assert "criteria met" in conclusions.lower(), "Criteria summary missing"

    def test_conclusions_provide_recommendations(self, sample_validation_results):
        """Verify conclusions provide actionable recommendations."""
        conclusions = generate_validation_conclusions(sample_validation_results)

        # Check for recommendation section
        assert "RECOMMENDATION" in conclusions, "Recommendation missing"

        # Should have overall assessment
        assert "OVERALL" in conclusions, "Overall assessment missing"

    def test_conclusions_include_diagnostic_suggestions(self):
        """Verify conclusions include diagnostic suggestions for failures."""
        # Create validation results with some failures
        failing_results = ValidationResults(
            subject_id="002",
            session_id="001",
            task="fingertapping",
            timestamp="2024-01-15T14:30:00",
            software_versions={"python": "3.11.0"},
            config={},
            quality_report=QualityReport(
                channels=[],
                n_total_channels=0,
                n_bad_channels=0,
                mean_sci=0.5,
                mean_saturation=0,
                mean_cardiac_power=0,
                mean_cv=0,
            ),
            erd_metrics=ERDMetrics(
                channel="C3",
                alpha_erd_percent=-10.0,  # Too weak
                alpha_p_value=0.15,  # Not significant
                alpha_significant=False,
                beta_erd_percent=-15.0,  # Too weak
                beta_p_value=0.08,
                beta_significant=False,
                beta_rebound_percent=5.0,
                beta_rebound_observed=False,
            ),
            hrf_validation=HRFValidation(
                channel="CCP3h-CP3 hbo",
                onset_time_sec=5.0,  # Too late
                onset_detected=False,
                time_to_peak_sec=12.0,  # Too late
                peak_plausible=False,
                plateau_amplitude_um=0.05,
                plateau_p_value=0.25,
                plateau_significant=False,
                trial_consistency_r=0.3,
                consistency_acceptable=False,
            ),
            coupling_metrics=CouplingMetrics(
                max_correlation=0.25,  # Too weak
                lag_seconds=2.0,  # Positive (wrong direction)
                eeg_precedes_fnirs=False,
                correlation_adequate=False,
            ),
        )

        conclusions = generate_validation_conclusions(failing_results)

        # Check for diagnostic suggestions
        assert "Diagnostic" in conclusions or "Suggestions" in conclusions, "Diagnostic suggestions missing"


class TestReproducibility:
    """Test reproducibility features in reporting."""

    def test_numerical_results_include_software_versions(self, sample_validation_results):
        """Verify numerical results include software versions."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            json_path = save_numerical_results(
                validation_results=sample_validation_results,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            with open(json_path) as f:
                results = json.load(f)

            # Check software versions present
            assert "software_versions" in results, "Software versions missing"
            assert "python" in results["software_versions"], "Python version missing"
            assert "mne" in results["software_versions"], "MNE version missing"

    def test_numerical_results_include_configuration(self, sample_validation_results):
        """Verify numerical results include pipeline configuration."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            json_path = save_numerical_results(
                validation_results=sample_validation_results,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            with open(json_path) as f:
                results = json.load(f)

            # Check configuration present
            assert "config" in results, "Configuration missing"
            assert "filter" in results["config"], "Filter config missing"
            assert "quality" in results["config"], "Quality config missing"

    def test_numerical_results_are_json_serializable(self, sample_validation_results):
        """Verify all numerical results can be serialized to JSON."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Should not raise exception
            json_path = save_numerical_results(
                validation_results=sample_validation_results,
                output_path=output_path,
                subject_id="002",
                session_id="001",
                task="fingertapping",
            )

            # Should be valid JSON
            with open(json_path) as f:
                results = json.load(f)

            # Verify structure is complete
            assert "quality_report" in results
            assert "erd_metrics" in results
            assert "hrf_validation" in results
            assert "coupling_metrics" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
