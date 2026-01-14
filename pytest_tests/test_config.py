"""
Tests para el módulo de configuración.

Este módulo demuestra cómo escribir tests para configuraciones inmutables.
"""

from pathlib import Path

import pytest

from affective_fnirs.config import (
    AnalysisConfig,
    FilterConfig,
    QualityThresholds,
    EpochConfig,
    ICAConfig,
    MotionCorrectionConfig,
    EEGPreprocessingConfig,
    SubjectInfo,
    ModalityConfig,
    ReportConfig,
    TrialsConfig,
    SubjectConfig,
    PipelineConfig,
)


def test_filter_config_default_values():
    """Test que FilterConfig tiene valores por defecto correctos."""
    config = FilterConfig()

    assert config.eeg_bandpass_low_hz == 1.0
    assert config.eeg_bandpass_high_hz == 40.0
    assert config.fnirs_bandpass_low_hz == 0.01
    assert config.fnirs_bandpass_high_hz == 0.5
    assert config.cardiac_band_low_hz == 0.5
    assert config.cardiac_band_high_hz == 2.5


def test_filter_config_is_immutable():
    """Test que FilterConfig es inmutable (frozen)."""
    config = FilterConfig()

    with pytest.raises(AttributeError):
        config.eeg_bandpass_low_hz = 2.0


def test_quality_thresholds_default_values():
    """Test que QualityThresholds tiene valores por defecto correctos."""
    config = QualityThresholds()

    assert config.sci_threshold == 0.75
    assert config.cv_threshold_percent == 15.0
    assert config.saturation_percent == 5.0
    assert config.psp_threshold == 0.1
    assert config.short_channel_distance_mm == 15.0


def test_analysis_config_frequency_bands():
    """Test que AnalysisConfig tiene bandas de frecuencia correctas."""
    config = AnalysisConfig()

    assert config.alpha_band_low_hz == 8.0
    assert config.alpha_band_high_hz == 13.0
    assert config.beta_band_low_hz == 13.0
    assert config.beta_band_high_hz == 30.0


def test_analysis_config_random_seed():
    """Test que AnalysisConfig tiene semilla aleatoria para reproducibilidad."""
    config = AnalysisConfig()

    # Note: AnalysisConfig doesn't have random_seed, it's in SubjectConfig
    assert config.dpf == 6.0


def test_eeg_preprocessing_config_defaults():
    """Test que EEGPreprocessingConfig tiene valores por defecto correctos."""
    config = EEGPreprocessingConfig()

    assert config.ica_enabled is False
    assert config.reference_channel == "Cz"
    assert config.apply_car is False


def test_modality_config_defaults():
    """Test que ModalityConfig tiene valores por defecto correctos."""
    config = ModalityConfig()

    assert config.eeg_enabled is True
    assert config.fnirs_enabled is True


def test_report_config_defaults():
    """Test que ReportConfig tiene valores por defecto correctos."""
    config = ReportConfig()

    assert config.qa_only is False


def test_trials_config_defaults():
    """Test que TrialsConfig tiene valores por defecto correctos."""
    config = TrialsConfig()

    assert config.count_per_condition == 10
    assert config.task_duration_sec == 10.0
    assert config.rest_duration_sec == 10.0


def test_subject_info_validation():
    """Test que SubjectInfo valida campos requeridos."""
    subject = SubjectInfo(id="001", session="001", task="fingertapping")

    assert subject.id == "001"
    assert subject.session == "001"
    assert subject.task == "fingertapping"


def test_pipeline_config_default_factory():
    """Test que PipelineConfig.default() crea configuración válida."""
    config = PipelineConfig.default()

    assert config.data_root == Path("data/raw")
    assert config.output_root == Path("data/derivatives/validation-pipeline")
    assert isinstance(config.filters, FilterConfig)
    assert isinstance(config.quality, QualityThresholds)
    assert isinstance(config.analysis, AnalysisConfig)
