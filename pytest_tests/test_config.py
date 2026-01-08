"""
Tests para el módulo de configuración.

Este módulo demuestra cómo escribir tests para configuraciones inmutables.
"""

from pathlib import Path

import pytest

from affective_fnirs.config import (
    AnalysisConfig,
    HemodynamicConfig,
    PipelineConfig,
    PreprocessingConfig,
)


def test_preprocessing_config_default_values():
    """Test que PreprocessingConfig tiene valores por defecto correctos."""
    config = PreprocessingConfig()

    assert config.sampling_rate_hz == 10.0
    assert config.lowpass_cutoff_hz == 0.5
    assert config.highpass_cutoff_hz == 0.01
    assert config.motion_correction_method == "spline"
    assert config.short_channel_regression is True
    assert config.sci_threshold == 0.5


def test_preprocessing_config_is_immutable():
    """Test que PreprocessingConfig es inmutable (frozen)."""
    config = PreprocessingConfig()

    with pytest.raises(AttributeError):
        config.sampling_rate_hz = 20.0


def test_hemodynamic_config_wavelengths():
    """Test que HemodynamicConfig tiene longitudes de onda correctas."""
    config = HemodynamicConfig()

    assert len(config.wavelengths_nm) == 2
    assert config.wavelengths_nm == (760.0, 850.0)
    assert config.dpf == (6.0, 6.0)


def test_analysis_config_random_seed():
    """Test que AnalysisConfig tiene semilla aleatoria para reproducibilidad."""
    config = AnalysisConfig()

    assert isinstance(config.random_seed, int)
    assert config.random_seed == 42


def test_pipeline_config_default_factory():
    """Test que PipelineConfig.default() crea configuración válida."""
    data_root = Path("data/raw")
    output_root = Path("outputs")

    config = PipelineConfig.default(data_root, output_root)

    assert config.data_root == data_root
    assert config.output_root == output_root
    assert isinstance(config.preprocessing, PreprocessingConfig)
    assert isinstance(config.hemodynamic, HemodynamicConfig)
    assert isinstance(config.analysis, AnalysisConfig)


def test_pipeline_config_custom_values():
    """Test que PipelineConfig acepta configuraciones personalizadas."""
    custom_preprocessing = PreprocessingConfig(
        sampling_rate_hz=20.0, motion_correction_method="wavelet"
    )

    config = PipelineConfig(
        preprocessing=custom_preprocessing,
        hemodynamic=HemodynamicConfig(),
        analysis=AnalysisConfig(),
        data_root=Path("data"),
        output_root=Path("outputs"),
    )

    assert config.preprocessing.sampling_rate_hz == 20.0
    assert config.preprocessing.motion_correction_method == "wavelet"
