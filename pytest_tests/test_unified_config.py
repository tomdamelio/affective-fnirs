"""
Tests for unified analysis pipeline configuration.

Validates SubjectConfig, SubjectInfo, ModalityConfig, ReportConfig, and TrialsConfig.
"""

from pathlib import Path

import pytest

from affective_fnirs.config import (
    SubjectConfig,
    SubjectInfo,
    ModalityConfig,
    ReportConfig,
    TrialsConfig,
)


def test_subject_info_creation():
    """Test that SubjectInfo can be created with valid values."""
    subject = SubjectInfo(id="010", session="001", task="fingertapping")
    
    assert subject.id == "010"
    assert subject.session == "001"
    assert subject.task == "fingertapping"


def test_modality_config_defaults():
    """Test that ModalityConfig has correct default values."""
    modalities = ModalityConfig()
    
    assert modalities.eeg_enabled is True
    assert modalities.fnirs_enabled is True


def test_modality_config_custom():
    """Test that ModalityConfig accepts custom values."""
    modalities = ModalityConfig(eeg_enabled=False, fnirs_enabled=True)
    
    assert modalities.eeg_enabled is False
    assert modalities.fnirs_enabled is True


def test_report_config_defaults():
    """Test that ReportConfig has correct default values."""
    report = ReportConfig()
    
    assert report.qa_only is False


def test_trials_config_defaults():
    """Test that TrialsConfig has correct default values."""
    trials = TrialsConfig()
    
    assert trials.count_per_condition == 10
    assert trials.task_duration_sec == 10.0
    assert trials.rest_duration_sec == 10.0


def test_subject_config_from_yaml():
    """Test that SubjectConfig can be loaded from YAML file."""
    config_path = Path("configs/test_unified.yml")
    
    if not config_path.exists():
        pytest.skip(f"Test config file not found: {config_path}")
    
    config = SubjectConfig.from_yaml(config_path)
    
    # Verify subject info
    assert config.subject.id == "002"
    assert config.subject.session == "001"
    assert config.subject.task == "fingertapping"
    
    # Verify modality flags
    assert config.modalities.eeg_enabled is True
    assert config.modalities.fnirs_enabled is True
    
    # Verify report config
    assert config.report.qa_only is False
    
    # Verify EEG channels of interest
    assert "C3" in config.eeg_channels_of_interest
    assert "C4" in config.eeg_channels_of_interest
    
    # Verify trials config
    assert config.trials.count_per_condition == 10
    assert config.trials.task_duration_sec == 15.0
    assert config.trials.rest_duration_sec == 15.0


def test_subject_config_immutability():
    """Test that SubjectConfig components are immutable."""
    config_path = Path("configs/test_unified.yml")
    
    if not config_path.exists():
        pytest.skip(f"Test config file not found: {config_path}")
    
    config = SubjectConfig.from_yaml(config_path)
    
    # SubjectInfo is frozen
    with pytest.raises(AttributeError):
        config.subject.id = "999"
    
    # ModalityConfig is frozen
    with pytest.raises(AttributeError):
        config.modalities.eeg_enabled = False
    
    # ReportConfig is frozen
    with pytest.raises(AttributeError):
        config.report.qa_only = True
    
    # TrialsConfig is frozen
    with pytest.raises(AttributeError):
        config.trials.count_per_condition = 20


def test_eeg_channels_of_interest_default():
    """Test that eeg_channels_of_interest has correct default."""
    config_path = Path("configs/test_unified.yml")
    
    if not config_path.exists():
        pytest.skip(f"Test config file not found: {config_path}")
    
    config = SubjectConfig.from_yaml(config_path)
    
    # Should have at least C3 and C4
    assert len(config.eeg_channels_of_interest) >= 2
    assert "C3" in config.eeg_channels_of_interest
    assert "C4" in config.eeg_channels_of_interest
