"""
Tests for run_analysis.py CLI argument parsing.

Validates command-line interface behavior.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_analysis import parse_args


def test_parse_args_config_required():
    """Test that --config argument is required."""
    with patch("sys.argv", ["run_analysis.py"]):
        with pytest.raises(SystemExit):
            parse_args()


def test_parse_args_config_only():
    """Test parsing with only --config argument."""
    with patch("sys.argv", ["run_analysis.py", "--config", "configs/test.yml"]):
        args = parse_args()
        
        assert args.config == Path("configs/test.yml")
        assert args.eeg is None
        assert args.fnirs is None
        assert args.qa_only is False


def test_parse_args_eeg_override():
    """Test parsing with --eeg override."""
    with patch("sys.argv", ["run_analysis.py", "--config", "test.yml", "--eeg", "false"]):
        args = parse_args()
        
        assert args.eeg == "false"


def test_parse_args_fnirs_override():
    """Test parsing with --fnirs override."""
    with patch("sys.argv", ["run_analysis.py", "--config", "test.yml", "--fnirs", "true"]):
        args = parse_args()
        
        assert args.fnirs == "true"


def test_parse_args_qa_only():
    """Test parsing with --qa-only flag."""
    with patch("sys.argv", ["run_analysis.py", "--config", "test.yml", "--qa-only"]):
        args = parse_args()
        
        assert args.qa_only is True


def test_parse_args_all_flags():
    """Test parsing with all flags combined."""
    with patch(
        "sys.argv",
        [
            "run_analysis.py",
            "--config",
            "test.yml",
            "--eeg",
            "false",
            "--fnirs",
            "true",
            "--qa-only",
        ],
    ):
        args = parse_args()
        
        assert args.config == Path("test.yml")
        assert args.eeg == "false"
        assert args.fnirs == "true"
        assert args.qa_only is True


def test_parse_args_short_config():
    """Test parsing with -c short form."""
    with patch("sys.argv", ["run_analysis.py", "-c", "test.yml"]):
        args = parse_args()
        
        assert args.config == Path("test.yml")
