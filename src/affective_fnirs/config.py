"""
Pipeline Configuration for Multimodal EEG + fNIRS Validation.

This module defines comprehensive configuration dataclasses for the validation
pipeline processing simultaneous EEG + fNIRS recordings during finger tapping tasks.

All configurations use frozen dataclasses for immutability and include
YAML serialization/deserialization with validation.

Requirements: 10.1, 10.4
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal
import yaml


@dataclass(frozen=True)
class FilterConfig:
    """
    Filter configuration for EEG and fNIRS signal processing.

    Attributes:
        eeg_bandpass_low_hz: EEG high-pass cutoff frequency (Hz)
        eeg_bandpass_high_hz: EEG low-pass cutoff frequency (Hz)
        fnirs_bandpass_low_hz: fNIRS high-pass cutoff for hemodynamic signals (Hz)
        fnirs_bandpass_high_hz: fNIRS low-pass cutoff for hemodynamic signals (Hz)
        cardiac_band_low_hz: Cardiac band low frequency for SCI calculation (Hz)
        cardiac_band_high_hz: Cardiac band high frequency for SCI calculation (Hz)
    """

    eeg_bandpass_low_hz: float = 1.0
    eeg_bandpass_high_hz: float = 40.0
    fnirs_bandpass_low_hz: float = 0.01
    fnirs_bandpass_high_hz: float = 0.5
    cardiac_band_low_hz: float = 0.5
    cardiac_band_high_hz: float = 2.5

    def __post_init__(self) -> None:
        """Validate filter parameters."""
        if self.eeg_bandpass_low_hz >= self.eeg_bandpass_high_hz:
            raise ValueError(
                f"EEG bandpass low ({self.eeg_bandpass_low_hz}) must be < high "
                f"({self.eeg_bandpass_high_hz})"
            )
        if self.fnirs_bandpass_low_hz >= self.fnirs_bandpass_high_hz:
            raise ValueError(
                f"fNIRS bandpass low ({self.fnirs_bandpass_low_hz}) must be < high "
                f"({self.fnirs_bandpass_high_hz})"
            )
        if self.cardiac_band_low_hz >= self.cardiac_band_high_hz:
            raise ValueError(
                f"Cardiac band low ({self.cardiac_band_low_hz}) must be < high "
                f"({self.cardiac_band_high_hz})"
            )


@dataclass(frozen=True)
class QualityThresholds:
    """
    Quality assessment thresholds for fNIRS channel evaluation.

    Based on PHOEBE framework (Pollonini et al., 2016) and NIRSplot guidelines.

    Attributes:
        sci_threshold: Minimum Scalp Coupling Index (0-1). Default 0.75-0.80.
        cv_threshold_percent: Maximum Coefficient of Variation (%). Default 10-15%.
        saturation_percent: Maximum allowed saturation percentage. Default 5%.
        psp_threshold: Minimum Peak Spectral Power for cardiac detection. Default 0.1.
        short_channel_distance_mm: Maximum distance for short channel classification (mm).
    """

    sci_threshold: float = 0.75
    cv_threshold_percent: float = 15.0
    saturation_percent: float = 5.0
    psp_threshold: float = 0.1
    short_channel_distance_mm: float = 15.0

    def __post_init__(self) -> None:
        """Validate quality thresholds."""
        if not 0.0 <= self.sci_threshold <= 1.0:
            raise ValueError(
                f"SCI threshold must be in [0, 1], got {self.sci_threshold}"
            )
        if self.cv_threshold_percent <= 0:
            raise ValueError(
                f"CV threshold must be positive, got {self.cv_threshold_percent}"
            )
        if not 0.0 <= self.saturation_percent <= 100.0:
            raise ValueError(
                f"Saturation percent must be in [0, 100], got {self.saturation_percent}"
            )
        if self.psp_threshold < 0:
            raise ValueError(
                f"PSP threshold must be non-negative, got {self.psp_threshold}"
            )
        if self.short_channel_distance_mm <= 0:
            raise ValueError(
                f"Short channel distance must be positive, got "
                f"{self.short_channel_distance_mm}"
            )


@dataclass(frozen=True)
class EpochConfig:
    """
    Epoch extraction configuration for EEG and fNIRS.

    Attributes:
        eeg_tmin_sec: EEG epoch start time relative to event (seconds)
        eeg_tmax_sec: EEG epoch end time relative to event (seconds)
        fnirs_tmin_sec: fNIRS epoch start time relative to event (seconds)
        fnirs_tmax_sec: fNIRS epoch end time relative to event (seconds)
        baseline_tmin_sec: Baseline window start for correction (seconds)
        baseline_tmax_sec: Baseline window end for correction (seconds)
    """

    eeg_tmin_sec: float = -3.0
    eeg_tmax_sec: float = 15.0
    fnirs_tmin_sec: float = -3.0
    fnirs_tmax_sec: float = 15.0
    baseline_tmin_sec: float = -3.0
    baseline_tmax_sec: float = -1.0

    def __post_init__(self) -> None:
        """Validate epoch parameters."""
        if self.eeg_tmin_sec >= self.eeg_tmax_sec:
            raise ValueError(
                f"EEG tmin ({self.eeg_tmin_sec}) must be < tmax ({self.eeg_tmax_sec})"
            )
        if self.fnirs_tmin_sec >= self.fnirs_tmax_sec:
            raise ValueError(
                f"fNIRS tmin ({self.fnirs_tmin_sec}) must be < tmax "
                f"({self.fnirs_tmax_sec})"
            )
        if self.baseline_tmin_sec >= self.baseline_tmax_sec:
            raise ValueError(
                f"Baseline tmin ({self.baseline_tmin_sec}) must be < tmax "
                f"({self.baseline_tmax_sec})"
            )


@dataclass(frozen=True)
class AnalysisConfig:
    """
    Analysis configuration for EEG ERD/ERS and fNIRS HRF detection.

    Attributes:
        alpha_band_low_hz: Alpha/Mu band lower frequency (Hz)
        alpha_band_high_hz: Alpha/Mu band upper frequency (Hz)
        beta_band_low_hz: Beta band lower frequency (Hz)
        beta_band_high_hz: Beta band upper frequency (Hz)
        task_window_start_sec: Task analysis window start (seconds post-stimulus)
        task_window_end_sec: Task analysis window end (seconds post-stimulus)
        baseline_window_start_sec: Baseline window start for comparison (seconds)
        baseline_window_end_sec: Baseline window end for comparison (seconds)
        beta_rebound_window_start_sec: Beta rebound window start (seconds post-task)
        beta_rebound_window_end_sec: Beta rebound window end (seconds post-task)
        hrf_onset_window_start_sec: Expected HRF onset window start (seconds)
        hrf_onset_window_end_sec: Expected HRF onset window end (seconds)
        hrf_peak_window_start_sec: Expected HRF peak window start (seconds)
        hrf_peak_window_end_sec: Expected HRF peak window end (seconds)
        dpf: Differential Pathlength Factor for Beer-Lambert conversion
    """

    alpha_band_low_hz: float = 8.0
    alpha_band_high_hz: float = 13.0
    beta_band_low_hz: float = 13.0
    beta_band_high_hz: float = 30.0
    task_window_start_sec: float = 1.0
    task_window_end_sec: float = 14.0
    baseline_window_start_sec: float = -5.0
    baseline_window_end_sec: float = -1.0
    beta_rebound_window_start_sec: float = 15.0
    beta_rebound_window_end_sec: float = 20.0
    hrf_onset_window_start_sec: float = 2.0
    hrf_onset_window_end_sec: float = 3.0
    hrf_peak_window_start_sec: float = 4.0
    hrf_peak_window_end_sec: float = 8.0
    dpf: float = 6.0

    def __post_init__(self) -> None:
        """Validate analysis parameters."""
        if self.alpha_band_low_hz >= self.alpha_band_high_hz:
            raise ValueError(
                f"Alpha band low ({self.alpha_band_low_hz}) must be < high "
                f"({self.alpha_band_high_hz})"
            )
        if self.beta_band_low_hz >= self.beta_band_high_hz:
            raise ValueError(
                f"Beta band low ({self.beta_band_low_hz}) must be < high "
                f"({self.beta_band_high_hz})"
            )
        if self.task_window_start_sec >= self.task_window_end_sec:
            raise ValueError(
                f"Task window start ({self.task_window_start_sec}) must be < end "
                f"({self.task_window_end_sec})"
            )
        if self.beta_rebound_window_start_sec >= self.beta_rebound_window_end_sec:
            raise ValueError(
                f"Beta rebound window start ({self.beta_rebound_window_start_sec}) must be < end "
                f"({self.beta_rebound_window_end_sec})"
            )
        if self.dpf <= 0:
            raise ValueError(f"DPF must be positive, got {self.dpf}")


@dataclass(frozen=True)
class ICAConfig:
    """
    ICA configuration for EEG artifact removal.

    Attributes:
        enabled: Whether to apply ICA artifact removal (default: True)
        n_components: Number of ICA components (float for variance ratio, int for count)
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations for ICA fitting
        eog_threshold: Correlation threshold for EOG component detection
        emg_threshold: High-frequency power ratio threshold for EMG detection
        min_components: Minimum number of components to use
        max_bad_channels_for_skip: Skip ICA if bad channels <= this (default: 3)
    """

    enabled: bool = True
    n_components: int | float = 0.99  # Use 99% variance explained for motor tasks
    random_state: int = 42
    max_iter: int = 1000
    eog_threshold: float = 0.9
    emg_threshold: float = 2.5
    min_components: int = 15
    max_bad_channels_for_skip: int = 3

    def __post_init__(self) -> None:
        """Validate ICA parameters."""
        if isinstance(self.n_components, float) and not 0.0 < self.n_components <= 1.0:
            raise ValueError(
                f"n_components as variance ratio must be in (0, 1], "
                f"got {self.n_components}"
            )
        if self.random_state < 0:
            raise ValueError(
                f"random_state must be non-negative, got {self.random_state}"
            )
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
        if not 0.0 <= self.eog_threshold <= 1.0:
            raise ValueError(
                f"eog_threshold must be in [0, 1], got {self.eog_threshold}"
            )
        if self.emg_threshold <= 0:
            raise ValueError(
                f"emg_threshold must be positive, got {self.emg_threshold}"
            )
        if self.min_components <= 0:
            raise ValueError(
                f"min_components must be positive, got {self.min_components}"
            )


@dataclass(frozen=True)
class MotionCorrectionConfig:
    """
    Motion artifact correction configuration for fNIRS.

    Attributes:
        method: Motion correction method ('tddr', 'wavelet', 'none')
        wavelet_name: Wavelet type for wavelet-based correction
        wavelet_threshold: Threshold for wavelet coefficient rejection
    """

    method: Literal["tddr", "wavelet", "none"] = "tddr"
    wavelet_name: str = "db4"
    wavelet_threshold: float = 0.1

    def __post_init__(self) -> None:
        """Validate motion correction parameters."""
        valid_methods = {"tddr", "wavelet", "none"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{self.method}'"
            )


# =============================================================================
# EEG Preprocessing Configuration
# =============================================================================


# Valid EEG channel names from the 10-20 system (commonly used in motor studies)
VALID_EEG_CHANNELS = {
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
    "FC1", "FC2", "FC5", "FC6",
    "CP1", "CP2", "CP5", "CP6",
    "AF3", "AF4", "AF7", "AF8",
    "F1", "F2", "F5", "F6",
    "FT7", "FT8", "FT9", "FT10",
    "C1", "C2", "C5", "C6",
    "TP7", "TP8", "TP9", "TP10",
    "P1", "P2", "P5", "P6",
    "PO3", "PO4", "PO7", "PO8",
    "POz", "Fpz", "CPz", "FCz",
    "M1", "M2",  # Mastoids
    "A1", "A2",  # Earlobes
}


@dataclass(frozen=True)
class EEGPreprocessingConfig:
    """
    EEG preprocessing options for subject-specific configuration.

    Allows control over ICA, reference channel, and CAR application
    based on subject's data quality and channel count.

    Attributes:
        reference_channel: Initial reference channel (default: "Cz")
            - Standard motor cortex reference for ERD/ERS studies
            - Can be changed to other channels if Cz is noisy
        apply_car: Whether to apply Common Average Reference (default: False)
            - Set to True for dense arrays with many channels
            - Set to False to keep original reference
        ica_enabled: Whether to apply ICA artifact removal (default: False)
            - Set to True for subjects with sufficient channels (>20)
            - Set to False for subjects with few channels or clean data

    Requirements: 2.2, 3.1, 8.2, 8.3, 8.4
    """

    reference_channel: str = "Cz"
    apply_car: bool = False
    ica_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate EEG preprocessing parameters."""
        if not self.reference_channel or not self.reference_channel.strip():
            raise ValueError("reference_channel cannot be empty")
        if self.reference_channel not in VALID_EEG_CHANNELS:
            raise ValueError(
                f"reference_channel '{self.reference_channel}' is not a valid "
                f"EEG channel name. Valid channels include: Cz, C3, C4, Fz, etc."
            )


# =============================================================================
# Subject-Specific Configuration Classes (Unified Analysis Pipeline)
# =============================================================================


@dataclass(frozen=True)
class SubjectInfo:
    """
    Subject identification for BIDS compliance.

    Attributes:
        id: Subject identifier without 'sub-' prefix (e.g., '010')
        session: Session identifier without 'ses-' prefix (e.g., '001')
        task: Task name (e.g., 'fingertapping')
    """

    id: str
    session: str
    task: str

    def __post_init__(self) -> None:
        """Validate subject information fields."""
        if not self.id or not self.id.strip():
            raise ValueError("Subject id cannot be empty")
        if not self.session or not self.session.strip():
            raise ValueError("Session cannot be empty")
        if not self.task or not self.task.strip():
            raise ValueError("Task cannot be empty")


@dataclass(frozen=True)
class ModalityConfig:
    """
    Modality enable/disable flags for conditional processing.

    Attributes:
        eeg_enabled: Whether to process EEG data (default: True)
        fnirs_enabled: Whether to process fNIRS data (default: True)
    """

    eeg_enabled: bool = True
    fnirs_enabled: bool = True


@dataclass(frozen=True)
class ReportConfig:
    """
    Report generation options.

    Attributes:
        qa_only: If True, generate only QA report without full analysis (default: False)
    """

    qa_only: bool = False


@dataclass(frozen=True)
class TrialsConfig:
    """
    Trial structure information for the experiment.

    Attributes:
        count_per_condition: Number of trials per condition (default: 10)
        task_duration_sec: Duration of task period in seconds (default: 10.0)
        rest_duration_sec: Duration of rest period in seconds (default: 10.0)
    """

    count_per_condition: int = 10
    task_duration_sec: float = 10.0
    rest_duration_sec: float = 10.0

    def __post_init__(self) -> None:
        """Validate trial configuration parameters."""
        if self.count_per_condition <= 0:
            raise ValueError(
                f"count_per_condition must be positive, got {self.count_per_condition}"
            )
        if self.task_duration_sec <= 0:
            raise ValueError(
                f"task_duration_sec must be positive, got {self.task_duration_sec}"
            )
        if self.rest_duration_sec <= 0:
            raise ValueError(
                f"rest_duration_sec must be positive, got {self.rest_duration_sec}"
            )


@dataclass
class SubjectConfig:
    """
    Complete subject-specific configuration for the unified analysis pipeline.

    Composes subject identification, modality flags, report options, trial info,
    and all existing PipelineConfig sections. Supports YAML loading with defaults.

    Attributes:
        subject: Subject identification (id, session, task)
        modalities: Modality enable/disable flags
        report: Report generation options
        eeg_channels_of_interest: List of EEG channels for ERD/ERS analysis
        eeg_preprocessing: EEG preprocessing options (reference, CAR, ICA)
        trials: Trial structure information
        filters: Filter configuration for signal processing
        quality: Quality assessment thresholds
        epochs: Epoch extraction parameters
        analysis: Analysis parameters for ERD/ERS and HRF
        ica: ICA configuration for EEG artifact removal
        motion_correction: Motion correction configuration for fNIRS
        data_root: Root path for input data (read-only)
        output_root: Root path for derivative outputs
        random_seed: Global random seed for reproducibility

    Requirements: 2.6, 2.7
    """

    subject: SubjectInfo
    modalities: ModalityConfig = field(default_factory=ModalityConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    eeg_channels_of_interest: list[str] = field(
        default_factory=lambda: ["C3", "C4"]
    )
    eeg_preprocessing: EEGPreprocessingConfig = field(
        default_factory=EEGPreprocessingConfig
    )
    trials: TrialsConfig = field(default_factory=TrialsConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    epochs: EpochConfig = field(default_factory=EpochConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ica: ICAConfig = field(default_factory=ICAConfig)
    motion_correction: MotionCorrectionConfig = field(
        default_factory=MotionCorrectionConfig
    )
    data_root: Path = field(default_factory=lambda: Path("data/raw"))
    output_root: Path = field(
        default_factory=lambda: Path("data/derivatives/validation-pipeline")
    )
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Convert string paths to Path objects and validate."""
        if isinstance(self.data_root, str):
            object.__setattr__(self, "data_root", Path(self.data_root))
        if isinstance(self.output_root, str):
            object.__setattr__(self, "output_root", Path(self.output_root))
        if self.random_seed < 0:
            raise ValueError(
                f"random_seed must be non-negative, got {self.random_seed}"
            )
        # Ensure eeg_channels_of_interest defaults if empty
        if not self.eeg_channels_of_interest:
            object.__setattr__(self, "eeg_channels_of_interest", ["C3", "C4"])

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of the configuration.
        """
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, dict):
                result[key] = value
            else:
                result[key] = value
        return result

    def to_yaml(self, file_path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            file_path: Path to save the YAML configuration.
        """
        config_dict = self.to_dict()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                config_dict,
                yaml_file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SubjectConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values.

        Returns:
            SubjectConfig instance.

        Raises:
            ValueError: If required keys are missing or values are invalid.
        """
        # Required: subject section
        subject_data = config_dict.get("subject")
        if subject_data is None:
            raise ValueError("Missing required 'subject' section in configuration")
        subject = SubjectInfo(**subject_data)

        # Optional sections with defaults
        modalities = ModalityConfig(**config_dict.get("modalities", {}))
        report = ReportConfig(**config_dict.get("report", {}))
        eeg_channels = config_dict.get("eeg_channels_of_interest", ["C3", "C4"])
        eeg_preprocessing = EEGPreprocessingConfig(
            **config_dict.get("eeg_preprocessing", {})
        )
        trials = TrialsConfig(**config_dict.get("trials", {}))

        # Inherited PipelineConfig sections
        filters = FilterConfig(**config_dict.get("filters", {}))
        quality = QualityThresholds(**config_dict.get("quality", {}))
        epochs = EpochConfig(**config_dict.get("epochs", {}))
        analysis = AnalysisConfig(**config_dict.get("analysis", {}))
        ica = ICAConfig(**config_dict.get("ica", {}))
        motion_correction = MotionCorrectionConfig(
            **config_dict.get("motion_correction", {})
        )

        return cls(
            subject=subject,
            modalities=modalities,
            report=report,
            eeg_channels_of_interest=eeg_channels,
            eeg_preprocessing=eeg_preprocessing,
            trials=trials,
            filters=filters,
            quality=quality,
            epochs=epochs,
            analysis=analysis,
            ica=ica,
            motion_correction=motion_correction,
            data_root=Path(config_dict.get("data_root", "data/raw")),
            output_root=Path(
                config_dict.get("output_root", "data/derivatives/validation-pipeline")
            ),
            random_seed=config_dict.get("random_seed", 42),
        )

    @classmethod
    def from_yaml(cls, file_path: Path) -> "SubjectConfig":
        """
        Load configuration from YAML file.

        Args:
            file_path: Path to the YAML configuration file.

        Returns:
            SubjectConfig instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML content is invalid.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as yaml_file:
            config_dict = yaml.safe_load(yaml_file)

        if config_dict is None:
            raise ValueError(f"Empty or invalid YAML file: {file_path}")

        return cls.from_dict(config_dict)

    def validate_paths(self) -> None:
        """
        Validate that data paths exist and output paths can be created.

        Raises:
            FileNotFoundError: If data_root does not exist.
        """
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root directory does not exist: {self.data_root}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration for multimodal EEG + fNIRS validation.

    This is the main configuration class that aggregates all sub-configurations.
    Supports YAML serialization/deserialization for reproducibility.

    Attributes:
        filters: Filter configuration for signal processing
        quality: Quality assessment thresholds
        epochs: Epoch extraction parameters
        analysis: Analysis parameters for ERD/ERS and HRF
        ica: ICA configuration for EEG artifact removal
        motion_correction: Motion correction configuration for fNIRS
        data_root: Root path for input data (read-only)
        output_root: Root path for derivative outputs
        random_seed: Global random seed for reproducibility
    """

    filters: FilterConfig = field(default_factory=FilterConfig)
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    epochs: EpochConfig = field(default_factory=EpochConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ica: ICAConfig = field(default_factory=ICAConfig)
    motion_correction: MotionCorrectionConfig = field(
        default_factory=MotionCorrectionConfig
    )
    data_root: Path = field(default_factory=lambda: Path("data/raw"))
    output_root: Path = field(
        default_factory=lambda: Path("data/derivatives/validation-pipeline")
    )
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Convert string paths to Path objects and validate."""
        if isinstance(self.data_root, str):
            object.__setattr__(self, "data_root", Path(self.data_root))
        if isinstance(self.output_root, str):
            object.__setattr__(self, "output_root", Path(self.output_root))
        if self.random_seed < 0:
            raise ValueError(
                f"random_seed must be non-negative, got {self.random_seed}"
            )

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of the configuration.
        """
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, dict):
                result[key] = value
            else:
                result[key] = value
        return result

    def to_yaml(self, file_path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            file_path: Path to save the YAML configuration.
        """
        config_dict = self.to_dict()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                config_dict,
                yaml_file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values.

        Returns:
            PipelineConfig instance.

        Raises:
            ValueError: If required keys are missing or values are invalid.
        """
        filters = FilterConfig(**config_dict.get("filters", {}))
        quality = QualityThresholds(**config_dict.get("quality", {}))
        epochs = EpochConfig(**config_dict.get("epochs", {}))
        analysis = AnalysisConfig(**config_dict.get("analysis", {}))
        ica = ICAConfig(**config_dict.get("ica", {}))
        motion_correction = MotionCorrectionConfig(
            **config_dict.get("motion_correction", {})
        )

        return cls(
            filters=filters,
            quality=quality,
            epochs=epochs,
            analysis=analysis,
            ica=ica,
            motion_correction=motion_correction,
            data_root=Path(config_dict.get("data_root", "data/raw")),
            output_root=Path(
                config_dict.get("output_root", "data/derivatives/validation-pipeline")
            ),
            random_seed=config_dict.get("random_seed", 42),
        )

    @classmethod
    def from_yaml(cls, file_path: Path) -> "PipelineConfig":
        """
        Load configuration from YAML file.

        Args:
            file_path: Path to the YAML configuration file.

        Returns:
            PipelineConfig instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML content is invalid.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as yaml_file:
            config_dict = yaml.safe_load(yaml_file)

        if config_dict is None:
            raise ValueError(f"Empty or invalid YAML file: {file_path}")

        return cls.from_dict(config_dict)

    @classmethod
    def default(cls) -> "PipelineConfig":
        """
        Create a configuration with all default values.

        Returns:
            PipelineConfig with default parameters.
        """
        return cls()

    def validate_paths(self) -> None:
        """
        Validate that data paths exist and output paths can be created.

        Raises:
            FileNotFoundError: If data_root does not exist.
        """
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root directory does not exist: {self.data_root}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
