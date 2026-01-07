"""
Configuración del pipeline de análisis fNIRS.

Este módulo define las configuraciones para el procesamiento de datos fNIRS
usando dataclasses inmutables (frozen) para prevenir modificaciones accidentales.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Configuración para el preprocesamiento de señales fNIRS.

    Attributes:
        sampling_rate_hz: Frecuencia de muestreo en Hz
        lowpass_cutoff_hz: Frecuencia de corte del filtro pasa-bajos
        highpass_cutoff_hz: Frecuencia de corte del filtro pasa-altos
        motion_correction_method: Método de corrección de movimiento
        short_channel_regression: Si usar regresión de canales cortos
        sci_threshold: Umbral de Scalp Coupling Index para rechazo de canales
    """

    sampling_rate_hz: float = 10.0
    lowpass_cutoff_hz: float = 0.5
    highpass_cutoff_hz: float = 0.01
    motion_correction_method: Literal["spline", "wavelet", "tddr"] = "spline"
    short_channel_regression: bool = True
    sci_threshold: float = 0.5


@dataclass(frozen=True)
class HemodynamicConfig:
    """
    Configuración para conversión de intensidad óptica a hemoglobina.

    Attributes:
        wavelengths_nm: Longitudes de onda usadas (tuple de 2 valores)
        dpf: Differential Pathlength Factor para cada longitud de onda
        partial_volume_correction: Si aplicar corrección de volumen parcial
    """

    wavelengths_nm: tuple[float, float] = (760.0, 850.0)
    dpf: tuple[float, float] = (6.0, 6.0)
    partial_volume_correction: bool = False


@dataclass(frozen=True)
class AnalysisConfig:
    """
    Configuración para análisis estadístico de nivel individual.

    Attributes:
        baseline_window_sec: Ventana de baseline pre-estímulo (segundos)
        response_window_sec: Ventana de respuesta post-estímulo (segundos)
        hrf_model: Modelo de función de respuesta hemodinámica
        random_seed: Semilla para reproducibilidad
    """

    baseline_window_sec: tuple[float, float] = (-5.0, 0.0)
    response_window_sec: tuple[float, float] = (0.0, 20.0)
    hrf_model: Literal["canonical", "gamma", "spm"] = "canonical"
    random_seed: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuración completa del pipeline de análisis.

    Attributes:
        preprocessing: Configuración de preprocesamiento
        hemodynamic: Configuración de conversión hemodinámica
        analysis: Configuración de análisis estadístico
        data_root: Ruta raíz de los datos
        output_root: Ruta raíz de los outputs
    """

    preprocessing: PreprocessingConfig
    hemodynamic: HemodynamicConfig
    analysis: AnalysisConfig
    data_root: Path
    output_root: Path

    @classmethod
    def default(cls, data_root: Path, output_root: Path) -> "PipelineConfig":
        """
        Crea una configuración con valores por defecto.

        Args:
            data_root: Ruta raíz de los datos
            output_root: Ruta raíz de los outputs

        Returns:
            PipelineConfig con valores por defecto
        """
        return cls(
            preprocessing=PreprocessingConfig(),
            hemodynamic=HemodynamicConfig(),
            analysis=AnalysisConfig(),
            data_root=data_root,
            output_root=output_root,
        )
