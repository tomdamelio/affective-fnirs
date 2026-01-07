"""
affective_fnirs: Análisis de datos fNIRS para neurociencia cognitiva y afectiva.

Este paquete proporciona herramientas para el procesamiento y análisis de datos
de espectroscopía funcional de infrarrojo cercano (fNIRS) en experimentos de
neurociencia cognitiva y afectiva.
"""

from affective_fnirs.config import (
    AnalysisConfig,
    HemodynamicConfig,
    PipelineConfig,
    PreprocessingConfig,
)

__version__ = "0.1.0"
__author__ = "Sebastian et al."

# Public API
__all__ = [
    "PipelineConfig",
    "PreprocessingConfig",
    "HemodynamicConfig",
    "AnalysisConfig",
]
