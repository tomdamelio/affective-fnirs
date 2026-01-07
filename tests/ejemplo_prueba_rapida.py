"""
Script de prueba suelto - Ejemplo de cómo usar la carpeta tests/.

Este es un script simple para probar cosas rápidamente sin la estructura
formal de pytest. Úsalo para debugging, exploración, o prototipos.
"""

from pathlib import Path

from affective_fnirs import PipelineConfig, PreprocessingConfig

# Probar configuración por defecto
print("=" * 80)
print("PRUEBA: Configuración por defecto")
print("=" * 80)

config = PipelineConfig.default(
    data_root=Path("data/raw"), output_root=Path("outputs")
)

print(f"✅ Config creada")
print(f"   Sampling rate: {config.preprocessing.sampling_rate_hz} Hz")
print(f"   Lowpass cutoff: {config.preprocessing.lowpass_cutoff_hz} Hz")
print(f"   Motion correction: {config.preprocessing.motion_correction_method}")
print(f"   Random seed: {config.analysis.random_seed}")

# Probar configuración personalizada
print("\n" + "=" * 80)
print("PRUEBA: Configuración personalizada")
print("=" * 80)

custom_preprocessing = PreprocessingConfig(
    sampling_rate_hz=20.0,
    motion_correction_method="wavelet",
    short_channel_regression=False,
)

print(f"✅ Preprocessing personalizado creado")
print(f"   Sampling rate: {custom_preprocessing.sampling_rate_hz} Hz")
print(f"   Motion correction: {custom_preprocessing.motion_correction_method}")
print(f"   Short channel regression: {custom_preprocessing.short_channel_regression}")

# Probar inmutabilidad
print("\n" + "=" * 80)
print("PRUEBA: Inmutabilidad (debe fallar)")
print("=" * 80)

try:
    config.preprocessing.sampling_rate_hz = 999.0
    print("❌ ERROR: La configuración NO es inmutable!")
except AttributeError:
    print("✅ Configuración es inmutable (frozen dataclass)")

print("\n" + "=" * 80)
print("TODAS LAS PRUEBAS COMPLETADAS")
print("=" * 80)
