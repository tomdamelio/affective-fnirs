# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### Added
- Estructura inicial del proyecto siguiendo estándares BIDS y src-layout
- Configuración de entorno con micromamba (`environment.yml`)
- Configuración de tooling con `pyproject.toml` (black, ruff, pytest)
- Archivos de configuración del experimento piloto finger tapping
- Documentación de montaje fNIRS y EEG
- Separación de tests: `pytest_tests/` para tests formales, `tests/` para scripts sueltos
- Dependencias ampliadas: cedalion, xarray, pyvista, pyxdf, scikit-learn
- Script de verificación de imports (`scripts/verify_imports.py`)
- Documentación de dudas del proyecto (`docs/DUDAS_PROYECTO.md`)

### Changed
- Reorganización de archivos de configuración a carpeta `configs/`
- `tests/` ahora es para scripts de prueba sueltos (no pytest)
- Tests formales con pytest movidos a `pytest_tests/`
- Entorno recreado con numpy 2.2.6 (compatible con cedalion)
- Reemplazado `pysnirf2` por `snirf` (incluido en cedalion, compatible con numpy 2.x)

### Fixed
- **Montaje fNIRS**: Corregido `fnirs_channel_locations.csv` (S9→S13, S10→S14) para coincidir con archivos reales de sujetos
- **Documentación**: Establecido que archivos JSON en `data/raw/sub-00X/` son la referencia canónica del montaje
- **Verificación**: Confirmado que montaje es idéntico entre sub-001 y sub-002
- Versión de pysnirf2 corregida a `>=0.7.0` (0.8.0 no existe en PyPI)

## [0.1.0] - 2026-01-07

### Added
- Inicio del proyecto
- README inicial con descripción de archivos del piloto
