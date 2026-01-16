# affective-fnirs

Repositorio de análisis de datos fNIRS para experimentos de neurociencia cognitiva y afectiva.

## Estructura del Proyecto

```
affective-fnirs/
├── src/affective_fnirs/    # Paquete Python con lógica científica
├── tests/                   # Scripts de prueba sueltos
├── configs/                 # Configuraciones de experimentos
├── scripts/                 # Scripts de análisis y utilidades
├── docs/                    # Documentación y ejemplos
├── data/                    # Datos (no versionados)
│   ├── raw/                 # Datos crudos (SOLO LECTURA)
│   └── derivatives/         # Datos procesados
├── outputs/                 # Resultados generados (gitignored)
├── pyproject.toml           # Configuración del paquete
├── environment.yml          # Entorno reproducible (micromamba)
└── README.md
```

## Instalación

### 1. Crear el entorno
```powershell
micromamba create -f environment.yml
```

### 2. Instalar el paquete en modo desarrollo
```powershell
micromamba run -n affective-fnirs pip install -e .
```

### 3. Verificar instalación
```powershell
# Verificar imports
micromamba run -n affective-fnirs python scripts/verify_imports.py


## Uso

### Test Data

El proyecto incluye datos piloto de dos sujetos:
- **`sub-002`** ✅ - Datos completos (EEG + fNIRS + Markers) - **Usar para desarrollo**
- **`sub-001`** ⚠️ - Datos incompletos (solo fNIRS + Markers, sin EEG)

Ver [`docs/TEST_DATA.md`](docs/TEST_DATA.md) para detalles completos sobre los datos de prueba.

### Comandos Básicos

Todos los comandos Python deben ejecutarse usando `micromamba run`:

```powershell
# Ejecutar un script de prueba suelto
micromamba run -n affective-fnirs python tests/mi_script.py
```

### Pipeline de Análisis

El script principal `scripts/run_analysis.py` ejecuta el pipeline completo de análisis multimodal (EEG + fNIRS).

#### Ejecución Completa

```powershell
# Análisis completo (preprocesamiento + análisis)
micromamba run -n affective-fnirs python scripts/run_analysis.py --config configs/sub-010.yml
```

#### Flujo de Trabajo Incremental

El pipeline guarda datos intermedios para permitir iteraciones rápidas:

**1. Primera ejecución (preprocesamiento completo con inspecciones interactivas):**
```powershell
micromamba run -n affective-fnirs python scripts/run_analysis.py --config configs/sub-010.yml
```
- Inspección visual de canales malos (después de filtrado)
- Rechazo de épocas malas (antes de ICA)
- Selección interactiva de componentes ICA
- Guarda: `*_desc-preprocessed_eeg.fif`, `*_desc-cleaned_epo.fif`, `*_ica.fif`

**2. Cargar datos preprocesados Y épocas limpias (saltar TODO el preprocesamiento interactivo):**
```powershell
micromamba run -n affective-fnirs python scripts/run_analysis.py --config configs/sub-010.yml --load-preprocessed
```
- Carga datos preprocesados Y épocas limpias
- Ejecuta solo: TFR → ERD/ERS
- **No requiere interacción del usuario**
- Útil para iterar sobre parámetros de análisis espectral

**3. Cargar épocas limpias (equivalente a --load-preprocessed para EEG):**
```powershell
micromamba run -n affective-fnirs python scripts/run_analysis.py --config configs/sub-010.yml --load-epochs
```
- Mismo comportamiento que `--load-preprocessed` para análisis EEG
- Carga épocas ya limpias y objeto ICA
- Ejecuta solo: TFR → ERD/ERS

#### Archivos Guardados

El pipeline guarda automáticamente:
- `sub-{id}_ses-{session}_task-{task}_desc-preprocessed_eeg.fif` - EEG preprocesado (filtrado, CAR)
- `sub-{id}_ses-{session}_task-{task}_desc-preprocessed_fnirs.fif` - fNIRS preprocesado
- `sub-{id}_ses-{session}_task-{task}_desc-cleaned_epo.fif` - Épocas limpias (post-ICA)
- `sub-{id}_ses-{session}_task-{task}_ica.fif` - Objeto ICA (para referencia)

#### Otros Flags

```powershell
# Solo reporte de calidad (QA)
micromamba run -n affective-fnirs python scripts/run_analysis.py --config configs/sub-010.yml --qa-only

# Deshabilitar modalidades específicas
micromamba run -n affective-fnirs python scripts/run_analysis.py --config configs/sub-010.yml --eeg false
micromamba run -n affective-fnirs python scripts/run_analysis.py --config configs/sub-010.yml --fnirs false
```



## Contenido del Repositorio

### Archivos de Configuración (`configs/`)

**IMPORTANTE**: Los archivos JSON en `data/raw/sub-00X/` son la **referencia canónica** del montaje utilizado en cada sujeto. Los archivos en `configs/` son plantillas de referencia.

Ver `configs/README.md` para detalles completos sobre:
- Montaje fNIRS (36 canales, sistema 10-5)
- Localización espacial de canales
- Configuración EEG (35 canales)

### Documentación (`docs/`)

Ver `docs/README.md` para documentación técnica del formato SNIRF.

### Scripts (`scripts/`)

Scripts de análisis y utilidades. Ejemplo:

```powershell
# Explorar un archivo SNIRF
micromamba run -n affective-fnirs python scripts/example_load_snirf.py data/raw/sub-01/ses-01/nirs/sub-01_ses-01_task-fingertapping_nirs.snirf
```

## Estándares del Proyecto

Este proyecto sigue estrictos estándares de neurociencia computacional:

- **BIDS Compliance**: Todos los datos siguen Brain Imaging Data Structure
- **Reproducibilidad**: Entornos pinneados con micromamba
- **Code Style**: Black + Ruff (sin excepciones)
- **Naming**: Variables neuro-semánticas (no `x`, `data`, `t`)
- **Type Hints**: Obligatorios en todas las funciones públicas
- **Testing**: Pytest con cobertura
- **Git**: Commits atómicos con rationale científico

Ver `.kiro/steering/*.md` para reglas detalladas.

## Dudas y Problemas

- `DUDAS_PROYECTO.md`: Inconsistencias detectadas en el montaje experimental

## Licencia

MIT License - Ver `LICENSE` para detalles.

---

#### `fnirs_channel_locations.csv`
Tabla de localización espacial de canales fNIRS con 36 filas (18 canales × 2 longitudes de onda).

**Columnas:**
- `channel_name`: Identificador del canal (ej: "S1_D1 760")
- `source`: Fuente emisora (S1-S10)
- `detector`: Detector receptor (D1-D8)
- `wavelength_nm`: Longitud de onda (760 o 850 nm)
- `x_mm`, `y_mm`, `z_mm`: Coordenadas espaciales en milímetros
- `coord_frame`: Sistema de referencia (0 = coordenadas del dispositivo)

**Cobertura anatómica:**
- Canales bilaterales sobre corteza motora y sensoriomotora
- Rango X: ±80 mm (lateralidad izquierda/derecha)
- Rango Y: -48 a +15 mm (anterior-posterior)
- Rango Z: 37 a 85 mm (inferior-superior)

#### `montage_config.json`
Configuración de montaje fNIRS del experimento piloto con mapeo anatómico según sistema 10-5 extendido. Define 36 canales (32 largos + 4 cortos).

**Estructura:**
- `channel_idx`: Índice del canal (0-35)
- `source`: Fuente con etiqueta anatómica (ej: "S1_FCC3h")
- `detector`: Detector con etiqueta anatómica (ej: "D1_C1")
- `wavelength`: 760 nm (Hb) o 850 nm (HbO)
- `type`: "Long" (separación estándar ~3cm) o "Short" (separación corta ~8mm)
- `location_label`: Etiqueta descriptiva del canal

**Canales largos (32 canales, hemoglobina cortical):**
- **Fuentes**: S1-S8 (8 fuentes bilaterales)
  - Hemisferio izquierdo: S1_FCC3h, S3_FCC5h, S5_CCP5h, S7_CCP3h
  - Hemisferio derecho: S2_FCC4h, S4_FCC6h, S6_CCP6h, S8_CCP4h
- **Detectores**: D1-D8 (8 detectores bilaterales)
  - Línea media: D1_C1, D2_C2
  - Laterales izquierdos: D3_CP3, D5_C5, D7_FC3
  - Laterales derechos: D4_CP4, D6_C6, D8_FC4
- **Cobertura**: Corteza sensoriomotora bilateral (áreas motoras primarias y premotoras)

**Canales cortos (4 canales, regresión de señal superficial):**
- S13_ShortL → D3_CP3 (hemisferio izquierdo)
- S14_ShortR → D4_CP4 (hemisferio derecho)
- Cada uno mide Hb (760nm) y HbO (850nm)
- Permiten corrección de artefactos sistémicos y extracraneales mediante regresión

#### `pilot_finger_tapping.cfg`
Archivo de configuración para adquisición simultánea EEG (BrainVision).

**Parámetros del dispositivo:**
- Amplificador: AMP0502384MRplus (BrainAmp)
- Canales: 32
- Frecuencia de muestreo: 500 Hz
- Canales auxiliares: Habilitados
- Marcadores: Unsampleados y sampleados en EEG

**Montaje EEG (32 canales):**
- Sistema 10-20 extendido: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz, FC1, FC2, CP1, CP2, FC5, FC6, CP5, CP6, TP9, TP10, Oz
- Canales fisiológicos: ECG, RESP


#### `pysnirf2.log`
Log de ejecución de la librería `pysnirf2` (v0.8.0) para lectura/escritura de archivos SNIRF.

**Entradas:**
- 2025-12-26 16:52:39 - Proceso 13996
- 2025-12-26 16:55:59 - Proceso 23040

Indica que se han realizado operaciones de lectura/escritura de archivos SNIRF en esas sesiones.