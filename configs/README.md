# Configuraciones del Experimento

Esta carpeta contiene archivos de configuración de **referencia** del experimento piloto de finger tapping con co-registro EEG-fNIRS.

## ⚠️ IMPORTANTE: Fuente de Verdad

**Los archivos JSON en `data/raw/sub-00X/` son la referencia canónica del montaje real utilizado.**

Los archivos en esta carpeta (`configs/`) son plantillas de referencia que deben coincidir con los archivos de cada sujeto. En caso de discrepancia, los archivos en `data/raw/sub-00X/` tienen prioridad.

## Archivos

### `montage_config.json` ✅
Configuración del montaje fNIRS con mapeo anatómico según sistema 10-5 extendido.
- 36 canales totales (32 largos + 4 cortos)
- 2 longitudes de onda: 760 nm (Hb) y 850 nm (HbO)
- Cobertura: Corteza sensoriomotora bilateral
- **Estado**: Verificado contra `data/raw/sub-001/` y `data/raw/sub-002/`

### `fnirs_channel_locations.csv` ✅
Tabla de localización espacial de canales fNIRS extraída del archivo SNIRF.
- 18 canales únicos × 2 longitudes de onda = 36 filas
- Coordenadas 3D en milímetros
- Sistema de referencia: Coordenadas del dispositivo (coord_frame=0)
- **Estado**: Corregido (S9→S13, S10→S14) para coincidir con montaje real

### `pilot_finger_tapping.cfg`
Configuración del amplificador EEG (BrainVision BrainAmp).
- 35 canales EEG (sistema 10-20 extendido + AUX)
- Canales principales: Fp1, Fz, F3, F7, F9, FC5, FC1, C3, T7, CP5, CP1, Pz, P3, P7, P9, O1, Oz, O2, P10, P8, P4, CP2, CP6, T8, C4, Cz, FC2, FC6, F10, F8, F4, Fp2
- Canales auxiliares: AUX_1, AUX_2, AUX_3

## Montaje Verificado

### fNIRS (Cortivision Photon Cap C20)
- **Fuentes**: S1-S8 (largos) + S13-S14 (cortos)
- **Detectores**: D1-D8 con etiquetas anatómicas (D1_C1, D2_C2, D3_CP3, D4_CP4, D5_C5, D6_C6, D7_FC3, D8_FC4)
- **Separación**: 30mm (largos), 8mm (cortos)
- **Frecuencia**: 8.12 Hz

### EEG
- **Sistema**: 10-20 extendido
- **Canales centrales**: C3, Cz, C4 (coexisten con detectores fNIRS en posiciones 10-5 intermedias)

## Procesamiento

Para cargar el montaje correcto en análisis:

```python
import json

# Cargar montaje del sujeto específico (RECOMENDADO)
with open('data/raw/sub-001/sub-001_Tomi_ses-001_task-fingertapping_nirs.json') as f:
    montage = json.load(f)

# O usar la plantilla de referencia
with open('configs/montage_config.json') as f:
    montage_template = json.load(f)
```
