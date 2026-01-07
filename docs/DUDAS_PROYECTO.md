# Dudas y Aclaraciones Pendientes - Proyecto fNIRS

**Fecha de creación**: 2026-01-07  
**Responsable de resolver**: Sebastian (Líder de proyecto)

---

## 1. Discrepancia en Nomenclatura de Montaje EEG-fNIRS ✅ RESUELTO

**Prioridad**: 🔴 Alta → ✅ **RESUELTO** (2026-01-07)

**Descripción**:
Existía una inconsistencia entre las posiciones de electrodos EEG definidas en `pilot_finger_tapping.cfg` y las posiciones de detectores fNIRS en `montage_config.json` (que vienen del documento .doc "Piloto EEG-fNIRS: Montaje y protocolo").

**Resolución**:
Sebastian indicó que el montaje real utilizado está documentado en los archivos JSON de cada sujeto. Tras revisar los 4 archivos JSON (sub-001 y sub-002, tanto fNIRS como EEG), se confirma que el montaje es **idéntico y consistente** entre ambos sujetos:

### Montaje fNIRS Real (Cortivision Photon Cap C20):

**Fuentes (8 largas + 2 cortas)**:
- S1_FCC3h, S2_FCC4h (frontocentral medial)
- S3_FCC5h, S4_FCC6h (frontocentral lateral)
- S5_CCP5h, S6_CCP6h (centroparietal lateral)
- S7_CCP3h, S8_CCP4h (centroparietal medial)
- S13_ShortL, S14_ShortR (canales cortos, 8mm)

**Detectores (8 posiciones)**:
- D1_C1, D2_C2 (línea central, sistema 10-5)
- D3_CP3, D4_CP4 (centroparietal)
- D5_C5, D6_C6 (línea lateral)
- D7_FC3, D8_FC4 (frontocentral)

**Configuración de canales**:
- 32 canales largos (16 pares × 2 longitudes de onda: 760nm, 850nm)
- 4 canales cortos (2 pares × 2 longitudes de onda)
- Separación: 30mm (largos), 8mm (cortos)
- Frecuencia de muestreo: 8.12 Hz

### Montaje EEG Real (idéntico en sub-001 y sub-002):

**Canales EEG** (35 canales):
Fp1, Fz, F3, F7, F9, FC5, FC1, **C3**, T7, CP5, CP1, Pz, P3, P7, P9, O1, Oz, O2, P10, P8, P4, CP2, CP6, T8, **C4**, **Cz**, FC2, FC6, F10, F8, F4, Fp2, AUX_1, AUX_2, AUX_3

**Verificación**: ✅ Montaje EEG idéntico en ambos sujetos

**Respuestas a las preguntas originales**:
1. ✅ Los electrodos EEG usan nomenclatura 10-20 estándar (C3, Cz, C4)
2. ✅ Los detectores fNIRS usan nomenclatura 10-5 extendida (C1, C2, C5, C6)
3. ✅ Ambos sistemas coexisten: EEG en posiciones 10-20, fNIRS en posiciones 10-5 intermedias
4. ✅ S13/S14 son los canales cortos (NO S9/S10 del CSV extraído)
5. ✅ El archivo `fnirs_channel_locations.csv` tiene un error de etiquetado (S9/S10 deberían ser S13/S14)
6. ✅ Los archivos JSON en `data/raw/sub-00X/` son la fuente de verdad

**Acción requerida**:
- Actualizar `configs/montage_config.json` para que coincida con los archivos reales de los sujetos
- Corregir `configs/fnirs_channel_locations.csv` (S9→S13, S10→S14)
- Usar los archivos JSON de cada sujeto como referencia canónica para el procesamiento


---

## 2. [Espacio para futuras dudas]

**Prioridad**: 

**Descripción**:

**Preguntas**:

**Impacto**:

---

## 3. [Espacio para futuras dudas]

**Prioridad**: 

**Descripción**:

**Preguntas**:

**Impacto**:

---

## Notas

- Este documento se actualizará conforme surjan nuevas dudas durante el análisis
- Marcar como resueltas (✅) las dudas una vez aclaradas
- Agregar la fecha de resolución y la respuesta obtenida
