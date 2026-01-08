# Diagnóstico de Calidad - sub-002

## Resumen Ejecutivo

**Conclusión**: Los datos de sub-002 contienen información neurofisiológica válida, pero el pipeline de preprocesamiento está eliminando la señal de interés.

## Hallazgos Clave

### 1. Datos Crudos EEG - ✅ VÁLIDOS

**Análisis directo de XDF (sin preprocesamiento agresivo):**
- Alpha ERD: **-28.1%** ✅ (esperado: -20% a -50%)
- Beta ERD: **-26.5%** ✅ (esperado: -20% a -40%)
- 21 epochs válidos de 22 eventos

**Conclusión**: La señal ERD está presente y es robusta en los datos crudos.

### 2. Pipeline de Validación - ❌ PROBLEMA

**Resultados del pipeline:**
- Alpha ERD: **-0.03%** ❌ (100x más débil que en datos crudos)
- Beta ERD: **-0.09%** ❌ (300x más débil que en datos crudos)

**Causa raíz**: El preprocesamiento está eliminando la señal:
1. **Escalado correcto**: 5e-10 (ADC units → Volts)
2. **Bad channel detection**: Ahora funciona (0 bad channels vs 28 antes)
3. **ICA demasiado agresivo**: Elimina 5 componentes, posiblemente incluyendo señal neural

### 3. Cobertura Temporal

**EEG**:
- Duración: 660s (~11 min)
- Eventos cubiertos: 22/36 (61%)
- LEFT: 7, RIGHT: 7, NOTHING: 8

**fNIRS**:
- Duración: 1111s (~18.5 min)
- Eventos cubiertos: 36/36 (100%)

**Problema**: El EEG se detuvo prematuramente, perdiendo 14 eventos.

### 4. Datos fNIRS - ⚠️ CALIDAD BAJA

**Estadísticas**:
- Mean intensity: 0.024 (muy bajo)
- CV: 0.6-1% (razonable)
- Cambio hemodinámico: 0.31% (esperado: 1-5%)
- SCI: 0.556 (moderado)
- Bad channels: 26/42 (62%)

**Conclusión**: Los datos fNIRS tienen calidad insuficiente para detectar HRF robusta.

## Problemas Identificados

### P1: ICA Elimina Señal Neural

**Evidencia**:
- Datos crudos: ERD -28%
- Post-ICA: ERD -0.03%
- ICA elimina 5/5 componentes

**Solución propuesta**:
1. Reducir número de componentes ICA eliminados
2. Usar criterios más estrictos para identificar artefactos
3. Considerar no aplicar ICA si la calidad es buena

### P2: Grabación EEG Incompleta

**Evidencia**:
- EEG: 660s
- fNIRS: 1111s
- 14 eventos perdidos

**Solución**: Verificar configuración de hardware para futuras grabaciones.

### P3: Calidad fNIRS Insuficiente

**Evidencia**:
- Cambio hemodinámico: 0.31% (10x más débil que esperado)
- 26/42 canales malos
- SCI moderado (0.556)

**Posibles causas**:
1. Mal contacto de optodos
2. Cabello interfiriendo con señal
3. Calibración incorrecta del dispositivo

## Recomendaciones

### Inmediatas (para validar pipeline)

1. **Modificar ICA**:
   - Usar `n_components=15` en lugar de `0.99`
   - Aplicar criterios más estrictos para EOG/EMG
   - Considerar skip ICA si `len(bad_channels) < 5`

2. **Validar con datos sintéticos**:
   - Crear datos EEG simulados con ERD conocido
   - Verificar que el pipeline preserva la señal

### Para futuras adquisiciones

1. **EEG**:
   - Verificar que la grabación cubra toda la sesión
   - Mejorar contacto de electrodos (impedancias <5kΩ)

2. **fNIRS**:
   - Mejorar contacto de optodos
   - Considerar rasurar área de medición si hay cabello denso
   - Verificar calibración del dispositivo

3. **Sincronización**:
   - Iniciar todas las grabaciones simultáneamente
   - Verificar que los markers se envían a todos los sistemas

## Archivos de Diagnóstico

- `outputs/sub-002_diagnostic.png` - Diagnóstico general
- `outputs/sub-002_eeg_quality.png` - Análisis EEG detallado
- `outputs/sub-002_fnirs_diagnostic.png` - Análisis fNIRS detallado
- `scripts/check_eeg_quality_detail.py` - Script de análisis EEG
- `scripts/diagnose_fnirs_data.py` - Script de análisis fNIRS

## Conclusión Final

**El pipeline técnicamente funciona** (9/9 stages completados sin errores), pero **los parámetros de preprocesamiento son demasiado agresivos** para estos datos.

**Acción requerida**: Ajustar parámetros de ICA para preservar señal neural mientras se eliminan artefactos.
