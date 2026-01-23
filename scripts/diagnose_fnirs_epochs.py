#!/usr/bin/env python3
"""
Diagnóstico de datos fNIRS en epochs.

Este script examina los datos fNIRS para entender por qué los gráficos
aparecen vacíos o con valores cercanos a cero.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from affective_fnirs.config import SubjectConfig

# Cargar configuración
config_path = Path("configs/sub-009.yml")
config = SubjectConfig.from_yaml(config_path)

print("=" * 70)
print("DIAGNÓSTICO DE DATOS fNIRS")
print("=" * 70)

# Cargar datos preprocesados
preprocessed_path = Path(f"data/derivatives/validation-pipeline/sub-{config.subject.id}/ses-{config.subject.session}")
fnirs_path = preprocessed_path / f"sub-{config.subject.id}_ses-{config.subject.session}_task-{config.subject.task}_desc-preprocessed_fnirs.fif"

if not fnirs_path.exists():
    print(f"ERROR: No se encontró archivo preprocesado: {fnirs_path}")
    sys.exit(1)

import mne

print(f"\n1. Cargando datos fNIRS preprocesados: {fnirs_path}")
fnirs_raw = mne.io.read_raw_fif(fnirs_path, preload=True)

print(f"\n2. Información del Raw fNIRS:")
print(f"   - Canales: {len(fnirs_raw.ch_names)}")
print(f"   - Duración: {fnirs_raw.times[-1]:.1f} s")
print(f"   - Frecuencia de muestreo: {fnirs_raw.info['sfreq']} Hz")

# Verificar tipos de canales
ch_types = fnirs_raw.get_channel_types()
unique_types = set(ch_types)
print(f"   - Tipos de canales: {unique_types}")

# Contar por tipo
for ch_type in unique_types:
    count = ch_types.count(ch_type)
    print(f"     - {ch_type}: {count} canales")

# Verificar canales HbO y HbR
hbo_channels = [ch for ch in fnirs_raw.ch_names if 'hbo' in ch.lower()]
hbr_channels = [ch for ch in fnirs_raw.ch_names if 'hbr' in ch.lower()]
print(f"\n3. Canales de hemoglobina:")
print(f"   - HbO: {len(hbo_channels)} canales")
print(f"   - HbR: {len(hbr_channels)} canales")

if hbo_channels:
    print(f"   - Nombres HbO: {hbo_channels[:5]}...")
if hbr_channels:
    print(f"   - Nombres HbR: {hbr_channels[:5]}...")

# Verificar datos crudos
print(f"\n4. Estadísticas de datos crudos:")
data = fnirs_raw.get_data()
print(f"   - Shape: {data.shape}")
print(f"   - Min: {data.min():.2e}")
print(f"   - Max: {data.max():.2e}")
print(f"   - Mean: {data.mean():.2e}")
print(f"   - Std: {data.std():.2e}")

# Verificar si hay NaN o Inf
print(f"   - NaN count: {np.isnan(data).sum()}")
print(f"   - Inf count: {np.isinf(data).sum()}")

# Verificar por tipo de canal
if hbo_channels:
    hbo_data = fnirs_raw.get_data(picks=hbo_channels)
    print(f"\n5. Estadísticas HbO:")
    print(f"   - Shape: {hbo_data.shape}")
    print(f"   - Min: {hbo_data.min():.2e}")
    print(f"   - Max: {hbo_data.max():.2e}")
    print(f"   - Mean: {hbo_data.mean():.2e}")
    print(f"   - Std: {hbo_data.std():.2e}")
    
    # Verificar unidades esperadas
    # HbO típicamente está en mol/L (M) o μmol/L (μM)
    # Si está en M, valores típicos son ~1e-6 a 1e-5
    # Si está en μM, valores típicos son ~1 a 10
    if abs(hbo_data.mean()) < 1e-10:
        print("   ⚠️ ADVERTENCIA: Valores muy pequeños - posible problema de escala o datos vacíos")
    elif abs(hbo_data.mean()) < 1e-3:
        print("   ℹ️ Valores en rango típico para mol/L (M)")
    else:
        print("   ℹ️ Valores posiblemente en μmol/L (μM) o unidades arbitrarias")

# Verificar anotaciones/eventos
print(f"\n6. Anotaciones/Eventos:")
annotations = fnirs_raw.annotations
print(f"   - Total anotaciones: {len(annotations)}")
if len(annotations) > 0:
    unique_desc = set(annotations.description)
    print(f"   - Descripciones únicas: {unique_desc}")
    for desc in unique_desc:
        count = list(annotations.description).count(desc)
        print(f"     - '{desc}': {count} eventos")

# Crear epochs para verificar
print(f"\n7. Creando epochs de prueba...")
from affective_fnirs.fnirs_analysis import create_fnirs_epochs

event_mapping = {
    "LEFT": 1,
    "RIGHT": 2,
    "NOTHING": 3,
}

try:
    fnirs_epochs = create_fnirs_epochs(
        fnirs_raw,
        event_mapping=event_mapping,
        tmin=-2.0,
        tmax=config.trials.task_duration_sec + 10.0,
        baseline=(None, 0),
    )
    
    print(f"   - Epochs creados: {len(fnirs_epochs)}")
    print(f"   - Condiciones: {fnirs_epochs.event_id}")
    
    # Verificar datos de epochs
    epochs_data = fnirs_epochs.get_data()
    print(f"\n8. Estadísticas de epochs:")
    print(f"   - Shape: {epochs_data.shape} (epochs, channels, times)")
    print(f"   - Min: {epochs_data.min():.2e}")
    print(f"   - Max: {epochs_data.max():.2e}")
    print(f"   - Mean: {epochs_data.mean():.2e}")
    print(f"   - Std: {epochs_data.std():.2e}")
    
    # Verificar por condición
    for cond in fnirs_epochs.event_id.keys():
        cond_data = fnirs_epochs[cond].get_data()
        print(f"\n   Condición '{cond}':")
        print(f"   - N epochs: {cond_data.shape[0]}")
        print(f"   - Mean: {cond_data.mean():.2e}")
        print(f"   - Std: {cond_data.std():.2e}")
        
        # Verificar HbO específicamente
        if hbo_channels:
            hbo_picks = [fnirs_epochs.ch_names.index(ch) for ch in hbo_channels if ch in fnirs_epochs.ch_names]
            if hbo_picks:
                hbo_cond_data = cond_data[:, hbo_picks, :]
                print(f"   - HbO Mean: {hbo_cond_data.mean():.2e}")
                print(f"   - HbO en μM (x1e6): {hbo_cond_data.mean() * 1e6:.4f}")
    
    # Verificar ventana de tarea
    times = fnirs_epochs.times
    task_mask = (times >= 2) & (times <= config.trials.task_duration_sec + 5)
    print(f"\n9. Ventana de tarea (2s a {config.trials.task_duration_sec + 5}s):")
    print(f"   - Puntos de tiempo en ventana: {task_mask.sum()}")
    
    if hbo_channels and hbo_picks:
        for cond in fnirs_epochs.event_id.keys():
            cond_data = fnirs_epochs[cond].get_data()[:, hbo_picks, :]
            task_data = cond_data[:, :, task_mask]
            print(f"   - {cond} HbO task window mean: {task_data.mean():.2e}")
            print(f"   - {cond} HbO task window mean (μM): {task_data.mean() * 1e6:.4f}")

except Exception as e:
    print(f"   ERROR creando epochs: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("FIN DEL DIAGNÓSTICO")
print("=" * 70)
