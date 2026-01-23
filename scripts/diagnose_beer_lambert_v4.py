#!/usr/bin/env python3
"""
Diagnóstico más profundo de Beer-Lambert usando el ejemplo de MNE-NIRS.
"""

import numpy as np
import mne

print("=" * 70)
print("DIAGNÓSTICO BEER-LAMBERT V4 - Usando ejemplo MNE-NIRS")
print("=" * 70)

print(f"\nMNE version: {mne.__version__}")

# Probar con datos de ejemplo de MNE-NIRS
print("\n1. Cargando datos de ejemplo de MNE-NIRS...")
try:
    from mne_nirs.datasets import fnirs_motor_group
    raw_intensity = mne.io.read_raw_snirf(
        fnirs_motor_group.data_path() / "sub-01" / "nirs" / "sub-01_task-tapping_nirs.snirf",
        preload=True
    )
    print(f"   ✅ Datos de ejemplo cargados")
    print(f"   - Canales: {len(raw_intensity.ch_names)}")
    print(f"   - Tipos: {set(raw_intensity.get_channel_types())}")
    
    # Verificar posiciones
    print(f"\n2. Verificando posiciones en datos de ejemplo:")
    ch = raw_intensity.info['chs'][0]
    print(f"   - loc[0:3] (source): {ch['loc'][0:3]}")
    print(f"   - loc[3:6] (detector): {ch['loc'][3:6]}")
    print(f"   - loc[9] (wavelength): {ch['loc'][9]}")
    
    # Convertir a OD
    print(f"\n3. Convirtiendo a OD...")
    raw_od_example = mne.preprocessing.nirs.optical_density(raw_intensity)
    od_data = raw_od_example.get_data()
    print(f"   - Min: {od_data.min():.6e}")
    print(f"   - Max: {od_data.max():.6e}")
    
    # Aplicar Beer-Lambert
    print(f"\n4. Aplicando Beer-Lambert...")
    raw_haemo_example = mne.preprocessing.nirs.beer_lambert_law(raw_od_example, ppf=6.0)
    hb_data = raw_haemo_example.get_data()
    print(f"   - Min: {hb_data.min():.6e}")
    print(f"   - Max: {hb_data.max():.6e}")
    
    if hb_data.max() != 0:
        print(f"   ✅ Beer-Lambert funciona con datos de ejemplo!")
    
except Exception as e:
    print(f"   ❌ Error con datos de ejemplo: {e}")

# Ahora probar creando datos sintéticos correctamente
print("\n" + "=" * 70)
print("5. Creando datos sintéticos con formato correcto...")

# Usar mne_nirs para crear datos sintéticos
try:
    from mne_nirs.simulation import simulate_nirs_raw
    
    # Simular datos
    raw_sim = simulate_nirs_raw(
        sfreq=10.0,
        amplitude=1.0,
        sig_dur=60.0,
        stim_dur=5.0,
        isi_min=10.0,
        isi_max=15.0,
    )
    print(f"   - Canales simulados: {len(raw_sim.ch_names)}")
    print(f"   - Tipos: {set(raw_sim.get_channel_types())}")
    
    # Verificar posiciones
    ch = raw_sim.info['chs'][0]
    print(f"   - loc[0:3] (source): {ch['loc'][0:3]}")
    print(f"   - loc[3:6] (detector): {ch['loc'][3:6]}")
    
    # Convertir a OD y luego a Hb
    raw_od_sim = mne.preprocessing.nirs.optical_density(raw_sim)
    raw_haemo_sim = mne.preprocessing.nirs.beer_lambert_law(raw_od_sim, ppf=6.0)
    
    hb_sim_data = raw_haemo_sim.get_data()
    print(f"\n6. Resultado Beer-Lambert en datos simulados:")
    print(f"   - Min: {hb_sim_data.min():.6e}")
    print(f"   - Max: {hb_sim_data.max():.6e}")
    
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# Verificar qué hace beer_lambert_law internamente
print("\n" + "=" * 70)
print("7. Investigando beer_lambert_law internamente...")

# Crear datos mínimos para debug
n_samples = 100
sfreq = 10.0

# Crear info con el formato correcto de MNE-NIRS
info = mne.create_info(
    ch_names=['S1_D1 760', 'S1_D1 850'],
    sfreq=sfreq,
    ch_types=['fnirs_cw_amplitude', 'fnirs_cw_amplitude']
)

# Configurar posiciones correctamente
for i, ch_name in enumerate(info.ch_names):
    wavelength = 760e-9 if '760' in ch_name else 850e-9  # En metros!
    
    # Posiciones en metros
    source_pos = np.array([0.0, 0.0, 0.1])
    detector_pos = np.array([0.03, 0.0, 0.1])
    
    info['chs'][i]['loc'][0:3] = source_pos
    info['chs'][i]['loc'][3:6] = detector_pos
    info['chs'][i]['loc'][9] = wavelength  # Wavelength en METROS

# Crear datos de intensidad (valores típicos)
data = np.ones((2, n_samples)) * 0.5  # Intensidad ~0.5

raw_test = mne.io.RawArray(data, info)
print(f"   - Wavelength en loc[9]: {raw_test.info['chs'][0]['loc'][9]}")

# Convertir a OD
raw_od_test = mne.preprocessing.nirs.optical_density(raw_test)
print(f"   - OD min: {raw_od_test.get_data().min():.6e}")
print(f"   - OD max: {raw_od_test.get_data().max():.6e}")

# Verificar wavelength después de OD
print(f"   - Wavelength después de OD: {raw_od_test.info['chs'][0]['loc'][9]}")

# Aplicar Beer-Lambert
raw_haemo_test = mne.preprocessing.nirs.beer_lambert_law(raw_od_test, ppf=6.0)
print(f"   - Hb min: {raw_haemo_test.get_data().min():.6e}")
print(f"   - Hb max: {raw_haemo_test.get_data().max():.6e}")

print("\n" + "=" * 70)
print("CONCLUSIÓN")
print("=" * 70)
