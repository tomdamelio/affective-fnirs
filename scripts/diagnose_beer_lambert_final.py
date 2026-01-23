#!/usr/bin/env python3
"""
Diagnóstico final: probar Beer-Lambert con datos mínimos.
"""

import numpy as np
import mne

print("=" * 70)
print("DIAGNÓSTICO BEER-LAMBERT FINAL")
print("=" * 70)

# Crear datos sintéticos mínimos
n_samples = 1000
sfreq = 10.0

# Nombres de canales en formato MNE-NIRS
ch_names = ['S1_D1 760', 'S1_D1 850']

# Crear info
info = mne.create_info(
    ch_names=ch_names,
    sfreq=sfreq,
    ch_types=['fnirs_cw_amplitude'] * 2
)

# Configurar metadata de canales
for i, ch_name in enumerate(ch_names):
    wavelength = 760.0 if '760' in ch_name else 850.0
    
    # Posiciones en metros
    source_pos = np.array([0.0, 0.0, 0.1])
    detector_pos = np.array([0.03, 0.0, 0.1])
    
    info['chs'][i]['loc'][0:3] = source_pos
    info['chs'][i]['loc'][3:6] = detector_pos
    info['chs'][i]['loc'][9] = wavelength

# Crear datos de intensidad
np.random.seed(42)
data = np.random.randn(2, n_samples) * 0.01 + 0.5

raw = mne.io.RawArray(data, info)

print(f"\n1. Datos de intensidad:")
print(f"   - Min: {data.min():.4f}")
print(f"   - Max: {data.max():.4f}")

# Convertir a OD
print(f"\n2. Convirtiendo a OD...")
raw_od = mne.preprocessing.nirs.optical_density(raw)
od_data = raw_od.get_data()
print(f"   - OD min: {od_data.min():.6f}")
print(f"   - OD max: {od_data.max():.6f}")

# Verificar la estructura interna
print(f"\n3. Estructura de canales OD:")
for i, ch in enumerate(raw_od.info['chs']):
    print(f"   Canal {i} ({ch['ch_name']}):")
    print(f"     - loc[0:3]: {ch['loc'][0:3]}")
    print(f"     - loc[3:6]: {ch['loc'][3:6]}")
    print(f"     - loc[9]: {ch['loc'][9]}")
    print(f"     - coil_type: {ch['coil_type']}")

# Aplicar Beer-Lambert
print(f"\n4. Aplicando Beer-Lambert...")
try:
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
    hb_data = raw_haemo.get_data()
    print(f"   - Hb min: {hb_data.min():.6e}")
    print(f"   - Hb max: {hb_data.max():.6e}")
    
    if hb_data.max() != 0 or hb_data.min() != 0:
        print(f"   ✅ Beer-Lambert funciona!")
    else:
        print(f"   ❌ Datos en cero")
        
        # Investigar más
        print(f"\n5. Investigando el problema...")
        
        # Verificar si hay pares de canales
        from mne.io.pick import _picks_to_idx
        picks = _picks_to_idx(raw_od.info, 'fnirs_od')
        print(f"   - Picks OD: {picks}")
        
        # Verificar wavelengths
        for pick in picks:
            ch = raw_od.info['chs'][pick]
            print(f"   - {ch['ch_name']}: wavelength={ch['loc'][9]}")
        
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Probar con mne_nirs directamente
print(f"\n6. Probando con mne_nirs.experimental...")
try:
    import mne_nirs
    from mne_nirs.experimental import simulate_nirs_raw
    
    # Simular datos
    raw_sim = simulate_nirs_raw(sfreq=10.0, duration=60.0)
    print(f"   - Canales simulados: {len(raw_sim.ch_names)}")
    print(f"   - Tipos: {set(raw_sim.get_channel_types())}")
    
    # Verificar estructura
    ch = raw_sim.info['chs'][0]
    print(f"   - loc[0:3]: {ch['loc'][0:3]}")
    print(f"   - loc[9]: {ch['loc'][9]}")
    
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 70)
