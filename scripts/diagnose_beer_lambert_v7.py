#!/usr/bin/env python3
"""
Diagnóstico: probar con datos reales de MNE-NIRS sample.
"""

import numpy as np
import mne

print("=" * 70)
print("DIAGNÓSTICO BEER-LAMBERT V7 - Datos sample de MNE")
print("=" * 70)

# Usar el sample data de MNE para fNIRS
print("\n1. Buscando datos de ejemplo en MNE...")

# Verificar si hay datos de ejemplo disponibles
try:
    from mne.datasets import testing
    data_path = testing.data_path()
    print(f"   - Data path: {data_path}")
    
    # Buscar archivos SNIRF
    import os
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith('.snirf'):
                print(f"   - Encontrado: {os.path.join(root, f)}")
except Exception as e:
    print(f"   - Error: {e}")

# Crear datos sintéticos que funcionen
print("\n2. Creando datos sintéticos con formato MNE-NIRS correcto...")

# Usar mne.create_info con el formato correcto
n_samples = 1000
sfreq = 10.0

# Nombres de canales en formato MNE-NIRS
ch_names = ['S1_D1 760', 'S1_D1 850', 'S2_D2 760', 'S2_D2 850']

# Crear info
info = mne.create_info(
    ch_names=ch_names,
    sfreq=sfreq,
    ch_types=['fnirs_cw_amplitude'] * 4
)

# Configurar metadata de canales
# Posiciones ficticias pero válidas (en metros)
optode_positions = {
    'S1': np.array([0.0, 0.0, 0.1]),
    'D1': np.array([0.03, 0.0, 0.1]),
    'S2': np.array([0.0, 0.05, 0.1]),
    'D2': np.array([0.03, 0.05, 0.1]),
}

for i, ch_name in enumerate(ch_names):
    # Parsear nombre del canal
    parts = ch_name.split()
    sd_pair = parts[0]  # "S1_D1"
    wavelength = float(parts[1])  # 760 o 850
    
    source = sd_pair.split('_')[0]  # "S1"
    detector = sd_pair.split('_')[1]  # "D1"
    
    # Configurar posiciones
    info['chs'][i]['loc'][0:3] = optode_positions[source]
    info['chs'][i]['loc'][3:6] = optode_positions[detector]
    info['chs'][i]['loc'][9] = wavelength  # En nm

# Crear datos de intensidad realistas
np.random.seed(42)
# Simular señal fNIRS típica: baseline + ruido + respuesta hemodinámica
t = np.arange(n_samples) / sfreq

# Crear respuesta hemodinámica sintética
from scipy.signal import convolve

# HRF canónica simplificada
hrf_t = np.arange(0, 20, 1/sfreq)
hrf = np.exp(-hrf_t/5) * (hrf_t/5)**2 * (hrf_t > 0)
hrf = hrf / hrf.max()

# Estímulos
stim = np.zeros(n_samples)
stim[100] = 1
stim[300] = 1
stim[500] = 1
stim[700] = 1

# Respuesta
response = convolve(stim, hrf, mode='same')

# Datos de intensidad (valores típicos ~0.1-1.0)
data = np.zeros((4, n_samples))
for i in range(4):
    baseline = 0.5 + np.random.randn() * 0.1
    noise = np.random.randn(n_samples) * 0.01
    # Añadir respuesta hemodinámica (pequeña variación en intensidad)
    data[i] = baseline + noise + response * 0.02

print(f"   - Datos de intensidad: min={data.min():.4f}, max={data.max():.4f}")

# Crear Raw
raw = mne.io.RawArray(data, info)

print(f"\n3. Verificando estructura de canales:")
for i in range(len(ch_names)):
    ch = raw.info['chs'][i]
    print(f"   - {ch['ch_name']}:")
    print(f"     loc[0:3] (source): {ch['loc'][0:3]}")
    print(f"     loc[3:6] (detector): {ch['loc'][3:6]}")
    print(f"     loc[9] (wavelength): {ch['loc'][9]}")

# Convertir a OD
print(f"\n4. Convirtiendo a Optical Density...")
raw_od = mne.preprocessing.nirs.optical_density(raw)
od_data = raw_od.get_data()
print(f"   - OD: min={od_data.min():.6f}, max={od_data.max():.6f}")

# Aplicar Beer-Lambert
print(f"\n5. Aplicando Beer-Lambert Law...")
try:
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
    hb_data = raw_haemo.get_data()
    print(f"   - Hb: min={hb_data.min():.6e}, max={hb_data.max():.6e}")
    
    if hb_data.max() != 0 or hb_data.min() != 0:
        print(f"   ✅ Beer-Lambert funciona!")
        
        # Verificar canales HbO y HbR
        hbo_chs = [ch for ch in raw_haemo.ch_names if 'hbo' in ch]
        hbr_chs = [ch for ch in raw_haemo.ch_names if 'hbr' in ch]
        print(f"   - HbO channels: {hbo_chs}")
        print(f"   - HbR channels: {hbr_chs}")
    else:
        print(f"   ❌ Datos en cero")
        
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
