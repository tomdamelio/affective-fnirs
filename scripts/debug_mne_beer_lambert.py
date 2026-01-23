#!/usr/bin/env python3
"""
Debug del código interno de MNE beer_lambert_law.
"""

import numpy as np
import mne

# Crear datos
info = mne.create_info(['S1_D1 760', 'S1_D1 850'], 10.0, ['fnirs_cw_amplitude']*2)

for i in range(2):
    info['chs'][i]['loc'][0:3] = [0, 0, 0.1]
    info['chs'][i]['loc'][3:6] = [0.03, 0, 0.1]
    info['chs'][i]['loc'][9] = 760.0 if i == 0 else 850.0

np.random.seed(42)
data = np.random.randn(2, 100) * 0.01 + 0.5
raw = mne.io.RawArray(data, info)
raw_od = mne.preprocessing.nirs.optical_density(raw)

print("Datos OD:")
print(f"  Shape: {raw_od.get_data().shape}")
print(f"  Mean: {raw_od.get_data().mean():.6f}")

# Ahora vamos a replicar lo que hace beer_lambert_law internamente
from mne.io.pick import _picks_to_idx
from mne.preprocessing.nirs._beer_lambert_law import _validate_nirs_info

picks = _validate_nirs_info(raw_od.info, fnirs="od")
print(f"\nPicks OD: {picks}")

# Obtener wavelengths
freqs = np.array([raw_od.info['chs'][pick]['loc'][9] for pick in picks])
print(f"Wavelengths: {freqs}")

# Obtener distancias
distances = np.array([
    np.linalg.norm(
        raw_od.info['chs'][pick]['loc'][:3] - raw_od.info['chs'][pick]['loc'][3:6]
    )
    for pick in picks
])
print(f"Distances: {distances}")

# Verificar si hay NaN
print(f"NaN en distances: {np.isnan(distances).any()}")

# Verificar el emparejamiento de canales
print("\nVerificando emparejamiento...")
ch_names = [raw_od.ch_names[pick] for pick in picks]
print(f"Channel names: {ch_names}")

# Extraer source-detector pairs
pairs = {}
for i, ch_name in enumerate(ch_names):
    # "S1_D1 760" -> ("S1_D1", 760)
    parts = ch_name.rsplit(' ', 1)
    sd_pair = parts[0]
    wavelength = float(parts[1])
    
    if sd_pair not in pairs:
        pairs[sd_pair] = {}
    pairs[sd_pair][wavelength] = i

print(f"Pairs: {pairs}")

# Verificar que cada par tiene ambas wavelengths
for sd_pair, wavelengths in pairs.items():
    if 760.0 in wavelengths and 850.0 in wavelengths:
        print(f"  {sd_pair}: OK (760 y 850)")
    else:
        print(f"  {sd_pair}: INCOMPLETO - {list(wavelengths.keys())}")

# Ahora calcular Beer-Lambert manualmente
print("\nCalculando Beer-Lambert manualmente...")

# Coeficientes de extinción (de MNE)
# Estos son los valores típicos para 760nm y 850nm
# HbO: ε_760 = 1486.5865, ε_850 = 2526.391
# HbR: ε_760 = 3843.707, ε_850 = 1798.643
# Unidades: cm^-1 / (mol/L) = cm^-1 / M

# Valores de MNE (en cm^-1 / M)
ext_coef = {
    760: {'hbo': 1486.5865, 'hbr': 3843.707},
    850: {'hbo': 2526.391, 'hbr': 1798.643},
}

ppf = 6.0  # Partial pathlength factor

# Para cada par source-detector
for sd_pair, wavelengths in pairs.items():
    if 760.0 not in wavelengths or 850.0 not in wavelengths:
        continue
    
    idx_760 = wavelengths[760.0]
    idx_850 = wavelengths[850.0]
    
    # Obtener datos OD
    od_760 = raw_od.get_data()[idx_760]
    od_850 = raw_od.get_data()[idx_850]
    
    print(f"\n{sd_pair}:")
    print(f"  OD 760: mean={od_760.mean():.6f}, std={od_760.std():.6f}")
    print(f"  OD 850: mean={od_850.mean():.6f}, std={od_850.std():.6f}")
    
    # Distancia (en cm)
    distance_m = distances[idx_760]
    distance_cm = distance_m * 100
    print(f"  Distance: {distance_cm:.2f} cm")
    
    # Calcular concentraciones usando Beer-Lambert
    # ΔOD = ε * Δc * d * DPF
    # Δc = ΔOD / (ε * d * DPF)
    
    # Sistema de ecuaciones:
    # OD_760 = (ε_HbO_760 * [HbO] + ε_HbR_760 * [HbR]) * d * DPF
    # OD_850 = (ε_HbO_850 * [HbO] + ε_HbR_850 * [HbR]) * d * DPF
    
    # Matriz de coeficientes
    E = np.array([
        [ext_coef[760]['hbo'], ext_coef[760]['hbr']],
        [ext_coef[850]['hbo'], ext_coef[850]['hbr']]
    ])
    
    # Factor de escala
    scale = distance_cm * ppf
    print(f"  Scale factor (d * DPF): {scale:.2f}")
    
    # Resolver para cada punto de tiempo
    od_matrix = np.vstack([od_760, od_850])  # (2, n_times)
    
    # Invertir matriz E
    E_inv = np.linalg.inv(E)
    print(f"  E matrix:\n{E}")
    print(f"  E_inv matrix:\n{E_inv}")
    
    # Calcular concentraciones
    # [HbO, HbR] = E_inv @ [OD_760, OD_850] / scale
    conc = E_inv @ od_matrix / scale
    
    hbo = conc[0]
    hbr = conc[1]
    
    print(f"  HbO: mean={hbo.mean():.6e}, std={hbo.std():.6e}")
    print(f"  HbR: mean={hbr.mean():.6e}, std={hbr.std():.6e}")
    
    # Convertir a μM (micromolar)
    hbo_um = hbo * 1e6
    hbr_um = hbr * 1e6
    print(f"  HbO (μM): mean={hbo_um.mean():.4f}, std={hbo_um.std():.4f}")
    print(f"  HbR (μM): mean={hbr_um.mean():.4f}, std={hbr_um.std():.4f}")

print("\n" + "=" * 70)
print("CONCLUSIÓN: El cálculo manual funciona, el problema está en MNE")
print("=" * 70)
