#!/usr/bin/env python
"""
Script de ejemplo: Carga y exploración de archivos SNIRF.

Este script demuestra cómo cargar y explorar datos fNIRS en formato SNIRF
usando la librería pysnirf2.

Uso:
    micromamba run -n affective-fnirs python scripts/example_load_snirf.py <path_to_snirf>
"""

import sys
from pathlib import Path

import numpy as np
from pysnirf2 import Snirf


def load_and_explore_snirf(snirf_path: Path) -> None:
    """
    Carga un archivo SNIRF y muestra información básica.

    Args:
        snirf_path: Ruta al archivo .snirf

    Returns:
        None. Imprime información en consola.
    """
    print(f"Cargando archivo: {snirf_path}")
    print("-" * 80)

    with Snirf(str(snirf_path), "r") as snirf:
        # Información del sistema
        print("\n📊 INFORMACIÓN DEL SISTEMA")
        print(f"Dispositivo: {snirf.nirs[0].metaDataTags.ManufacturerName}")
        print(f"Fecha: {snirf.nirs[0].metaDataTags.MeasurementDate}")
        print(f"Sujeto: {snirf.nirs[0].metaDataTags.SubjectID}")

        # Información de canales
        n_channels = len(snirf.nirs[0].data[0].measurementList)
        print(f"\n📡 CANALES: {n_channels} canales ópticos")

        # Longitudes de onda
        wavelengths = snirf.nirs[0].probe.wavelengths
        print(f"Longitudes de onda: {wavelengths} nm")

        # Fuentes y detectores
        n_sources = len(snirf.nirs[0].probe.sourcePos3D)
        n_detectors = len(snirf.nirs[0].probe.detectorPos3D)
        print(f"Fuentes: {n_sources}, Detectores: {n_detectors}")

        # Datos temporales
        data_shape = snirf.nirs[0].data[0].dataTimeSeries.shape
        print(f"\n⏱️  DATOS TEMPORALES: {data_shape}")
        print(f"Muestras: {data_shape[0]}, Canales: {data_shape[1]}")

        # Eventos
        if hasattr(snirf.nirs[0], "stim") and len(snirf.nirs[0].stim) > 0:
            print(f"\n🎯 EVENTOS: {len(snirf.nirs[0].stim)} tipos de estímulos")
            for idx, stim in enumerate(snirf.nirs[0].stim):
                n_events = len(stim.data)
                print(f"  - {stim.name}: {n_events} eventos")

        # Datos auxiliares
        if hasattr(snirf.nirs[0], "aux") and len(snirf.nirs[0].aux) > 0:
            print(f"\n📈 DATOS AUXILIARES: {len(snirf.nirs[0].aux)} canales")
            for aux in snirf.nirs[0].aux:
                print(f"  - {aux.name}: {aux.dataTimeSeries.shape[0]} muestras")

    print("\n" + "=" * 80)
    print("✅ Carga exitosa")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scripts/example_load_snirf.py <path_to_snirf>")
        sys.exit(1)

    snirf_file = Path(sys.argv[1])

    if not snirf_file.exists():
        print(f"❌ Error: Archivo no encontrado: {snirf_file}")
        sys.exit(1)

    load_and_explore_snirf(snirf_file)
