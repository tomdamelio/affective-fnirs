"""Script para verificar que todos los imports del proyecto funcionan."""

import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
import mne
import matplotlib.pyplot as plt
import pyvista as pv
import pyxdf
import cedalion
import cedalion.io as cio
from sklearn.decomposition import PCA
from pint import UnitRegistry
from snirf import Snirf  # cedalion usa snirf, no pysnirf2

print("✓ Todos los imports funcionan correctamente")
print(f"  numpy: {np.__version__}")
print(f"  pandas: {pd.__version__}")
print(f"  scipy: {sp.__version__}")
print(f"  mne: {mne.__version__}")
print("  cedalion: OK")
