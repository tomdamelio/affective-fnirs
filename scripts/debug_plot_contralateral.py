
import sys
from pathlib import Path
import inspect

# Add src to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent / "src"))

import run_analysis

print("Inspecting run_analysis module...")
functions = [o[0] for o in inspect.getmembers(run_analysis, inspect.isfunction)]
print(f"generate_contralateral_erd_plots in functions: {'generate_contralateral_erd_plots' in functions}")
print(f"generate_erp_analysis in functions: {'generate_erp_analysis' in functions}")
print(f"generate_visualizations in functions: {'generate_visualizations' in functions}")

if 'generate_contralateral_erd_plots' in functions:
    print("\nSource of generate_contralateral_erd_plots:")
    print(inspect.getsource(run_analysis.generate_contralateral_erd_plots))

if 'generate_erp_analysis' in functions:
    print("\nSource of generate_erp_analysis:")
    print(inspect.getsource(run_analysis.generate_erp_analysis))
