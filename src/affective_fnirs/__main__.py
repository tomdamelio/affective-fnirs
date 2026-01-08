"""
Entry point for running the pipeline as a module.

Usage:
    python -m affective_fnirs.pipeline [arguments]
    
Or:
    micromamba run -n affective-fnirs python -m affective_fnirs.pipeline [arguments]
"""

from affective_fnirs.pipeline import main

if __name__ == "__main__":
    main()
