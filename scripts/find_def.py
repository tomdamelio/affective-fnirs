
import sys

def find_def(filename, func_name):
    print(f"Searching for 'def {func_name}' in {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if f"def {func_name}" in line:
                    print(f"Found at line {i+1}: {line.strip()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_def(r"c:\Users\tdamelio\Desktop\fnirs\affective-fnirs\scripts\run_analysis.py", "build_mne_objects")
