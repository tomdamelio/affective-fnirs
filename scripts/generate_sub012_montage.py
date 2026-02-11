
import json
from pathlib import Path

def generate_montage():
    # Define channels pattern from analysis of screenshot
    # Format: (Source, Detector, Type, LabelSuffix)
    # Types: Long, Short
    # Wavelengths: 760, 850 (generated automatically)
    
    # Source Map
    S = {
        1: "S1_AFF5h",
        2: "S2_AFF6h",
        3: "S3_CCP5h",
        4: "S4_TPP7h",
        5: "S5_TPP7h", 
        6: "S6_CCP6h",
        7: "S7_TPP8h",
        8: "S8_TPP8h",
        9: "S9_F7",
        10: "S10_F8",
        13: "S13_xAF7",
        14: "S14_xAF8",
        15: "S15_xCCP5h",
        16: "S16_xCPP6h"
    }
    
    # Detector Map
    D = {
        1: "D1_AF7",
        2: "D2_F5",
        3: "D3_AF8",
        4: "D4_F6",
        5: "D5_C5",
        6: "D6_CPP5h",
        7: "D7_TP7",
        8: "D8_C6",
        9: "D9_CPP6h",
        10: "D10_TP8"
    }
    
    # Pairs (C1 to C24)
    # List of (SourceIdx, DetectorIdx, Type)
    pairs = [
        (1, 1, "Long"),   # C1
        (1, 2, "Long"),   # C2
        (2, 3, "Long"),   # C3
        (2, 4, "Long"),   # C4
        (3, 5, "Long"),   # C5
        (4, 5, "Long"),   # C6
        (4, 7, "Long"),   # C7
        (8, 10, "Long"),  # C8
        (3, 6, "Long"),   # C9
        (6, 9, "Long"),   # C10
        (6, 8, "Long"),   # C11
        (7, 8, "Long"),   # C12
        (8, 9, "Long"),   # C13
        (7, 10, "Long"),  # C14
        (5, 6, "Long"),   # C15
        (5, 7, "Long"),   # C16
        (10, 4, "Long"),  # C17
        (10, 3, "Long"),  # C18
        (14, 3, "Short"), # C19
        (15, 6, "Short"), # C20
        (16, 9, "Short"), # C21
        (13, 1, "Short"), # C22 (Inferred S13, Short)
        (9, 2, "Long"),   # C23
        (9, 1, "Long")    # C24
    ]
    
    montage = []
    idx_counter = 0
    
    for i, (s_idx, d_idx, ch_type) in enumerate(pairs):
        s_label = S[s_idx]
        d_label = D[d_idx]
        
        # Clean labels for location_label (remove S#_ prefix)
        s_loc = s_label.split('_')[1]
        d_loc = d_label.split('_')[1]
        
        # Hb (760)
        montage.append({
            "channel_idx": idx_counter,
            "source": s_label,
            "detector": d_label,
            "wavelength": 760,
            "type": ch_type,
            "location_label": f"{s_loc}-{d_loc}_Hb"
        })
        idx_counter += 1
        
        # HbO (850)
        montage.append({
            "channel_idx": idx_counter,
            "source": s_label,
            "detector": d_label,
            "wavelength": 850,
            "type": ch_type,
            "location_label": f"{s_loc}-{d_loc}_HbO"
        })
        idx_counter += 1
        
    json_content = {
        "TaskName": "fingertapping",
        "SamplingFrequency": 8.12,
        "Manufacturer": "Cortivision",
        "ManufacturersModelName": "Photon Cap C20",
        "NIRSChannelCount": 48,  # 24 pairs * 2
        "SourceDetectorSeparation": {
            "Long": "30 mm",
            "Short": "8 mm"
        },
        "ChMontage": montage
    }
    
    output_path = Path("data/raw/sub-012/ses-001/sub-012_ses-001_task-fingertapping_nirs.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(json_content, f, indent=4)
        
    print(f"Generated {output_path} with {len(montage)} channels.")

if __name__ == "__main__":
    generate_montage()
