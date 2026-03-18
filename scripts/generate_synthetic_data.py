"""
generate_synthetic_data.py
==========================
Generates a realistic (but entirely fake) synthetic dataset of repair orders.
This allows anyone cloning the repository to run `make train` and `make predict`
end-to-end without needing access to the proprietary, confidential customer data.

The generated data mimics the schema of the real JSON:
{
    "input": {
        "make": "VW",
        "calculatedPositions": [ { "text": "...", "genericCostCenter": "...", "totalPrice": 100, "totalTime": ... } ]
    },
    "output": {
        "bodyrepair": 1.5,
        ...
    }
}
"""

import json
import random
from pathlib import Path

# Important paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "synthetic_orders.json"

# Targets that we need to generate outputs for
TARGETS = [
    "calibration", "wheelmeasurement", "bodymeasurement",
    "dismounting", "bodyrepair", "assembly", "plasticrepair",
    "allTiresService", "cleaning", "paintingPreparation",
    "paintingSpraying", "paintingFinish", "hailrepair", "glas"
]

MAKES = ["VW", "BMW", "MERCEDES", "AUDI", "FORD", "OPEL", "SKODA", "RENAULT", "MINI", "PORSCHE", "TESLA", "TOYOTA"]

COST_CENTERS = ["bodywork", "painting", "paintmaterial", "material", "others"]

# Fake vocabulary to construct positions
PAINTING_WORDS = ["LACKIEREN", "BEILACKIEREN", "FARBTON", "ANMISCHEN", "KLARLACK", "VORARBEITEN", "DECKLACK"]
BODYWORK_WORDS = ["STOSSFAENGER", "SCHWEISSEN", "RICHTEN", "A+E", "ERSATZ", "VERKLEIDUNG", "KOTFLUEGEL", "SEITENTEIL", "DELLEN", "HAGEL", "FRONT", "AUSTAUSCH"]
MATERIAL_WORDS = ["KLEBSTOFF", "SCHRAUBE", "CLIP", "HALTER", "REINIGER", "KLEINMATERIAL", "DICHTMASSE"]
MISC_WORDS = ["KALIBRIEREN", "VERMESSEN", "REIFEN", "RAD", "ACHSMESSUNG", "SCHEIBE", "GLAS"]

def generate_position(cc_bias):
    """Generates a realistic-looking line item based on the cost center."""
    text_words = []
    
    # 1 to 4 words per line item
    num_words = random.randint(1, 4)
    
    vocab = BODYWORK_WORDS  # default
    if cc_bias == "painting" or cc_bias == "paintmaterial":
        vocab = PAINTING_WORDS
    elif cc_bias == "material":
        vocab = MATERIAL_WORDS
    elif cc_bias == "others":
        vocab = MISC_WORDS
        
    for _ in range(num_words):
        text_words.append(random.choice(vocab))
        
    text = " ".join(text_words)
    
    # Prices and times (time is typical AW - Arbeitswerte, e.g. 5 mins = 5, 12 per hour)
    price = round(random.uniform(5.0, 500.0), 2)
    time_units = random.randint(0, 30) if cc_bias != "material" else 0
    
    return {
        "text": text,
        "genericCostCenter": cc_bias,
        "totalPrice": price,
        "totalTime": time_units
    }

def generate_order(force_target=None):
    """Generates a single fake repair order."""
    make = random.choice(MAKES)
    num_positions = random.randint(1, 15)
    
    positions = []
    for _ in range(num_positions):
        # Pick a random cost center
        cc = random.choice(COST_CENTERS)
        positions.append(generate_position(cc))
        
    # Generate random output targets
    # Most targets are 0 most of the time (sparsity)
    outputs = {}
    for tgt in TARGETS:
        if force_target == tgt:
            is_active = True
        else:
            is_active = random.random() < 0.15
            
        if is_active:
            # Random duration between 0.5 and 10 hours
            # Some are quick (cleaning), some are long (bodyrepair)
            if tgt in ["cleaning", "wheelmeasurement"]:
                duration = round(random.uniform(0.1, 1.5), 2)
            else:
                duration = round(random.uniform(0.5, 8.0), 2)
        else:
            duration = 0.0
            
        outputs[tgt] = duration
        
    # Occasionally trigger a 'hailrepair' rule
    if any("HAGEL" in p["text"] or "DELLEN" in p["text"] for p in positions):
        outputs["hailrepair"] = round(random.uniform(2.0, 25.0), 2)

    if force_target == "hailrepair" and outputs["hailrepair"] == 0.0:
         outputs["hailrepair"] = round(random.uniform(2.0, 25.0), 2)

    return {
        "input": {
            "make": make,
            "calculatedPositions": positions
        },
        "output": outputs
    }

def main():
    print("Generating synthetic dataset (500 orders)...")
    dataset = []
    
    # Guarantee at least 10 positive examples of each target to prevent split errors
    for tgt in TARGETS:
        for _ in range(10):  # 10 × 14 = 140 guaranteed, rest filled randomly
            dataset.append(generate_order(force_target=tgt))
            
    # Fill the rest with random data
    while len(dataset) < 500:
        dataset.append(generate_order())
        
    random.shuffle(dataset)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Success! Wrote {len(dataset)} fake orders to {OUTPUT_FILE}")
    print("This allows anyone to run 'make train' locally.")

if __name__ == "__main__":
    main()
