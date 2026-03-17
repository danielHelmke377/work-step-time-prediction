"""
config.py — Shared constants for the Work Step Time Prediction project.
=====================================================================
All scripts (model_phase2.py, predict.py, gbert equivalents) should
import their constants from here to avoid copy-paste drift.
"""

# 14 output work-step targets (order matters for display)
OUTPUT_TARGETS = [
    "calibration", "wheelmeasurement", "bodymeasurement",
    "dismounting", "bodyrepair", "assembly", "plasticrepair",
    "allTiresService", "cleaning", "paintingPreparation",
    "paintingSpraying", "paintingFinish", "hailrepair", "glas",
]

# Human-readable labels for reports and plots
TARGET_LABELS = {
    "calibration":         "Calibration (ADAS/cameras)",
    "wheelmeasurement":    "Wheel alignment measurement",
    "bodymeasurement":     "Body/chassis measurement",
    "dismounting":         "Dis-/mounting",
    "bodyrepair":          "Body repair",
    "assembly":            "Assembly",
    "plasticrepair":       "Plastic repair",
    "allTiresService":     "Tyre service",
    "cleaning":            "Cleaning",
    "paintingPreparation": "Painting — preparation",
    "paintingSpraying":    "Painting — spraying",
    "paintingFinish":      "Painting — finish",
    "hailrepair":          "Hail repair",
    "glas":                "Glass replacement",
}

# Generic cost centers present in order positions
COST_CENTERS = ["bodywork", "painting", "paintmaterial", "material", "others", "hail"]

# Top vehicle makes for one-hot encoding
TOP_MAKES = ["VOLKSWAGEN", "MERCEDES-BENZ", "BMW", "FORD", "SKODA", "AUDI", "OPEL", "TESLA"]

# Domain keyword regex patterns (German automotive body-shop vocabulary)
KEYWORD_FLAGS = {
    "kw_vermessung":    r"vermess|kinematik|spur|sturz",
    "kw_kalibrier":     r"kalibrier|adas|fas|kamera.*kalib|radar.*ausrich",
    "kw_glas":          r"glas|scheibe|frontscheibe|windschutz|heckscheibe|windlauf",
    "kw_hagel":         r"hagel|dellen|pdr|smart.*repar",
    "kw_reifen":        r"reifen|felge|rad(?:wechsel|montage|service)",
    "kw_reinigung":     r"reinigung|waesch|polier",
    "kw_lack":          r"lackier|lack(?!material)|oberflaech.*lack|neu.*lack",
    "kw_vorbereitung":  r"vorbereitung|grundier|fuellerauftrag|vorbereit",
    "kw_klebetechnik":  r"klebe|klebetechnik",
    "kw_montage":       r"a\+e|montage|einbau|ausbau|demontage|ersatz",
    "kw_hybrid":        r"hybrid|elektro|hochspannung|hv.system",
    "kw_plastik":       r"plastik|kunststoff|stossfaenger.*repar",
    "kw_karosserie":    r"karosserie|blech|beul|richt",
    "kw_scheibe_ers":   r"scheibe.*ers|frontscheibe|windschutz.*ers",
    "kw_dellen":        r"dellen|dent|beule",
    "kw_sensor":        r"sensor|pdc|adas|ultraschall",
    "kw_material":      r"kleinmaterial|ersatzteil|lackmaterial",
}

# Keyword → target mapping (used for explanation in inference)
TARGET_KEYWORD_MAP = {
    "calibration":         ["kw_kalibrier", "kw_sensor"],
    "wheelmeasurement":    ["kw_vermessung"],
    "bodymeasurement":     ["kw_vermessung", "kw_karosserie"],
    "dismounting":         ["kw_montage"],
    "bodyrepair":          ["kw_karosserie"],
    "assembly":            ["kw_montage"],
    "plasticrepair":       ["kw_plastik"],
    "allTiresService":     ["kw_reifen"],
    "cleaning":            ["kw_reinigung"],
    "paintingPreparation": ["kw_vorbereitung", "kw_lack"],
    "paintingSpraying":    ["kw_lack"],
    "paintingFinish":      ["kw_lack", "kw_klebetechnik"],
    "hailrepair":          ["kw_hagel", "kw_dellen"],
    "glas":                ["kw_glas", "kw_scheibe_ers"],
}
