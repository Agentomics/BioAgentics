"""Configuration for ehr-sepsis-early-warning project."""

from __future__ import annotations

from bioagentics.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "ehr-sepsis-early-warning"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "ehr-sepsis-early-warning"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# --- MIMIC-IV table item IDs ---
# Vitals from chartevents (MIMIC-IV itemid mappings)
VITAL_ITEMIDS: dict[str, list[int]] = {
    "heart_rate": [220045],
    "sbp": [220050, 220179],
    "map": [220052, 220181],
    "resp_rate": [220210, 224690],
    "spo2": [220277],
    "temperature": [223761, 223762],  # F and C
}

# Labs from labevents (MIMIC-IV itemid mappings)
LAB_ITEMIDS: dict[str, list[int]] = {
    "wbc": [51301, 51300],
    "hemoglobin": [51222, 50811],
    "hematocrit": [51221, 50810],
    "platelets": [51265],
    "lactate": [50813],
    "creatinine": [50912],
    "bun": [51006],
    "glucose": [50931, 50809],
    "sodium": [50983, 50824],
    "potassium": [50971, 50822],
    "chloride": [50902, 50806],
    "bicarbonate": [50882, 50803],
    "bilirubin_total": [50885],
    "alt": [50861],
    "ast": [50878],
    "albumin": [50862],
    "inr": [51237],
    "ptt": [51275],
    "pao2": [50821],
    "fio2": [50816, 223835],
}

# Temperature conversion (Fahrenheit itemid)
TEMP_FAHRENHEIT_ITEMIDS = [223761]

# Demographics
ETHNICITY_MAP: dict[str, str] = {
    "WHITE": "white",
    "WHITE - RUSSIAN": "white",
    "WHITE - OTHER EUROPEAN": "white",
    "WHITE - BRAZILIAN": "white",
    "WHITE - EASTERN EUROPEAN": "white",
    "BLACK/AFRICAN AMERICAN": "black",
    "BLACK/AFRICAN": "black",
    "BLACK/CAPE VERDEAN": "black",
    "BLACK/HAITIAN": "black",
    "HISPANIC/LATINO - PUERTO RICAN": "hispanic",
    "HISPANIC/LATINO - DOMINICAN": "hispanic",
    "HISPANIC/LATINO - GUATEMALAN": "hispanic",
    "HISPANIC/LATINO - CUBAN": "hispanic",
    "HISPANIC/LATINO - SALVADORAN": "hispanic",
    "HISPANIC/LATINO - CENTRAL AMERICAN": "hispanic",
    "HISPANIC/LATINO - MEXICAN": "hispanic",
    "HISPANIC/LATINO - COLOMBIAN": "hispanic",
    "HISPANIC/LATINO - HONDURAN": "hispanic",
    "HISPANIC OR LATINO": "hispanic",
    "ASIAN": "asian",
    "ASIAN - CHINESE": "asian",
    "ASIAN - SOUTH EAST ASIAN": "asian",
    "ASIAN - ASIAN INDIAN": "asian",
    "ASIAN - KOREAN": "asian",
    "ASIAN - JAPANESE": "asian",
}

# All vital and lab feature names (ordered)
VITAL_FEATURES = list(VITAL_ITEMIDS.keys())
LAB_FEATURES = list(LAB_ITEMIDS.keys())
ALL_FEATURES = VITAL_FEATURES + LAB_FEATURES

# Prediction lookahead hours
LOOKAHEAD_HOURS = [4, 6, 8, 12]

# Cross-validation
OUTER_CV_FOLDS = 5
INNER_CV_FOLDS = 3
RANDOM_STATE = 42

# SOFA score thresholds
SOFA_THRESHOLD = 2

# Lookback window for temporal models
TEMPORAL_LOOKBACK_HOURS = 24
