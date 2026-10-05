"""Single source of truth for features, units, bounds and operating rules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, MODEL_DIR = ROOT / "data", ROOT / "models"
TARGET = "Copper_Recovery_pct"
RECOVERY_RANGE = (45.0, 99.0)           # clip range for predictions (data spans 47.2 - 98.6)
DATA_FILE = "flotation_dataset.csv"
TEST_SIZE, SEED = 0.2, 42               # random 80/20 split, reproducible
ID_COLS = ["Sample_ID", "Date", "Shift", "Bank", "Cell no"]   # identifiers: not process settings, not model inputs
CU_PER_CHALCO = 0.346                    # Cu mass fraction of CuFeS2 (matches the data exactly)
CUT_KEY = "Hydrocyclone_Cut_Size_um"
FIXED_PARAMS = {"Collector type": "SIPX (sodium isopropyl xanthate)",
                "Frother type": "Pine oil",
                "pH regulator": "Lime (CaO)"}


def F(key, label, unit, group, dec=1, adjustable=True, max_step=1.0, derived=False, fallback=None):
    return dict(key=key, label=label, unit=unit, group=group, dec=dec, adjustable=adjustable,
                max_step=max_step, derived=derived, fallback=fallback)


# adjustable=False -> feed property / equipment: shown and validated, never changed by the optimizer.
# max_step = largest fractional move the optimizer may suggest for that variable.
FEATURES = [
    F("Cu_Feed_Grade_pct", "Copper feed grade", "%", "Feed", 3, False, derived=True),
    F("Chalcopyrite_Grade_pct", "Chalcopyrite grade", "%", "Feed", 3, False),
    F(CUT_KEY, "Hydrocyclone cut size (d50)", "µm", "Hydrocyclone", 0, fallback=(20, 300)),
    F("Hydrocyclone_Split_pct", "Hydrocyclone split", "%", "Hydrocyclone", 1),
    F("Feed_Solids_pct", "Feed solids to flotation", "%", "Pulp", 1),
    F("Pulp_pH", "Pulp pH", "", "Pulp", 2, max_step=0.08),
    F("Collector_Dosage_g_per_t", "Collector dosage", "g/t", "Reagents", 1),
    F("Frother_Dosage_g_per_t", "Frother dosage", "g/t", "Reagents", 1),
    F("Depressant_Dosage_g_per_t", "Depressant dosage", "g/t", "Reagents", 1),
    F("pH_Regulator_Dosage_g_per_t", "pH regulator dosage", "g/t", "Reagents", 0),
    F("Air_Flow_Rate_cm_per_s", "Air flow rate", "cm/s", "Cell", 2),
    F("Bubble_Size_mm", "Bubble size", "mm", "Cell", 2),
    F("Impeller_Speed_rpm", "Impeller speed", "rpm", "Cell", 0),
    F("Cell_Volume_m3", "Cell volume", "m³", "Cell", 1, adjustable=False),
    F("Feed_Rate_t_per_h", "Feed rate", "t/h", "Throughput", 1),
    F("Residence_Time_min", "Residence time", "min", "Throughput", 1),
]
MODELS = {"xgboost": "XGBoost", "random_forest": "Random Forest"}
