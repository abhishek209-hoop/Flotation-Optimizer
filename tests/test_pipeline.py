"""Run with: pytest -q   (model tests are skipped until `python train.py` has been run)"""
import pandas as pd, pytest
from app.config import DATA_DIR, DATA_FILE, FEATURES, MODEL_DIR, TARGET, CU_PER_CHALCO

needs_models = pytest.mark.skipif(not (MODEL_DIR / "meta.json").exists(), reason="run python train.py first")


def test_dataset_schema():
    df = pd.read_csv(DATA_DIR / DATA_FILE)
    missing = [f["key"] for f in FEATURES if f["key"] not in df.columns and not f["fallback"]]
    assert not missing, missing
    assert TARGET in df.columns and not df[TARGET].isna().any()


def _service_and_inputs():
    from app.service import Service
    s = Service()
    x = {f["key"]: f["median"] for f in s.meta["features"]}
    x["Cu_Feed_Grade_pct"] = CU_PER_CHALCO * x["Chalcopyrite_Grade_pct"]
    return s, x


@needs_models
def test_validation_rejects_bad_input():
    s, x = _service_and_inputs()
    x["Pulp_pH"], x["Feed_Rate_t_per_h"] = 99, "abc"
    errors = s.analyze(dict(algo="xgboost", inputs=x))["errors"]
    assert {"Pulp_pH", "Feed_Rate_t_per_h"} <= set(errors)


@needs_models
@pytest.mark.parametrize("algo", ["xgboost", "random_forest"])
def test_suggestions_respect_bounds(algo):
    s, x = _service_and_inputs()
    now = s.analyze(dict(algo=algo, inputs=x))["prediction"]
    opt = s.analyze(dict(algo=algo, inputs=x, target=min(now + 3, 98.5), max_change_pct=20))["optimization"]
    assert opt["changes"], "expected at least one suggested change"      # guards against a vacuous pass
    for c in opt["changes"]:
        sp = s.spec[c["key"]]
        tol = 0.5 * 10 ** -sp["dec"]
        assert sp["adjustable"]
        assert sp["p01"] - tol <= c["suggested"] <= sp["p99"] + tol
        assert abs(c["pct"]) <= 20.5
    assert opt["achieved"] >= opt["current"]
