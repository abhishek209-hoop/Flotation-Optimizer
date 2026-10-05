"""Framework-free service layer: loading, validation, prediction, optimisation."""
import json, math, joblib, numpy as np, pandas as pd
from .config import MODEL_DIR, MODELS, CU_PER_CHALCO
from .optimizer import suggest


class Service:
    def __init__(self):
        self.meta = json.loads((MODEL_DIR / "meta.json").read_text())
        self.models = {n: joblib.load(MODEL_DIR / f"{n}.joblib") for n in MODELS}
        self.spec = {f["key"]: f for f in self.meta["features"]}
        self.cols = self.meta["model_features"]
        self.lo, self.hi = self.meta["recovery_range"]

    def public_meta(self):
        return self.meta

    def _predict(self, name, df):
        return np.clip(self.models[name].predict(df[self.cols]), self.lo, self.hi)

    def validate(self, body):
        errors, x = {}, {}
        inputs = body.get("inputs") or {}
        for k, s in self.spec.items():
            raw = inputs.get(k)
            if k == "Cu_Feed_Grade_pct" and raw in (None, ""):
                continue                                    # derived from chalcopyrite grade
            try:
                v = float(raw)
                if not math.isfinite(v):
                    raise ValueError
            except (TypeError, ValueError):
                errors[k] = f"{s['label']} is required and must be a number."
                continue
            if not s["min"] <= v <= s["max"]:
                u = f" {s['unit']}" if s["unit"] else ""
                errors[k] = f"{s['label']} must be between {s['min']:g} and {s['max']:g}{u}."
            x[k] = v
        algo = body.get("algo", "xgboost")
        if algo not in MODELS:
            errors["algo"] = "Choose XGBoost or Random Forest."
        target, step = body.get("target"), body.get("max_change_pct", 25)
        if target not in (None, ""):
            try:
                target = float(target)
                if not 30 <= target <= self.hi:
                    errors["target"] = f"Target recovery must be between 30 and {self.hi:g} %."
            except (TypeError, ValueError):
                errors["target"] = "Target recovery must be a number."
        else:
            target = None
        try:
            step = float(step)
            if not 1 <= step <= 50:
                raise ValueError
        except (TypeError, ValueError):
            errors["max_change_pct"] = "Maximum change must be between 1 and 50 %."
        return errors, x, algo, target, step

    def analyze(self, body):
        errors, x, algo, target, step = self.validate(body)
        if errors:
            return {"errors": errors}
        warnings = []
        cu = body["inputs"].get("Cu_Feed_Grade_pct")
        if cu not in (None, "") and abs(float(cu) - CU_PER_CHALCO * x["Chalcopyrite_Grade_pct"]) > 0.05 * float(cu):
            warnings.append("Copper feed grade does not match chalcopyrite grade (Cu = 0.346 x CuFeS2). "
                            "The model uses the chalcopyrite grade.")
        r = x["Residence_Time_min"] / (x["Cell_Volume_m3"] * 60 / x["Feed_Rate_t_per_h"])
        if not self.meta["rt_ratio"][0] <= r <= self.meta["rt_ratio"][1]:
            warnings.append("Residence time is unusual for this cell volume and feed rate compared with the training data; "
                            "predictions may be less reliable.")
        if not self.spec["Hydrocyclone_Cut_Size_um"]["in_model"]:
            warnings.append("Hydrocyclone cut size is not in the training data, so it is recorded but does not affect the prediction.")
        df = pd.DataFrame([{k: x[k] for k in self.cols}])
        preds = {n: float(self._predict(n, df)[0]) for n in MODELS}
        res = dict(algo=algo, prediction=preds[algo], band=self.meta["metrics"][algo]["mae"],
                   comparison=preds, warnings=warnings, optimization=None)
        if target is not None:
            res["optimization"] = suggest(lambda d: self._predict(algo, d), self.meta,
                                          {k: x[k] for k in self.cols}, target, step / 100)
        return res
