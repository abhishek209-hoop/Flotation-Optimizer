"""Train XGBoost + Random Forest on an 80% split of data/flotation_dataset.csv, evaluate once on the held-out 20%.
Usage: python train.py"""
import json, os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor
from app.config import *

def load():
    df = pd.read_csv(DATA_DIR / DATA_FILE)
    cut = next((c for c in df.columns if "cut" in c.lower()), None)   # auto-detect cut-size column
    if cut and cut != CUT_KEY:
        df = df.rename(columns={cut: CUT_KEY})
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    return train_test_split(df, test_size=TEST_SIZE, random_state=SEED)

def build_meta(tr):
    feats = []
    for f in FEATURES:
        s = dict(f, in_model=(not f["derived"]) and f["key"] in tr.columns)
        if f["key"] in tr.columns:
            c = tr[f["key"]]
            s.update(min=float(c.min()), max=float(c.max()), p01=float(c.quantile(.01)),
                     p99=float(c.quantile(.99)), median=float(c.median()))
        else:                                   # e.g. cut size absent from the data -> informational only
            lo, hi = f["fallback"]; s.update(min=lo, max=hi, p01=lo, p99=hi, median=(lo + hi) / 2)
        feats.append(s)
    r = tr.Residence_Time_min / (tr.Cell_Volume_m3 * 60 / tr.Feed_Rate_t_per_h)
    return dict(features=feats, model_features=[s["key"] for s in feats if s["in_model"]],
                rt_ratio=[float(r.quantile(.01)), float(r.quantile(.99))],
                fixed_params=FIXED_PARAMS, cu_per_chalco=CU_PER_CHALCO,
                recovery_range=RECOVERY_RANGE, models=MODELS)

N_ITER = int(os.environ.get("N_ITER", 20))   # search iterations per model (lower for a quick run)
SPACES = {
  "random_forest": (RandomForestRegressor(random_state=42, n_jobs=1),
      dict(n_estimators=[200, 300], max_depth=[None, 12, 20], min_samples_leaf=[1, 2, 4], max_features=[0.5, 0.7, 1.0])),
  "xgboost": (XGBRegressor(random_state=42, n_jobs=1, objective="reg:squarederror"),
      dict(n_estimators=[300, 500, 800], learning_rate=[0.02, 0.05, 0.1], max_depth=[3, 4, 6],
           subsample=[0.7, 0.9, 1.0], colsample_bytree=[0.7, 0.9, 1.0], min_child_weight=[1, 3, 5], reg_lambda=[1, 3, 10])),
}

def main():
    tr, te = load(); meta = build_meta(tr); cols = meta["model_features"]
    Xtr, ytr, Xte, yte = tr[cols], tr[TARGET], te[cols], te[TARGET]
    print(f"Rows: {len(tr)} train / {len(te)} test")
    print(f"Features used by the models ({len(cols)}): {cols}")
    if CUT_KEY not in cols:
        print(f"NOTE: no cut-size column in the data -> '{CUT_KEY}' is shown in the UI but NOT used by the models.")
    meta["metrics"], meta["importance"] = {}, {}
    MODEL_DIR.mkdir(exist_ok=True)
    for name, (est, space) in SPACES.items():
        search = RandomizedSearchCV(est, space, n_iter=N_ITER, cv=KFold(5, shuffle=True, random_state=42),
                                    scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
        search.fit(Xtr, ytr)
        best = search.best_estimator_
        p = np.clip(best.predict(Xte), *RECOVERY_RANGE)
        meta["metrics"][name] = dict(r2=r2_score(yte, p), mae=mean_absolute_error(yte, p),
                                     rmse=mean_squared_error(yte, p) ** .5, cv_mae=-search.best_score_,
                                     params={k: (v if v is None else float(v) if isinstance(v, float) else v)
                                             for k, v in search.best_params_.items()})
        meta["importance"][name] = dict(zip(cols, map(float, best.feature_importances_)))
        joblib.dump(best, MODEL_DIR / f"{name}.joblib")
        m = meta["metrics"][name]
        print(f"{MODELS[name]:14s} test R2={m['r2']:.3f}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  (CV MAE {m['cv_mae']:.2f})")
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))
    print("Saved models and meta.json to", MODEL_DIR)

if __name__ == "__main__":
    main()
