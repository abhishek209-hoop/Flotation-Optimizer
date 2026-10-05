"""Surrogate-based suggestion engine (gradient-free, model-agnostic).

Phase 1  maximise predicted recovery inside bounded steps  -> best reachable value
Phase 2  find the smallest (L1) change that reaches the target (or the best reachable value if it can't)
Phase 3  prune: revert each variable, smallest move first, whenever the target is still met
Tree ensembles are piecewise-constant, so differential evolution (no gradients) is used for both models."""
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution

RATIO_W = 300.0   # penalty weight for breaking the residence-time / (volume / feed rate) consistency band


def suggest(predict, meta, x0, target, max_change=0.25, seed=0):
    spec = {f["key"]: f for f in meta["features"]}
    feats = meta["model_features"]
    adj = [k for k in feats if spec[k]["adjustable"]]
    lo, hi, rng = [], [], []
    for k in adj:
        s, step = spec[k], min(max_change, spec[k]["max_step"])
        lo.append(min(x0[k], max(s["p01"], x0[k] * (1 - step))))   # never leave the data-supported range
        hi.append(max(x0[k], min(s["p99"], x0[k] * (1 + step))))
        rng.append(max(s["max"] - s["min"], 1e-9))
    lo, hi, rng, v0 = map(np.array, (lo, hi, rng, [x0[k] for k in adj]))

    def frame(X):
        S = X.shape[1]
        df = pd.DataFrame({k: np.full(S, x0[k], dtype=float) for k in feats})
        for i, k in enumerate(adj):
            df[k] = X[i]
        return df

    r_lo, r_hi = meta["rt_ratio"]
    r0 = x0["Residence_Time_min"] / (x0["Cell_Volume_m3"] * 60 / x0["Feed_Rate_t_per_h"])
    r_lo, r_hi = min(r_lo, r0), max(r_hi, r0)

    def ratio_pen(df):
        r = df.Residence_Time_min / (df.Cell_Volume_m3 * 60 / df.Feed_Rate_t_per_h)
        return np.maximum(0, r_lo - r) + np.maximum(0, r - r_hi)

    def pen_total(df):
        return RATIO_W * ratio_pen(df).values

    def de(obj):
        return differential_evolution(obj, list(zip(lo, hi)), x0=v0, vectorized=True, updating="deferred",
                                      popsize=12, maxiter=60, tol=1e-7, seed=seed, polish=False).x

    cur = float(predict(frame(v0[:, None]))[0])
    out = dict(current=cur, target=target, changes=[], notes=[])
    if target <= cur + 0.05:
        return dict(out, status="already_meets", achieved=cur, max_reachable=cur,
                    notes=["The model already predicts at or above the target for these inputs."])

    def obj1(X):
        df = frame(X)
        return -predict(df) + pen_total(df)

    best1 = de(obj1)
    reach = float(predict(frame(best1[:, None]))[0])
    eff = target if reach >= target else reach - 0.2

    def obj2(X):
        df = frame(X)
        p = predict(df)
        return 100 * (np.abs(X - v0[:, None]) / rng[:, None]).sum(0) + 50 * np.maximum(0, eff - p) ** 2 + pen_total(df)

    x = de(obj2)
    if float(predict(frame(x[:, None]))[0]) < eff - 0.3:
        x = best1                                                    # fall back to the phase-1 optimum

    for i in np.argsort(np.abs(x - v0) / rng):                       # prune smallest moves first
        t = x.copy()
        t[i] = v0[i]
        d = frame(t[:, None])
        tol = 0.4 if abs(x[i] - v0[i]) < 0.02 * abs(v0[i]) else 0.1   # drop noise-level moves more eagerly
        if predict(d)[0] >= eff - tol and ratio_pen(d).iloc[0] <= ratio_pen(frame(x[:, None])).iloc[0] + 1e-9:
            x = t

    achieved = float(predict(frame(x[:, None]))[0])
    for i, k in enumerate(adj):
        dec = spec[k]["dec"]
        new, old = round(float(x[i]), dec), round(float(x0[k]), dec)
        if new == old:
            continue
        out["changes"].append(dict(key=k, label=spec[k]["label"], unit=spec[k]["unit"], current=old, suggested=new,
                                   delta=round(new - old, dec), pct=round(100 * (new - old) / old, 1) if old else None,
                                   direction="increase" if new > old else "decrease"))
    out["changes"].sort(key=lambda c: -abs(c["pct"] or 0))
    out.update(status="reached" if achieved >= target - 0.5 else "closest_feasible",
               achieved=achieved, max_reachable=reach)
    if out["status"] == "closest_feasible":
        out["notes"].append(f"The target is not reachable within the allowed step limits; "
                            f"the best predicted value is {reach:.1f}%.")
    if {"Feed_Rate_t_per_h", "Residence_Time_min"} & {c["key"] for c in out["changes"]}:
        out["notes"].append("Feed rate and residence time are linked (residence time ~ cell volume / feed rate). "
                            "Check that both changes can be achieved together at the plant.")
    return out
