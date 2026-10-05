# Problem statement

## Context
In a copper concentrator, froth flotation separates copper minerals (chalcopyrite) from waste rock. Copper recovery, the share of feed copper that reaches the concentrate, depends on many interacting settings: hydrocyclone split, feed solids, pulp pH, reagent dosages (collector, frother, depressant, pH regulator), air flow, bubble size, impeller speed, feed rate and residence time. Operators usually adjust these by experience and trial and error. A recovery shortfall of even one percentage point is worth a large amount of copper over a year.

## Problem
Operators and process engineers need to answer two questions quickly, using the plant's own historical data:
1. **Given the current settings, what recovery should we expect?**
2. **If we want a higher recovery (for example 92%), which settings should go up or down, and by roughly how much?**

Today there is no tool that does both, and one-variable-at-a-time trials are slow, costly and risky on a live circuit.

## Objective
Build a locally hosted web application that:
- predicts copper recovery from the process inputs, using a choice of **XGBoost** or **Random Forest** (trained and compared on the same data);
- accepts a user-defined **target recovery** and recommends which variables to **increase or decrease** and by approximately how much;
- keeps every suggestion inside realistic physical and operational bounds, and prefers the smallest set of changes;
- states clearly that suggestions are model-based approximations, not guarantees.

## Data
`data/flotation_dataset.csv`: 1,500 synthetic flotation records (Jan 2025 to Feb 2026) across 3 shifts and 4 banks, with 15 process inputs, the target `Copper_Recovery_pct`, and two output columns (concentrate and tailings grade) that are excluded to avoid leakage. See `data/DATA_DICTIONARY.md`.

## Scope
**In scope**
- Supervised regression for copper recovery, with an 80/20 train/test split and cross-validated tuning.
- Comparison of the two models on held-out data (R2, MAE, RMSE).
- A surrogate-based optimization layer (gradient-free search) with bounds, step limits and a residence-time consistency rule.
- Input validation, clear output formatting, model selection in the UI, local serving on localhost.
- A Power BI dashboard for exploring the historical data (`dashboard/`).

**Out of scope**
- Real-time control of the plant or connection to live sensors.
- Causal guarantees. The model learns associations from the data.
- Economic optimization (reagent cost versus recovery gain).
- Concentrate grade as a second target.

## Constraints and design rules
- Collector (SIPX), frother (Pine oil) and pH regulator (Lime) are fixed and shown read-only.
- Feed grade and cell volume are never changed by the optimizer.
- Suggested values stay within the 1st to 99th percentile of the training data and within a user-set maximum change (default 25%; pulp pH capped at 8%).
- Residence time must stay consistent with cell volume and feed rate.
- Hydrocyclone cut size is requested as an input but is absent from the dataset, so it is recorded and validated but not used by the models.

## Success criteria
- Both models trained and evaluated on the held-out 20% test set, with metrics shown in the app.
- Boosted model reaches at least R2 0.85 and MAE under 3 percentage points on the test set (a baseline run of a gradient-boosting model reached about 0.89 and 2.4, so this is realistic).
- For reachable targets, the predicted recovery after the suggested changes meets the target; for unreachable targets, the app returns the closest feasible result and says so.
- No suggestion falls outside its bounds (checked by `tests/`).
- The app starts with two commands and runs offline on localhost.

## Assumptions and limitations
- The data are synthetic, so findings illustrate the method and do not transfer directly to a real plant.
- Tree models cannot extrapolate beyond the training data, so inputs outside the observed ranges are rejected.
- Predictions have a typical error (the test MAE), shown next to every result.
- Any real-world change must be validated through plant testing and operator judgement.
