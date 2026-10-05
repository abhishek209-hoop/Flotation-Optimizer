<<<<<<< HEAD
# Flotation Optimizer

Predict copper recovery from flotation process settings with **XGBoost** or **Random Forest**, then get model-based suggestions for which variables to increase or decrease to reach a target recovery. Runs locally with FastAPI.

> Suggestions are approximations learned from historical data. They are not guarantees. See [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md).

![App screenshot]([docs/app_screenshot.png](https://1drv.ms/i/c/1b580748fa8bcff0/IQA3ytxG-OiWTI1rxqGE3TplAcDDigYlWch1jaxuqGHp-mI?e=CCSQ5h)
(https://1drv.ms/i/c/1b580748fa8bcff0/IQBqKN0MGDZWTqxPwYlItxdOATuQE6m3vDTKJvAPWO-ad5M?e=XhvWh0))
<!-- Add a screenshot of the running app at docs/app_screenshot.png -->

## Features
- Predicts copper recovery from 14 process inputs, with a typical-error band.
- Model picker in the UI (XGBoost or Random Forest), with side-by-side predictions and test metrics.
- Target-recovery mode: lists variables to increase or decrease, by how much, and the predicted result.
- Respects physical and operational bounds, step limits and the residence-time rule. Smallest change first.
- Clear input validation with field-level messages. Fixed reagents (SIPX, Pine oil, Lime) are shown read-only.
- Power BI dashboard for exploring the historical data.

## Dataset
`data/flotation_dataset.csv`: 1,500 synthetic records, 3 shifts, 4 banks, Jan 2025 to Feb 2026. Column details: [data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md).

## Quick start
```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py                   # trains both models (about 5-15 min; N_ITER=5 python train.py is faster)
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000. API docs: http://127.0.0.1:8000/docs. Use Python 3.9-3.12.

Run the checks with `pytest -q` (model tests run after training).

## How it works
1. **Prepare:** drop identifiers, constants and output columns (concentrate and tailings grade); keep chalcopyrite grade (copper grade is 0.346 x chalcopyrite, so it is redundant).
2. **Train:** random 80/20 split (seed 42); 5-fold cross-validated random search on the training part for each model; one evaluation on the held-out 20%.
3. **Serve:** FastAPI loads both models; `/api/analyze` validates input, predicts, and optionally optimizes.
4. **Optimize:** the trained model is a surrogate. Differential evolution (gradient-free, suits tree models) finds the best reachable recovery inside bounded steps, then the smallest change that reaches the target, then prunes small moves.

## Results
Filled in after `python train.py` :

| Model | Test R2 | MAE (pts) | RMSE (pts) |
|---|---|---|---|
| Random Forest | [0.735] | [5.3] | [7.8] |
| XGBoost | [0.871] | [3.8] | [5.4] |

## Project structure
```
flotation-optimizer/
├── app/
│   ├── config.py        # features, units, adjustable flags, step caps
│   ├── optimizer.py     # suggestion engine
│   ├── service.py       # validation, prediction, optimization
│   ├── main.py          # FastAPI routes
│   └── static/index.html
├── data/                # flotation_dataset.csv, DATA_DICTIONARY.md
├── dashboard/           # Flotation_dashboard.pbix
├── models/              # created by train.py (not committed)
├── tests/test_pipeline.py
├── train.py
├── PROBLEM_STATEMENT.md
└── requirements.txt
```

## Dashboard
`dashboard/Flotation_dashboard.pbix` (open in Power BI Desktop) has KPI cards for recovery, concentrate grade and tailings grade, slicers for bank, shift and date, and charts of recovery against residence time, collector dosage, frother dosage, pulp pH, hydrocyclone split bins and feed solids. The report's table is named `flotation_dataset_synthetic_86_2`. After the file rename, point it to `data/flotation_dataset.csv` under **Transform data > Data source settings**.

## Notes and limitations
- **Cut size:** hydrocyclone cut size is not in the dataset. The field is shown and validated but ignored by the models. To include it, add a column with "cut" in its name and re-run `python train.py`.
- The data are synthetic; results illustrate the method.
- Trees cannot extrapolate, so inputs outside the training ranges are rejected.

## Future work
Prediction intervals, cost-aware optimization, concentrate grade as a second target, live data and scheduled retraining.

## License
MIT. See [LICENSE](LICENSE).

## Author
Saranga Abhishek, IIT (ISM) Dhanbad. GitHub: [abhishek209-hoop](https://github.com/abhishek209-hoop)
=======
# Flotation-Optimizer
>>>>>>> 63fd62ae83ce8ba41f13a4eac9cbac2bf0a2a172
