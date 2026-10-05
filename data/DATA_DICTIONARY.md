# Data dictionary: `flotation_dataset.csv`

1,500 rows, 26 columns, no missing values or duplicates. Synthetic copper flotation data, 01-01-2025 to 28-02-2026 (dates are `dd-mm-yyyy`), balanced across 3 shifts and 4 banks.

| Column | Unit | Role in the project |
|---|---|---|
| `Sample_ID`, `Date`, `Shift`, `Bank`, `Cell no` | - | Identifiers and dashboard slicers. Not model inputs |
| `Chalcopyrite_Grade_pct` | % | Input (feed property, not adjustable) |
| `Cu_Feed_Grade_pct` | % | Equals 0.346 x chalcopyrite grade (CuFeS2 stoichiometry). Redundant, so shown in the UI but not used by the models |
| `Hydrocyclone_Split_pct` | % | Input, adjustable |
| `Feed_Solids_pct` | % | Input, adjustable |
| `Pulp_pH` | - | Input, adjustable (max step 8%) |
| `Collector_Dosage_g_per_t` | g/t | Input, adjustable |
| `Frother_Dosage_g_per_t` | g/t | Input, adjustable |
| `Depressant_Dosage_g_per_t` | g/t | Input, adjustable |
| `pH_Regulator_Dosage_g_per_t` | g/t | Input, adjustable |
| `Air_Flow_Rate_cm_per_s` | cm/s | Input, adjustable |
| `Bubble_Size_mm` | mm | Input, adjustable |
| `Impeller_Speed_rpm` | rpm | Input, adjustable |
| `Cell_Volume_m3` | m3 | Input (equipment, not adjustable) |
| `Feed_Rate_t_per_h` | t/h | Input, adjustable |
| `Residence_Time_min` | min | Input, adjustable, kept consistent with cell volume and feed rate |
| `Collector_Type`, `Frother_Type`, `pH_Regulator_Type` | - | Constant (SIPX, Pine Oil, Lime). Shown as fixed in the UI, not modelled |
| `Copper_Recovery_pct` | % | **Target** |
| `Concentrate_Grade_pct`, `Tailings_Grade_pct` | % | Process outputs. Excluded from the model to avoid leakage |

**Not in the data:** hydrocyclone cut size. The UI has the field, but the models ignore it until a column is added (see README).
