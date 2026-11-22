# Ames Housing: Sale Price Prediction

Comparing eight regression models, from a regularised linear model to boosted trees, on the Kaggle **[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)** dataset (Ames, Iowa). The project covers the full workflow: missing-value handling, feature engineering, skew and target transforms, outlier removal, leakage-aware preprocessing, cross-validated hyperparameter search, a stacking check, and interpretation of the results.

> This was a learning exercise. Several rounds of hyperparameter tuning were run along the way. Stacking was tried once, at the end, and gave no gain beyond fold-to-fold noise.

## Results

Each model was tuned with 5-fold cross-validation on 1,165 training houses (scored by RMSE on the log of the price, i.e. RMSLE), then scored once on a held-out test set of 292 houses. The mean training price is about $180,000.

| Model | CV RMSLE | CV std | Test RMSE | RMSE as % of mean price | Test MAE | Test R² |
|---|---|---|---|---|---|---|
| ElasticNet | 0.1131 | 0.0046 | $18,796 | 10.4% | $13,863 | 0.943 |
| XGBoost | 0.1136 | 0.0074 | $19,937 | 11.1% | $13,692 | 0.936 |
| Gradient Boosting | 0.1173 | 0.0068 | $20,454 | 11.4% | $14,380 | 0.933 |
| Random Forest | 0.1300 | 0.0067 | $24,381 | 13.5% | $16,046 | 0.904 |
| AdaBoost | 0.1343 | 0.0043 | $24,508 | 13.6% | $16,453 | 0.903 |
| SVR | 0.1513 | 0.0039 | $40,361 | 22.4% | $20,317 | 0.737 |
| Decision Tree | 0.1727 | 0.0107 | $32,409 | 18.0% | $22,402 | 0.831 |
| KNN | 0.1826 | 0.0099 | $42,522 | 23.6% | $27,158 | 0.709 |

Models are ranked by **cross-validated RMSLE**. The test set is small, so it is used as a final check and not for choosing a model.

### Takeaways

- **Three models are effectively tied at the top:** ElasticNet, XGBoost and Gradient Boosting. Their CV scores (0.1131, 0.1136, 0.1173) differ by less than the fold-to-fold standard deviation (0.005 to 0.007), and they are also the three best on the test set (RMSE about $18.8k to $20.5k, roughly 10% to 11% of the mean price). ElasticNet has the lowest test RMSE and the highest R², XGBoost the lowest test MAE. No ranking inside the group is claimed.
- **Stacking did not separate from the best single model.** A Ridge meta-model over ElasticNet, XGBoost and Gradient Boosting gave a CV RMSLE of 0.1093 (std 0.0053), about 0.004 (roughly 3%) lower than ElasticNet's 0.1131 (std 0.0046). That gap is smaller than the fold-to-fold standard deviation, so it is not significant, and the stack is treated as a tie with the top group. Its test RMSE was $18,271 (10.1% of the mean price) against $18,796 for ElasticNet. The meta-model weights were ElasticNet 0.519, XGBoost 0.411 and Gradient Boosting 0.090. ElasticNet remains the simplest model that matches the best score.
- **A regularised linear model matched the boosted trees.** The best ElasticNet uses `alpha` of about 0.0045 and `l1_ratio` 0.9.
- **Random Forest and AdaBoost** are clearly behind the top group in cross-validation (0.130 and 0.134), and **SVR, Decision Tree and KNN** are the weakest (0.151 to 0.183). SVR's test RMSE ($40.4k) is high compared with its test RMSLE (0.156), which suggests a few large dollar errors; this was not investigated.
- **Main drivers (ElasticNet coefficients on standardised features):** `QualityxSize` (+0.10), `GrLivArea` (+0.048), `QualityxCond` (+0.043), `LotArea` (+0.037) and `1stFlrSF` (+0.035) push the log price up, while `HouseAge` (-0.046) and `RemodAge` (-0.011) push it down. Correlated features share credit, and no test was run to measure how much each engineered feature contributes.

## Approach

**1. Cleaning.** In this dataset a missing value usually means the house does not have that feature, so each fill followed a check on the data.

| Columns | Action |
|---|---|
| `PoolQC`, `MiscFeature` (99.5% and 96.3% missing) | Dropped. The `Has_PoolArea` and `Has_MiscVal` flags keep the information |
| `Alley`, `Fence`, `MasVnrType`, `FireplaceQu`, `Garage*`, `Bsmt*` | Explicit `NA` category (checked, for example, that all missing garage rows have `GarageArea = 0`) |
| `GarageYrBlt` | `0` for houses without a garage |
| `MasVnrArea` | `0` |
| `Electrical` | The single row with a missing value is dropped |
| `LotFrontage` (17.7% missing) | Filled with the (building type, neighbourhood) mean, then the neighbourhood median, both computed from **training rows only** |

**2. Feature engineering.**

| Step | Details |
|---|---|
| Flags | `Has_*` flags for 13 mostly-zero columns (porches, pool, second floor, veneer, and others), `MultiKitchen`, `Has_Garage` |
| Ages | `GarageAge` (sale year minus garage year, or -1 with no garage), `HouseAge`, `RemodAge`, `RemodDone` |
| Totals and ratios | `TotalNumBathrooms`, `TotRoomsInclBath`, `AvgRoomSize` |
| Interactions | `QualityxSize` (quality x living area), `QualityxCond` |
| Ordinal encoding | 18 quality, condition and finish columns mapped to numbers that keep their order |
| One-hot encoding | The remaining text columns (`drop_first=True`) |

**3. Outliers.** Two houses (large `QualityxSize` but priced under $300,000, both in Edwards) were removed. A third large, cheaper house was looked at and kept. This leaves 1,457 of the original 1,460 houses.

**4. Transforms.** `log1p` of the target and of 16 skewed size columns, applied after the derived features were built. Predictions are converted back with `expm1` for the dollar metrics.

**5. Modelling and search.** Every model is a scikit-learn `Pipeline` of `StandardScaler` followed by the estimator, so scaling is fitted on the training folds only. The split is 80/20 (`random_state=42`). ElasticNet, KNN, SVR and the Decision Tree use `GridSearchCV`. Random Forest, AdaBoost, Gradient Boosting and XGBoost use `RandomizedSearchCV` with 30 draws. All searches use the same shuffled 5-fold split (seed 42).

**6. Stacking check.** ElasticNet, XGBoost and Gradient Boosting, each at its tuned parameters, were combined with scikit-learn's `StackingRegressor` and a `RidgeCV` meta-model. The meta-model is trained on out-of-fold predictions built with the same 5-fold split, so it only sees predictions the base models made for houses they were not trained on. The stack was scored with the same CV and then once on the test set.

<details>
<summary>Best parameters found</summary>

| Model | Best parameters |
|---|---|
| ElasticNet | `alpha=0.0045`, `l1_ratio=0.9` |
| KNN | `n_neighbors=6` |
| SVR | `kernel=poly`, `degree=3`, `C=10`, `epsilon=0.01`, `gamma=auto` |
| Decision Tree | `max_depth=29`, `min_samples_leaf=8` |
| Random Forest | `n_estimators=489`, `max_depth=19`, `max_features=0.365`, `min_samples_leaf=3` |
| AdaBoost | `n_estimators=570`, `learning_rate=0.120`, tree `max_depth=7`, tree `min_samples_leaf=5` |
| Gradient Boosting | `n_estimators=1105`, `learning_rate=0.0213`, `max_depth=3`, `min_samples_leaf=25` |
| XGBoost | `n_estimators=1436`, `learning_rate=0.0147`, `max_depth=5`, `min_child_weight=8`, `subsample=0.70`, `colsample_bytree=0.47`, `reg_lambda=1.28`, `reg_alpha=0.0016` |

</details>

## Getting started

**1. Install the dependencies** (Python 3.10 or newer is recommended):

```bash
pip install -r requirements.txt
```

**2. Get the data.** The dataset is not included in this repository. Download `train.csv` (and optionally `data_description.txt`) from the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), or with the Kaggle CLI:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
```

Put the files in a `data/` folder and set the file paths in the notebook's loading cells (they currently point to a local folder on the author's machine) to `data/train.csv` and `data/data_description.txt`.

**3. Run the notebook.**

```bash
jupyter notebook Housing_Price_Advanced.ipynb
```

Run the cells from top to bottom, once (the log transforms would be applied twice if a cell were run again). The tuning loop is the slow part, because the boosted models run many fits.

## Project structure

```
.
├── README.md
├── Housing_Price_Advanced.ipynb   # the full analysis
├── requirements.txt
├── LICENSE
└── data/                          # train.csv goes here (not committed)
```

## Limitations

- **Small test set.** 292 houses, so small differences between models are not reliable. Models were chosen on cross-validation.
- **CV to test gap.** For the top three models, the test RMSLE (0.121, 0.125, 0.128) is 0.008 to 0.011 higher than the CV RMSLE. A harder test split, or a CV score that is slightly optimistic because it is the best of many tried settings, could both explain this.
- **Search ranges.** Several best values sit at or near the edge of their range: Decision Tree `max_depth` 29 (range 3 to 29), AdaBoost tree depth 7 (2 to 7), Random Forest `max_depth` 19 (2 to 19), XGBoost `max_depth` 5 (2 to 5), Gradient Boosting `learning_rate` 0.021 (0.02 to 0.4) and SVR `epsilon` 0.01 (the lowest value tried). A wider search might change those models' scores. It was not run.
- **Search budget.** The random searches used 30 draws per model, and whether they had converged was not checked.
- **Preprocessing outside the pipeline.** One-hot encoding is done on all rows before the split (it does not use the target). The `LotFrontage` fill is computed from the training set once, not inside each cross-validation fold.
- **No before/after comparison** was run for the individual changes (ordinal encoding, the `GarageAge` definition, the outlier removal, the training-only `LotFrontage` fill), so their separate effects are not measured.
- **Garage age.** `GarageAge` uses `GarageYrBlt` as recorded, with no check for garage years before the house was built or after the sale.
- **Outliers.** Two houses were removed before the split, so the test set excludes them too.
- **Stacking test.** Stacking was run once, with three base models and a Ridge meta-model. The base-model settings were chosen on the same CV folds that score the stack, as for the single models. No fold-by-fold comparison with ElasticNet was run, and the correlation between the models' errors was not checked.
- **Reproducibility.** The split, the CV folds and the random search are seeded. The Random Forest, Decision Tree, AdaBoost and Gradient Boosting models have no `random_state`, so a rerun gives slightly different numbers.

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Jupyter.
