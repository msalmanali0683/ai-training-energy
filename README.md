# The energy and carbon cost of training AI models

Code and data for the article *The energy and carbon cost of training AI models: a validated probabilistic
reconstruction* (Muhammad Salman Ali, The University of Lahore; submitted to *Joule*).

The code estimates the training energy and CO₂-equivalent emissions of 290 notable AI models (2012–2026) from
public data, with Monte Carlo uncertainty, validation against published values, sensitivity analysis, trend fits,
machine-learning imputation with SHAP, and scenarios for the largest training run in 2030.

## Run it

Python 3.10 or later. From the repository root:

```bash
pip install -r requirements.txt
pytest tests
python src/prepare.py
python src/estimate.py
python src/validate.py
python src/trends.py
python src/ml.py
python src/figures.py
```

The scripts run in this order and write to `data/processed/` and `results/`. The random seed is 42 everywhere,
so a run reproduces the tables in `results/tables/`. `notebooks/ai_energy_pipeline.ipynb` runs the same stages
in a Kaggle environment.

## Layout

| Path | What |
|---|---|
| `src/energy.py` | Energy and emissions formulas (Methods A and B), assumption distributions, Monte Carlo; unit-tested in `tests/` |
| `src/prepare.py` | Joins Epoch AI models, hardware and grid data into `data/processed/models_energy.csv` |
| `src/estimate.py` | Monte Carlo estimates, MFU calibration (leave-one-out), Method A vs B, comparison with Epoch AI, Sobol and one-at-a-time sensitivity |
| `src/validate.py` | Comparison with published energy, GPU-hours and emissions (`data/validation/reported_energy.csv`) |
| `src/trends.py` | Doubling times (bootstrap), breakpoints, hardware efficiency, 2030 scenarios, grid-location experiment |
| `src/ml.py` | Ridge, random forest, XGBoost and LightGBM against a physical baseline; repeated 5-fold CV, conformal intervals, SHAP, imputation |
| `src/figures.py` | Figures (PNG and PDF) |
| `data/raw/` | Frozen public inputs with SHA-256 hashes; sources and licences in `data/README.md` |
| `data/validation/` | Hand-collected published values, each with source and page or table locator |
| `results/` | Output tables and figures |

Figure files and article figures: `fig3_validation` = Figure 1, `fig1_trends` = Figure 2, `fig2_efficiency` =
Figure 3, `fig4_sensitivity` = Figure 4, `fig5_forecast_location` = Figure 5, `fig6_shap` = Figure S1.

## Licence

Code: MIT. Input data: CC BY 4.0 from Epoch AI, Our World in Data and Ember; please cite the original sources
listed in `data/README.md`.
