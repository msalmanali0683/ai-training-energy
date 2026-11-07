# Data

All raw files were downloaded on **2026-11-07** and are frozen in `raw/`. Never edit them.
SHA-256, size, and source URL of every file: [`raw/download_log.csv`](raw/download_log.csv).
The Kaggle notebook re-checks these hashes before running.

| Folder | Source | Licence | Notes |
|---|---|---|---|
| `raw/epoch/ai_models/` | Epoch AI, *Data on AI models*, https://epoch.ai/data/ai_models.zip (files dated 2026-11-07) | CC BY 4.0 | `notable_ai_models.csv` (1,092 rows, 47 cols) is the main table; `frontier_ai_models.csv`, `large_scale_ai_models.csv`, `all_ai_models.csv` kept for robustness checks. Cite as in the `README.md` inside the folder. |
| `raw/epoch/ml_hardware/` | Epoch AI, *Data on machine learning hardware*, https://epoch.ai/data/ml_hardware.zip | CC BY 4.0 | 183 accelerators: TDP, peak FLOP/s by precision, release date. |
| `raw/owid/` | Our World in Data, *Carbon intensity of electricity* (grapher `carbon-intensity-electricity`, built from Ember and Energy Institute data) | CC BY 4.0 | gCO2/kWh by country-year, latest year 2025. Metadata JSON kept. |
| `raw/ember/` | Ember, *Yearly electricity data* (global release, long format), https://files.ember-energy.org/public-downloads/generation/outputs/release_generation_yearly_global.csv | CC BY 4.0 | Used for national electricity **demand** (TWh) in context comparisons. The old storage.googleapis.com URL now returns a notice pointing to this URL. |
| `validation/` | Hand-collected reported values (papers, model cards) | — | Every row has `source` and `source_locator` (table/section). See `validation/collection_log.md`. |

## Processing decisions (implemented in `src/prepare.py`)

- **Scope:** notable models published from 2012-01-01 (945 rows).
- **Hardware join:** first listed chip of `Training hardware`; 96% of names match exactly. Six same-chip variants are mapped by hand (`HW_MAP`, e.g. *NVIDIA Tesla K40c → K40s*, *NVIDIA H200 → H200 SXM*). Chips absent from the hardware table (Amazon Trainium1, Cerebras CS-2, Tesla C2075, GTX 590, Quadro GP100, TITAN RTX; 7 models) are left unmatched rather than filled from other sources.
- **Peak FLOP/s:** chosen to match `Numerical format` (FP8/FP4 → FP8/FP4 column; FP32 → FP32; TF32 → TF32; otherwise tensor FP16/BF16, falling back to FP32 for chips without tensor units).
- **Fine-tunes:** Epoch's `Training compute (FLOP)` for fine-tuned models includes the base model. The run's own compute is `Finetune compute (FLOP)`; fine-tunes without it are not estimated by Method B and are excluded from trend fits.
- **Circularity flag:** 88 of 143 records with both compute and hardware-time have compute *derived* from hardware × time × an assumed utilisation (detected in `Training compute notes`). They are excluded from MFU checks and from the Method A vs B comparison (`independent_ab`).
- **Inconsistent records:** 7 records imply MFU > 1 (compute, chips and time cannot all be right); flagged `inconsistent`.
- **Dataset size:** 4 records list two comma-separated sizes (training stages); the larger is kept.
- **Grid carbon intensity:** first-listed organisation country, publication year (latest available year if later); `World` if missing. A proxy for data-centre location, stated as a limitation.
- **Frontier:** Epoch `Frontier model` flag OR top-10 pre-training compute in its publication year.
