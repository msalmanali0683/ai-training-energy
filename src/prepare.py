"""Build data/processed/models_energy.csv from the frozen raw Epoch and OWID files.

Run from the project root:  python -P src/prepare.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

START_YEAR = 2012

# Model-table hardware name -> hardware-table name. Only same-chip variants are mapped.
HW_MAP = {
    "NVIDIA Tesla K40c": "NVIDIA Tesla K40s",   # GK110B variants
    "NVIDIA Tesla K40m": "NVIDIA Tesla K40s",
    "NVIDIA Tesla K20m": "NVIDIA Tesla K20c",   # GK110 variants
    "NVIDIA H200": "NVIDIA H200 SXM",
    "NVIDIA B200 GPUs": "NVIDIA B200",
    "Nvidia H100 80G": "NVIDIA H100 SXM5 80GB",
}

# Epoch country label -> OWID entity label (others are identical)
COUNTRY_MAP = {"United States of America": "United States", "Korea (Republic of)": "South Korea"}

CONSUMER_PAT = r"GeForce|TITAN|Titan|Quadro|RTX"

COLS = {
    "Model": "model", "Organization": "organization", "Publication date": "pub_date",
    "Domain": "domain", "Parameters": "params", "Training compute (FLOP)": "flop",
    "Training dataset size (total)": "dataset_size", "Training hardware": "hardware_raw",
    "Hardware quantity": "chips", "Training time (hours)": "hours",
    "Hardware utilization (MFU)": "mfu_reported", "Training power draw (W)": "epoch_power_w",
    "Training compute cost (2023 USD)": "cost_usd2023", "Numerical format": "num_format",
    "Frontier model": "epoch_frontier", "Organization categorization": "org_type_raw",
    "Country (of organization)": "country_raw", "Confidence": "confidence",
    "Model accessibility": "accessibility", "Base model": "base_model",
    "Finetune compute (FLOP)": "ft_flop", "Training compute notes": "compute_notes",
}

# Epoch compute notes that derive FLOP from hardware x time x an assumed/quoted utilisation.
# Such rows cannot be used to test MFU or to compare Method A with Method B (circular).
HW_DERIVED_PAT = r"utili[sz]ation|\bMFU\b|\butil\b|% of peak|assum"


def load_hardware() -> pd.DataFrame:
    h = pd.read_csv(RAW / "epoch/ml_hardware/ml_hardware.csv")
    h = h.rename(columns={
        "Hardware name": "hw", "TDP (W)": "tdp_w", "Release date": "hw_release",
        "Tensor-FP16/BF16 performance (FLOP/s)": "f_tensor16", "FP8 performance (FLOP/s)": "f_fp8",
        "FP4 performance (FLOP/s)": "f_fp4", "FP32 (single precision) performance (FLOP/s)": "f_fp32",
        "TF32 (TensorFloat-32) performance (FLOP/s)": "f_tf32", "FP16 (half precision) performance (FLOP/s)": "f_fp16",
        "Manufacturer": "manufacturer", "Type": "hw_type",
    })
    return h[["hw", "manufacturer", "hw_type", "hw_release", "tdp_w", "f_tensor16", "f_fp8", "f_fp4",
              "f_fp32", "f_tf32", "f_fp16"]].set_index("hw")


def pick_peak(row) -> tuple[float, str]:
    """Peak FLOP/s matching the training precision (falls back to tensor-16, then FP32)."""
    fmt = str(row.get("num_format") or "").upper()
    order = []
    if "FP4" in fmt:
        order += ["f_fp4", "f_fp8"]
    if "FP8" in fmt:
        order += ["f_fp8"]
    if fmt == "FP32":
        order += ["f_fp32"]
    if fmt == "TF32":
        order += ["f_tf32"]
    order += ["f_tensor16"]
    if fmt in ("FP16", "BF16"):
        order += ["f_fp16"]
    order += ["f_fp32"]
    for c in order:
        v = row.get(c)
        if pd.notna(v) and v > 0:
            return float(v), c
    return np.nan, ""


def load_carbon() -> pd.DataFrame:
    o = pd.read_csv(RAW / "owid/carbon-intensity-electricity.csv")
    return o.rename(columns={"co2_intensity__gco2_kwh": "ci"})[["entity", "year", "ci"]]


def carbon_lookup(ci: pd.DataFrame, country: str, year: int) -> tuple[float, str]:
    s = ci[ci.entity == country]
    if s.empty:
        s, country = ci[ci.entity == "World"], "World"
    s = s[s.year <= year] if (s.year <= year).any() else s
    r = s.sort_values("year").iloc[-1]
    return float(r.ci), f"{country}:{int(r.year)}"


def first(x):
    return str(x).split(",")[0].strip() if pd.notna(x) else np.nan


def build() -> pd.DataFrame:
    d = pd.read_csv(RAW / "epoch/ai_models/notable_ai_models.csv", low_memory=False)
    d = d[list(COLS)].rename(columns=COLS)
    d["pub_date"] = pd.to_datetime(d["pub_date"], errors="coerce")
    # 4 records list two comma-separated dataset sizes (training stages); keep the larger
    d["dataset_size"] = d.dataset_size.map(
        lambda v: max(float(p) for p in str(v).split(",")) if pd.notna(v) else np.nan)
    d = d[d.pub_date.dt.year >= START_YEAR].copy()
    d["year_frac"] = d.pub_date.dt.year + (d.pub_date.dt.dayofyear - 1) / 365.25
    d["multi_hardware"] = d.hardware_raw.fillna("").str.contains(",")
    d["hw"] = d.hardware_raw.map(first).replace(HW_MAP)

    h = load_hardware()
    d = d.join(h, on="hw")
    d["hw_matched"] = d.hw.notna() & d.tdp_w.notna()
    peaks = d.apply(pick_peak, axis=1, result_type="expand")
    d["peak_flops"], d["peak_source"] = peaks[0], peaks[1]
    d["consumer"] = d.hw.fillna("").str.contains(CONSUMER_PAT)
    d["tpu"] = d.hw.fillna("").str.contains("Google TPU")
    d["epoch_frontier"] = d.epoch_frontier.eq(True)
    d["org_type"] = d.org_type_raw.fillna("").map(
        lambda s: "Industry" if set(s.split(",")) == {"Industry"} else
        ("Academia" if set(s.split(",")) == {"Academia"} else ("Mixed" if s else "Unknown")))
    d["country"] = d.country_raw.map(first).replace(COUNTRY_MAP)

    ci = load_carbon()
    look = [carbon_lookup(ci, c if isinstance(c, str) else "World", int(min(y, ci.year.max())))
            for c, y in zip(d.country, d.pub_date.dt.year)]
    d["ci_g_kwh"] = [v for v, _ in look]
    d["ci_source"] = [s for _, s in look]

    # Fine-tunes: Epoch's training compute includes the base model, so the run's own
    # compute is the fine-tune compute (missing -> Method B not possible).
    d["is_finetune"] = d.base_model.notna()
    d["flop_run"] = np.where(d.is_finetune, d.ft_flop, d.flop)
    d["compute_hw_derived"] = d.compute_notes.fillna("").str.contains(HW_DERIVED_PAT, case=False, regex=True)

    ok_a = d.chips.gt(0) & d.hours.gt(0) & d.hw_matched
    ok_b = d.flop_run.gt(0) & d.peak_flops.gt(0) & d.hw_matched
    d["can_a"], d["can_b"] = ok_a, ok_b
    d["method"] = np.where(ok_a, "A", np.where(ok_b, "B", ""))
    with np.errstate(invalid="ignore", divide="ignore"):
        d["implied_mfu"] = d.flop_run / (d.chips * d.hours * 3600 * d.peak_flops)
    # records whose compute, chips and time cannot all be right (MFU > 1)
    d["inconsistent"] = d.implied_mfu.gt(1.0)
    d["independent_ab"] = d.can_a & d.can_b & ~d.compute_hw_derived & ~d.inconsistent

    # frontier = Epoch flag OR top-10 by compute in its publication year (pre-training runs only)
    base = d[~d.is_finetune]
    d["top10_year"] = base.groupby(base.pub_date.dt.year)["flop"].rank(ascending=False, method="first").le(10)
    d["top10_year"] = d.top10_year.fillna(False).astype(bool)
    d["frontier"] = (d.epoch_frontier | d.top10_year) & ~d.is_finetune
    return d.sort_values("pub_date").reset_index(drop=True)


def coverage(d: pd.DataFrame) -> pd.DataFrame:
    fields = ["flop", "params", "dataset_size", "hardware_raw", "chips", "hours", "mfu_reported",
              "epoch_power_w", "cost_usd2023"]
    rows = [(f, int(d[f].notna().sum()), round(100 * d[f].notna().mean(), 1)) for f in fields]
    rows += [("hardware matched (TDP known)", int(d.hw_matched.sum()), round(100 * d.hw_matched.mean(), 1)),
             ("Method A eligible", int(d.can_a.sum()), round(100 * d.can_a.mean(), 1)),
             ("Method B eligible", int(d.can_b.sum()), round(100 * d.can_b.mean(), 1)),
             ("Both A and B", int((d.can_a & d.can_b).sum()), round(100 * (d.can_a & d.can_b).mean(), 1)),
             ("Both, compute not hardware-derived", int(d.independent_ab.sum()), round(100 * d.independent_ab.mean(), 1)),
             ("Fine-tunes", int(d.is_finetune.sum()), round(100 * d.is_finetune.mean(), 1)),
             ("Inconsistent records (MFU>1)", int(d.inconsistent.sum()), round(100 * d.inconsistent.mean(), 1)),
             ("Any method", int((d.method != "").sum()), round(100 * (d.method != "").mean(), 1))]
    return pd.DataFrame(rows, columns=["field", "n_models", "pct_of_models"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    d = build()
    d.to_csv(OUT / "models_energy.csv", index=False)
    cov = coverage(d)
    cov.to_csv(ROOT / "results/tables/coverage.csv", index=False)
    unmatched = d.loc[d.hardware_raw.notna() & ~d.hw_matched, "hw"].value_counts()
    (OUT / "unmatched_hardware.json").write_text(json.dumps(unmatched.to_dict(), indent=1))
    print(f"{len(d)} models since {START_YEAR}"); print(cov.to_string(index=False))
    print("unmatched hardware:", unmatched.to_dict())
    print("implied MFU, independent rows:", d.loc[d.independent_ab, "implied_mfu"].describe().round(3).to_dict())
