"""Validation of estimates against values reported in papers and model cards.

Like-for-like comparisons:
  * energy, 'estimated_by_authors' (GPU-hours x TDP, e.g. Meta model cards) -> our chip-hours x TDP
    (load = overhead = PUE = 1), i.e. the same accounting boundary;
  * energy, 'measured' -> our full facility estimate (Monte Carlo median and 90% interval);
  * GPU-hours -> our chip-hours, from Method A (Epoch chips x time) and, blind, from Method B
    (compute / (peak x MFU)), which is the path used for models without hardware-time data.
Run after estimate.py:  python -P src/validate.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import energy as E  # noqa: E402
from estimate import agreement_stats, load, run_mc, set_mfu, set_tpu, tpu_spec  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, TAB = ROOT / "data/processed", ROOT / "results/tables"

# Accounting boundary of each reported energy value (from the row's notes in reported_energy.csv):
#   facility     = measured or modelled system power incl. host overhead and PUE  -> our full MC estimate
#   chip_tdp_pue = GPU-hours x TDP x reported PUE                                  -> our chip-h x TDP x PUE_reported
#   chip_tdp     = GPU-hours x TDP, GPUs only                                      -> our chip-h x TDP
BOUNDARY = {"T5-11B": "facility", "Meena": "facility", "Switch": "facility", "GLaM": "facility",
            "GPT-3 175B (davinci)": "facility", "Transformer": "facility", "ELMo": "facility",
            "LLaMA-65B": "chip_tdp_pue", "BLOOM-176B": "chip_tdp"}
# Reported values excluded from a comparison, with reason
EXCLUDE = {("Llama 3.3 70B", "co2_t"): "model card CO2 (12.9 t) inconsistent with its own 7.0M GPU-h; "
                                        "Llama 3.1 70B reports 2,040 t for the same GPU-h"}


def main():
    v = pd.read_csv(ROOT / "data/validation/reported_energy.csv")
    v = v[v.epoch_model_name.notna()].copy()
    for c in ["reported_gpu_hours", "reported_energy_mwh", "reported_co2_t"]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    d = load()
    print("MFU distribution:", set_mfu(d, "calibrated"))
    print("TPU system factor:", set_tpu())
    d = d[d.model.isin(v.epoch_model_name) & (d.method != "")].drop_duplicates("model").reset_index(drop=True)
    if d.empty:
        print("no reported model matches an estimable Epoch record")
        return
    e, c, ch, _ = run_mc(d)
    db = d[d.can_b].reset_index(drop=True)          # Method B, blind to chips x time
    chb = run_mc(db, "B")[2] if len(db) else np.empty((0, 1))
    chb_q = {m: (np.quantile(r, .05), np.median(r), np.quantile(r, .95)) for m, r in zip(db.model, chb)}

    rows = []
    for i, r in d.iterrows():
        for _, rep in v[v.epoch_model_name == r.model].iterrows():
            base = dict(model=r.model, method=r.method, source=rep.source, value_type=rep.value_type)
            if pd.notna(rep.reported_energy_mwh):
                bd = BOUNDARY.get(r.model, "facility")
                chip = ch[i] * r.tdp_w / 1000                                     # kWh at TDP
                e_i = e[i]
                if bd == "facility" and bool(r.get("tpu", False)):
                    # leave-one-out: TPU factor refitted without this model's own measurement
                    a = dict(E.ASSUMPTIONS, tpu_system=tpu_spec(exclude=r.model))
                    e_i = run_mc(d.iloc[[i]], assumptions=a)[0][0]
                arr = {"facility": e_i, "chip_tdp": chip,
                       "chip_tdp_pue": chip * float(rep.pue_reported)}[bd] / 1000  # MWh
                rows.append(dict(base, quantity="energy_mwh", boundary=bd,
                                 reported=rep.reported_energy_mwh, estimated=np.median(arr),
                                 p05=np.quantile(arr, .05), p95=np.quantile(arr, .95)))
            if pd.notna(rep.reported_gpu_hours):
                rows.append(dict(base, quantity="chip_hours", boundary="", reported=rep.reported_gpu_hours,
                                 estimated=np.median(ch[i]), p05=np.quantile(ch[i], .05), p95=np.quantile(ch[i], .95)))
                if r.model in chb_q:
                    lo, med, hi = chb_q[r.model]
                    rows.append(dict(base, method="B", quantity="chip_hours_methodB_blind", boundary="",
                                     reported=rep.reported_gpu_hours, estimated=med, p05=lo, p95=hi))
            if pd.notna(rep.reported_co2_t) and (r.model, "co2_t") not in EXCLUDE:
                rows.append(dict(base, quantity="co2_t", boundary="default grid", reported=rep.reported_co2_t,
                                 estimated=np.median(c[i]), p05=np.quantile(c[i], .05), p95=np.quantile(c[i], .95)))
    # Before calibration: TPU facility rows with the generic GPU assumptions (TDP x load x overhead)
    a0 = {k: val for k, val in E.ASSUMPTIONS.items() if k != "tpu_system"}
    for i, r in d.iterrows():
        if BOUNDARY.get(r.model) == "facility" and bool(r.get("tpu", False)):
            e0 = run_mc(d.iloc[[i]], assumptions=a0)[0][0] / 1000
            for _, rep in v[(v.epoch_model_name == r.model) & v.reported_energy_mwh.notna()].iterrows():
                rows.append(dict(model=r.model, method=r.method, source=rep.source, value_type=rep.value_type,
                                 quantity="energy_mwh_tpu_uncalibrated", boundary="facility",
                                 reported=rep.reported_energy_mwh, estimated=np.median(e0),
                                 p05=np.quantile(e0, .05), p95=np.quantile(e0, .95)))
    pts = pd.DataFrame(rows)
    pts["log10_ratio"] = np.log10(pts.estimated / pts.reported)
    pts["inside_90pi"] = (pts.p05 <= pts.reported) & (pts.reported <= pts.p95)
    TAB.mkdir(parents=True, exist_ok=True)
    pts.to_csv(TAB / "validation_all.csv", index=False)
    pts[pts.quantity == "energy_mwh"].to_csv(TAB / "validation_points.csv", index=False)
    summ = pd.DataFrame([dict(quantity=q, **agreement_stats(g.log10_ratio, g.inside_90pi)) for q, g in pts.groupby("quantity")])
    summ.to_csv(TAB / "validation_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"reported rows: {len(v)}; matched to estimable Epoch models: {d.model.nunique()}")
    print(summ.round(3).to_string(index=False))
    print(pts[["model", "quantity", "boundary", "method", "reported", "estimated", "p05", "p95"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
