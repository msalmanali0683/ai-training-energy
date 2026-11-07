"""Historical trends (doubling times), hardware efficiency, 2030 scenarios, location experiment.

Run after prepare.py and estimate.py:  python -P src/trends.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import energy as E  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, PROC, TAB = ROOT / "data/raw", ROOT / "data/processed", ROOT / "results/tables"
SEED, N_BOOT = 42, 2000
LOG2 = np.log10(2)
LOCATIONS = ["Norway", "France", "Brazil", "United Kingdom", "United States", "Germany",
             "China", "Pakistan", "Australia", "Poland", "India", "South Africa"]


# ---------------------------------------------------------------- regression helpers
def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm = x.mean()
    b = np.sum((x - xm) * (y - y.mean())) / np.sum((x - xm) ** 2)
    return b, y.mean() - b * xm


def boot_slopes(x, y_draws, rng, n=N_BOOT):
    """Bootstrap over records; each resample also picks one Monte Carlo draw per record.
    y_draws: (records, draws) of log10 values (draws=1 for observed quantities)."""
    x = np.asarray(x, float)
    m, k = y_draws.shape
    out = np.empty((n, 2))
    for i in range(n):
        idx = rng.integers(0, m, m)
        col = rng.integers(0, k, m)
        out[i] = ols(x[idx], y_draws[idx, col])
    return out  # columns: slope, intercept


def trend_row(name, group, x, y_draws, rng):
    b, _ = ols(x, np.median(y_draws, 1))
    bs = boot_slopes(x, y_draws, rng)
    q = np.quantile(bs[:, 0], [0.025, 0.975])
    dt = lambda s: 12 * LOG2 / s  # months
    return dict(quantity=name, group=group, n=len(x), slope_log10_per_yr=b,
                slope_ci_lo=q[0], slope_ci_hi=q[1], growth_x_per_yr=10 ** b,
                doubling_months=dt(b), doubling_ci_lo=dt(q[1]), doubling_ci_hi=dt(q[0])), bs


def breakpoint(x, y, lo=2014.0, hi=2022.0, step=0.25):
    """Two separate line segments; best break by SSE, and BIC gain over a single line."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    b, a = ols(x, y)
    sse1 = np.sum((y - (a + b * x)) ** 2)
    best = (np.inf, None, None, None)
    for t in np.arange(lo, hi + 1e-9, step):
        l, r = x < t, x >= t
        if l.sum() < 8 or r.sum() < 8:
            continue
        b1, a1 = ols(x[l], y[l]); b2, a2 = ols(x[r], y[r])
        sse = np.sum((y[l] - a1 - b1 * x[l]) ** 2) + np.sum((y[r] - a2 - b2 * x[r]) ** 2)
        if sse < best[0]:
            best = (sse, t, b1, b2)
    sse2, t, b1, b2 = best
    bic1 = n * np.log(sse1 / n) + 2 * np.log(n)
    bic2 = n * np.log(sse2 / n) + 5 * np.log(n)
    return dict(break_year=t, slope_before=b1, slope_after=b2, doubling_before_m=12 * LOG2 / b1,
                doubling_after_m=12 * LOG2 / b2, delta_bic=bic1 - bic2)


# ---------------------------------------------------------------- data
def load():
    d = pd.read_csv(PROC / "models_energy.csv", parse_dates=["pub_date"])
    est = pd.read_csv(PROC / "estimates.csv", parse_dates=["pub_date"])
    e = np.load(PROC / "mc_energy_kwh_sample.npy").astype(float)
    c = np.load(PROC / "mc_co2_t_sample.npy").astype(float)
    return d, est, e, c


def hardware_efficiency(rng):
    h = pd.read_csv(RAW / "epoch/ml_hardware/ml_hardware.csv", parse_dates=["Release date"])
    h["year"] = h["Release date"].dt.year + (h["Release date"].dt.dayofyear - 1) / 365.25
    h = h[h.Type.isin(["GPU", "TPU", "GPGPU", "NPU"]) & h["TDP (W)"].gt(0)]
    rows, series = [], {}
    for label, col in [("FP32", "FP32 (single precision) performance (FLOP/s)"),
                       ("Tensor FP16/BF16", "Tensor-FP16/BF16 performance (FLOP/s)")]:
        x = h[h[col].gt(0) & h.year.ge(2008)]
        y = np.log10((x[col] / x["TDP (W)"]).to_numpy())[:, None]
        r, _ = trend_row(f"hardware FLOP/s per W ({label})", "accelerators", x.year, y, rng)
        rows.append(r)
        series[label] = x.assign(flops_per_w=10 ** y[:, 0])[["Hardware name", "Manufacturer", "Type", "year", "flops_per_w"]]
    return pd.DataFrame(rows), series


# ---------------------------------------------------------------- main analysis
def main():
    rng = np.random.default_rng(SEED)
    TAB.mkdir(parents=True, exist_ok=True)
    d, est, e, c = load()
    rows, boots = [], {}

    # compute trend: all pre-training runs with compute (larger set than energy estimates)
    base = d[~d.is_finetune & d.flop.gt(0)]
    for grp, sub in [("all notable", base), ("frontier", base[base.frontier])]:
        r, bs = trend_row("training compute (FLOP)", grp, sub.year_frac, np.log10(sub.flop.to_numpy())[:, None], rng)
        rows.append(r); boots[("compute", grp)] = bs

    # energy and CO2 trends with Monte Carlo draws propagated
    ok = ~est.is_finetune.to_numpy(bool)
    for grp, mask in [("all notable", ok), ("frontier", ok & est.frontier.to_numpy(bool))]:
        x = est.year_frac.to_numpy()[mask]
        for q, arr in [("training energy (kWh)", e), ("training CO2 (t)", c)]:
            r, bs = trend_row(q, grp, x, np.log10(arr[mask]), rng)
            rows.append(r); boots[(q, grp)] = bs
        # effective energy per FLOP (J/FLOP): includes MFU, power, overhead and PUE
        jpf = np.log10(e[mask] * 3.6e6 / est.flop_run.to_numpy()[mask, None])
        good = np.isfinite(jpf).all(1)
        r, bs = trend_row("energy per FLOP (J/FLOP)", grp, x[good], jpf[good], rng)
        rows.append(r); boots[("jpf", grp)] = bs

    hw_rows, hw_series = hardware_efficiency(rng)
    trends = pd.concat([pd.DataFrame(rows), hw_rows], ignore_index=True)
    trends.to_csv(TAB / "doubling_times.csv", index=False)
    pd.concat(hw_series, names=["precision"]).reset_index(level=0).to_csv(TAB / "hardware_efficiency.csv", index=False)

    bp = breakpoint(base.year_frac, np.log10(base.flop))
    bpe = breakpoint(est.year_frac[ok], np.log10(est.energy_mwh_p50[ok]))
    pd.DataFrame([dict(quantity="compute", **bp), dict(quantity="energy (median)", **bpe)]).to_csv(TAB / "breakpoints.csv", index=False)

    # ------------------------------------------------ 2030 scenarios for the largest run
    yr = base.assign(y=base.pub_date.dt.year)
    mx = yr[yr.y >= 2016].loc[lambda t: t.groupby("y").flop.idxmax()]
    bs_max = boot_slopes(mx.year_frac, np.log10(mx.flop.to_numpy())[:, None], rng)
    bs_jpf = boots[("jpf", "frontier")]
    T0, T1 = 2026.0, 2030.5
    ci = pd.read_csv(RAW / "owid/carbon-intensity-electricity.csv").rename(columns={"co2_intensity__gco2_kwh": "ci"})
    ci_year = int(ci[ci.entity == "World"].year.max())
    latest = ci[ci.year == ci_year].set_index("entity").ci
    ci_us = latest["United States"]
    ci_clean = latest[["Norway", "France"]].mean()
    ci_dirty = latest[["Poland", "South Africa", "India"]].mean()
    frontier_days = est.loc[est.frontier & est.hours.gt(0) & est.pub_date.dt.year.ge(2023), "hours"] / 24
    lo_d, hi_d = np.quantile(frontier_days, [0.25, 0.75]) if len(frontier_days) >= 4 else (60, 180)
    days = rng.uniform(lo_d, hi_d, N_BOOT)
    scen = {"Business as usual": (1.0, 1.0, ci_us), "Slowdown after 2026": (0.5, 1.0, ci_us),
            "Efficient & clean": (1.0, 1.5, ci_clean), "Dirty grid": (1.0, 1.0, ci_dirty)}
    q = lambda v: np.quantile(v, [0.05, 0.5, 0.95])
    out = []
    for name, (g_after, eff_mult, cgrid) in scen.items():
        sc, ic = bs_max[:, 0], bs_max[:, 1]
        sj, ij = bs_jpf[:, 0], bs_jpf[:, 1]
        logC = ic + sc * T0 + sc * g_after * (T1 - T0)
        logJ = ij + sj * T0 + sj * eff_mult * (T1 - T0)
        kwh = 10 ** (logC + logJ) / 3.6e6
        co2 = E.co2_tonnes(kwh, cgrid)
        mw = kwh / (days * 24) / 1000
        row = dict(scenario=name, compute_growth_after_2026=g_after, efficiency_rate_mult=eff_mult, grid_g_kwh=cgrid)
        for lab, v in [("flop", 10 ** logC), ("energy_gwh", kwh / 1e6), ("co2_kt", co2 / 1e3), ("avg_power_mw", mw)]:
            row.update({f"{lab}_{k}": x for k, x in zip(["p05", "p50", "p95"], q(v))})
        out.append(row)
    fc = pd.DataFrame(out)
    fc.to_csv(TAB / "forecast_2030.csv", index=False)
    mx[["model", "pub_date", "flop", "confidence"]].to_csv(TAB / "largest_run_per_year.csv", index=False)

    # ------------------------------------------------ robustness: drop records Epoch marks 'Speculative'
    spec_models = set(d.loc[d.confidence.eq("Speculative"), "model"])
    rob = []
    em = ok & est.frontier.to_numpy(bool) & ~est.model.isin(spec_models).to_numpy()
    r2, bs_j2 = trend_row("training energy (kWh)", "frontier excl. speculative", est.year_frac.to_numpy()[em], np.log10(e[em]), rng)
    jpf2 = np.log10(e[em] * 3.6e6 / est.flop_run.to_numpy()[em, None])
    _, bs_jpf2 = trend_row("energy per FLOP (J/FLOP)", "frontier excl. speculative", est.year_frac.to_numpy()[em], jpf2, rng)
    b2 = base[~base.model.isin(spec_models)]
    y2 = b2.assign(y=b2.pub_date.dt.year)
    mx2 = y2[y2.y >= 2016].loc[lambda t: t.groupby("y").flop.idxmax()]
    bs_max2 = boot_slopes(mx2.year_frac, np.log10(mx2.flop.to_numpy())[:, None], rng)
    kwh2 = 10 ** ((bs_max2[:, 1] + bs_max2[:, 0] * T1) + (bs_jpf2[:, 1] + bs_jpf2[:, 0] * T1)) / 3.6e6
    for lab, r_main, r_alt in [("frontier energy doubling (months)",
                                trends[(trends.quantity == "training energy (kWh)") & (trends.group == "frontier")].doubling_months.item(),
                                r2["doubling_months"]),
                               ("BAU 2030 largest-run energy, median (GWh)", bau_gwh_main := fc.loc[fc.scenario == "Business as usual", "energy_gwh_p50"].item(),
                                float(np.median(kwh2) / 1e6))]:
        rob.append(dict(check=lab, all_records=r_main, excluding_speculative=r_alt))
    rob.append(dict(check="BAU 2030 90% interval (GWh), excl. speculative", all_records=np.nan,
                    excluding_speculative=f"{np.quantile(kwh2, .05) / 1e6:.0f}-{np.quantile(kwh2, .95) / 1e6:.0f}"))
    rob.append(dict(check="n speculative records removed (frontier)", all_records=np.nan,
                    excluding_speculative=int((est.frontier & est.model.isin(spec_models)).sum())))
    pd.DataFrame(rob).to_csv(TAB / "robustness_speculative.csv", index=False)
    print("\nrobustness (excluding Epoch 'Speculative'):\n", pd.DataFrame(rob).to_string(index=False))

    # ------------------------------------------------ location experiment
    ref = est.loc[est.frontier].sort_values("energy_mwh_p50").iloc[-1]
    bau_gwh = fc.loc[fc.scenario == "Business as usual", "energy_gwh_p50"].item()
    loc = []
    for cty in LOCATIONS:
        g = latest.get(cty, np.nan)
        loc.append(dict(country=cty, ci_g_kwh=g, ci_year=ci_year, reference_run=ref.model,
                        co2_kt_reference_run=ref.energy_mwh_p50 * g / 1e6,
                        co2_kt_bau_2030_run=bau_gwh * g / 1e3))
    lc = pd.DataFrame(loc).sort_values("ci_g_kwh")
    lc["ratio_to_cleanest"] = lc.ci_g_kwh / lc.ci_g_kwh.min()
    lc.to_csv(TAB / "location_experiment.csv", index=False)

    # ------------------------------------------------ context: national electricity demand (Ember)
    em = pd.read_csv(RAW / "ember/release_generation_yearly_global.csv", low_memory=False)
    dem = em[(em["Electricity source"] == "Demand") & (em["Area type"] == "Country or economy")]
    dem = dem[dem.Year == dem.Year.max()][["Area", "Year", "Generation (TWh)"]].rename(columns={"Generation (TWh)": "demand_twh"})
    ctx = []
    for label, gwh in [(f"Largest estimated run ({ref.model})", ref.energy_mwh_p50 / 1e3),
                       ("BAU 2030 largest run (median)", bau_gwh)]:
        twh = gwh / 1e3
        closest = dem.iloc[(dem.demand_twh - twh).abs().argsort()[:3]]
        ctx.append(dict(run=label, energy_twh=twh, comparable_countries="; ".join(
            f"{a} ({v:.2f} TWh, {y})" for a, v, y in zip(closest.Area, closest.demand_twh, closest.Year))))
    pd.DataFrame(ctx).to_csv(TAB / "energy_context.csv", index=False)

    pd.set_option("display.width", 220)
    print(trends[["quantity", "group", "n", "growth_x_per_yr", "doubling_months", "doubling_ci_lo", "doubling_ci_hi"]].round(2).to_string(index=False))
    print("\nbreakpoints:\n", pd.read_csv(TAB / "breakpoints.csv").round(2).to_string(index=False))
    print("\nlargest run per year:\n", mx[["model", "pub_date", "flop"]].to_string(index=False))
    print("\nforecast:\n", fc[["scenario", "flop_p50", "energy_gwh_p05", "energy_gwh_p50", "energy_gwh_p95", "co2_kt_p50", "avg_power_mw_p50"]].to_string(index=False))
    print("\nfrontier training days (2023+):", frontier_days.describe().round(0).to_dict())
    print("\nlocation:\n", lc.round(1).to_string(index=False))
    print("\ncontext:\n", pd.DataFrame(ctx).to_string(index=False))


if __name__ == "__main__":
    main()
