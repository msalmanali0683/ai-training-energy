"""Training-energy and CO2 estimation for AI models.

Units used throughout (kept explicit in names to avoid unit errors):
    flop        total floating-point operations (FLOP)
    flops       throughput (FLOP per second)
    *_w         watts,  *_kw kilowatts
    *_h         hours,  *_s seconds
    *_kwh       kilowatt-hours
    ci_g_kwh    grid carbon intensity, grams CO2 per kWh
    co2_t       tonnes CO2

Method A (hardware-time):
    E = chips * TDP * hours * load * overhead * PUE
Method B (compute-based):
    chip_hours = FLOP / (peak_flops * MFU * 3600)
    E = chip_hours * TDP * load * overhead * PUE

load     = average accelerator power / TDP during training
overhead = (accelerator + host CPU, memory, network) power / accelerator power
PUE      = facility power / IT power
"""
from __future__ import annotations

import numpy as np

SECONDS_PER_HOUR = 3600.0
W_PER_KW = 1000.0
G_PER_T = 1e6


# --------------------------------------------------------------------------
# Deterministic formulas (scalars or numpy arrays, broadcasting)
# --------------------------------------------------------------------------
def chip_hours_from_compute(flop, peak_flops, mfu):
    """Accelerator-hours needed to perform `flop` at `mfu` of `peak_flops`."""
    flop, peak_flops, mfu = map(np.asarray, (flop, peak_flops, mfu))
    if np.any(peak_flops <= 0) or np.any(mfu <= 0):
        raise ValueError("peak_flops and mfu must be positive")
    return flop / (peak_flops * mfu * SECONDS_PER_HOUR)


def energy_kwh_from_chip_hours(chip_hours, tdp_w, load=1.0, overhead=1.0, pue=1.0):
    """Facility energy (kWh) for `chip_hours` accelerator-hours."""
    return np.asarray(chip_hours) * (np.asarray(tdp_w) / W_PER_KW) * load * overhead * pue


def energy_method_a(chips, tdp_w, hours, load=1.0, overhead=1.0, pue=1.0):
    """Method A: energy (kWh) from chip count and wall-clock training hours."""
    return energy_kwh_from_chip_hours(np.asarray(chips) * np.asarray(hours), tdp_w, load, overhead, pue)


def energy_method_b(flop, peak_flops, mfu, tdp_w, load=1.0, overhead=1.0, pue=1.0):
    """Method B: energy (kWh) from training compute and hardware peak throughput."""
    return energy_kwh_from_chip_hours(chip_hours_from_compute(flop, peak_flops, mfu), tdp_w, load, overhead, pue)


def implied_mfu(flop, chips, hours, peak_flops):
    """MFU implied by reported compute, chips and time. Values > 1 flag inconsistent records."""
    return np.asarray(flop) / (np.asarray(chips) * np.asarray(hours) * SECONDS_PER_HOUR * np.asarray(peak_flops))


def co2_tonnes(energy_kwh, ci_g_kwh):
    """Operational CO2 (t) = energy (kWh) * carbon intensity (g/kWh) / 1e6."""
    return np.asarray(energy_kwh) * np.asarray(ci_g_kwh) / G_PER_T


def doubling_time_years(slope_log10_per_year):
    """Doubling time (years) for a log10-linear trend with the given slope."""
    return np.log10(2.0) / np.asarray(slope_log10_per_year)


# --------------------------------------------------------------------------
# Assumption distributions for Monte Carlo (justified in docs/assumptions.md)
# --------------------------------------------------------------------------
ASSUMPTIONS = {
    "mfu": ("triangular", (0.15, 0.35, 0.55)),          # used when MFU is not reported
    "load": ("uniform", (0.60, 1.00)),                  # avg accelerator power / TDP
    "overhead_dc": ("uniform", (1.30, 1.90)),           # data-centre accelerators
    "overhead_consumer": ("uniform", (1.10, 1.50)),     # consumer / workstation GPUs
    "pue": ("triangular", (1.08, 1.20, 1.60)),
    # Google TPUs: average system power per chip (incl. host) / TDP, replacing load x overhead.
    # Log-normal fitted to 8 measured/stated Google runs (data/validation/measured_system_power.csv);
    # src/estimate.py refits it from that file at run time.
    "tpu_system": ("lognormal", (np.log(0.727), 0.238, 0.3, 1.5)),
    "ci_rel_sd": ("const", 0.15),                        # relative s.d. on grid carbon intensity
}


def draw(spec, size, rng):
    kind, p = spec
    if kind == "uniform":
        return rng.uniform(p[0], p[1], size)
    if kind == "triangular":
        return rng.triangular(p[0], p[1], p[2], size)
    if kind == "const":
        return np.full(size, p)
    if kind == "lognormal":            # p = (mu, sigma, lower clip, upper clip) on the natural-log scale
        return np.clip(np.exp(rng.normal(p[0], p[1], size)), p[2], p[3])
    raise ValueError(kind)


def calibrated_mfu(implied, lo=0.02, hi=0.9):
    """Log-normal MFU fitted to implied MFU of records whose compute is independent of hardware-time."""
    lm = np.log(np.asarray(implied, float))
    return ("lognormal", (float(lm.mean()), float(lm.std(ddof=1)), lo, hi))


def _col(df, name):
    if name in df:
        return df[name].to_numpy(float)[:, None]
    return np.full((len(df), 1), np.nan)


def monte_carlo(df, n=10_000, seed=42, assumptions=None, ci_col="ci_g_kwh"):
    """Monte Carlo energy/CO2 for every row of `df`.

    Required columns: method ('A' or 'B'), tdp_w, consumer (bool), ci_g_kwh,
    and chips+hours (method A) or flop+peak_flops (method B). Optional: mfu_reported.
    Returns (energy_kwh, co2_t, chip_hours, draws); arrays are rows x n.
    """
    a = dict(ASSUMPTIONS if assumptions is None else assumptions)
    rng = np.random.default_rng(seed)
    shape = (len(df), n)
    load = draw(a["load"], shape, rng)
    pue = draw(a["pue"], shape, rng)
    ov_dc = draw(a["overhead_dc"], shape, rng)
    ov_cons = draw(a["overhead_consumer"], shape, rng)
    consumer = df["consumer"].to_numpy(bool)[:, None]
    overhead = np.where(consumer, ov_cons, ov_dc)
    if "tpu" in df and "tpu_system" in a:
        # TPU rows: one measured system-power factor replaces load x overhead
        tpu = df["tpu"].to_numpy(bool)[:, None]
        sys_f = draw(a["tpu_system"], shape, rng)
        load = np.where(tpu, sys_f, load)
        overhead = np.where(tpu, 1.0, overhead)
    mfu = draw(a["mfu"], shape, rng)
    rep = _col(df, "mfu_reported")
    mfu = np.where(np.isfinite(rep), rep, mfu)

    tdp = _col(df, "tdp_w")
    is_a = (df["method"].to_numpy() == "A")[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        ch_b = _col(df, "flop") / (_col(df, "peak_flops") * mfu * SECONDS_PER_HOUR)
    chip_h = np.where(is_a, _col(df, "chips") * _col(df, "hours"), ch_b)
    energy = energy_kwh_from_chip_hours(chip_h, tdp, load, overhead, pue)

    ci_sd = draw(a["ci_rel_sd"], 1, rng)[0]
    ci_draw = _col(df, ci_col) * np.clip(rng.normal(1.0, ci_sd, shape), 0.2, None)
    co2 = co2_tonnes(energy, ci_draw)
    draws = dict(load=load, pue=pue, overhead=overhead, mfu=mfu, ci=ci_draw)
    return energy, co2, chip_h, draws


def summarise(x, q=(0.05, 0.5, 0.95)):
    """Row-wise quantiles -> (rows x len(q)) array."""
    return np.quantile(x, q, axis=1).T
