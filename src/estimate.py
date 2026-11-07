"""Monte Carlo energy/CO2 estimates, Method A vs B agreement, Epoch comparison, sensitivity.

Reads data/processed/models_energy.csv; writes data/processed/estimates.csv and results/tables/*.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import energy as E  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, TAB = ROOT / "data/processed", ROOT / "results/tables"
N_DRAWS, SEED = 10_000, 42


def load() -> pd.DataFrame:
    d = pd.read_csv(PROC / "models_energy.csv", parse_dates=["pub_date"])
    d = d.rename(columns={"flop": "flop_total"}).assign(flop=lambda x: x.flop_run)
    d["method"] = d["method"].fillna("")   # empty string is read back as NaN
    return d


def run_mc(d: pd.DataFrame, method: str | None = None, n=N_DRAWS, seed=SEED, assumptions=None):
    x = d.copy()
    if method:
        x["method"] = method
    return E.monte_carlo(x, n=n, seed=seed, assumptions=assumptions)


def estimates(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = d[d.method != ""].reset_index(drop=True)
    e, c, ch, _ = run_mc(x)
    qe, qc, qh = E.summarise(e / 1000), E.summarise(c), E.summarise(ch)  # MWh, t, chip-h
    for i, q in enumerate(["p05", "p50", "p95"]):
        x[f"energy_mwh_{q}"] = qe[:, i]
        x[f"co2_t_{q}"] = qc[:, i]
        x[f"chip_hours_{q}"] = qh[:, i]
    # Epoch's peak-power figure turned into energy (provisioned power x wall-clock time)
    x["epoch_energy_mwh"] = x.epoch_power_w * x.hours / 1e6
    return x, e, c


def ab_agreement(d: pd.DataFrame) -> pd.DataFrame:
    """Method B vs Method A on records whose compute was not derived from hardware x time."""
    x = d[d.independent_ab].reset_index(drop=True)
    ea, *_ = run_mc(x, "A")
    eb, *_ = run_mc(x, "B")   # same seed -> same load/overhead/PUE draws; differs only via MFU
    a50, b50 = np.median(ea, 1), np.median(eb, 1)
    x["energy_A_mwh"], x["energy_B_mwh"] = a50 / 1000, b50 / 1000
    x["log10_ratio_B_A"] = np.log10(b50 / a50)
    x["A_in_B_90pi"] = (np.quantile(eb, .05, 1) <= a50) & (a50 <= np.quantile(eb, .95, 1))
    return x


def mfu_calibration_check(d: pd.DataFrame, n=4000) -> pd.DataFrame:
    """Leave-one-out: fit a log-normal MFU to the implied MFU of the other independent records,
    rerun Method B for the held-out record, and test whether Method A falls in its 90% interval.
    Compares the literature prior (ASSUMPTIONS['mfu']) with the data-calibrated alternative."""
    x = d[d.independent_ab & d.implied_mfu.between(0.02, 1.0)].reset_index(drop=True)
    ea, *_ = run_mc(x, "A", n=n)
    a50 = np.median(ea, 1)
    out = []
    for i in range(len(x)):
        lm = np.log(np.delete(x.implied_mfu.to_numpy(), i))
        rng = np.random.default_rng(SEED + i)
        mfu = np.clip(np.exp(rng.normal(lm.mean(), lm.std(ddof=1), n)), 0.02, 0.9)
        a = dict(E.ASSUMPTIONS)
        xi = x.iloc[[i]].assign(mfu_reported=np.nan)
        eb_prior, *_ = run_mc(xi, "B", n=n, seed=SEED + i)
        e_cal = []
        for _ in range(1):
            # same draws for the other factors; swap MFU draws for the calibrated ones
            _, _, _, dr = E.monte_carlo(xi.assign(method="B"), n=n, seed=SEED + i, assumptions=a)
            ch = xi.flop.to_numpy()[:, None] / (xi.peak_flops.to_numpy()[:, None] * mfu[None, :] * 3600)
            e_cal = E.energy_kwh_from_chip_hours(ch, xi.tdp_w.to_numpy()[:, None], dr["load"], dr["overhead"], dr["pue"])
        for lab, eb in [("literature prior", eb_prior), ("LOO data-calibrated", e_cal)]:
            out.append(dict(model=x.model[i], mfu_dist=lab, A=a50[i], B50=np.median(eb),
                            inside=bool(np.quantile(eb, .05) <= a50[i] <= np.quantile(eb, .95))))
    r = pd.DataFrame(out)
    r["log10_ratio"] = np.log10(r.B50 / r.A)
    return r.groupby("mfu_dist").apply(lambda g: pd.Series(agreement_stats(g.log10_ratio, g.inside))).reset_index()


def agreement_stats(lr: pd.Series, inside: pd.Series | None = None) -> dict:
    lr = np.asarray(lr, float)
    r = 10 ** lr
    out = dict(n=int(lr.size), median_ratio=float(np.median(r)),
               mean_abs_log10_err=float(np.mean(np.abs(lr))),
               within_2x=float(np.mean(np.abs(lr) <= np.log10(2))),
               within_3x=float(np.mean(np.abs(lr) <= np.log10(3))),
               mape_pct=float(100 * np.mean(np.abs(r - 1))),
               median_ape_pct=float(100 * np.median(np.abs(r - 1))))
    if inside is not None:
        out["coverage_90pi"] = float(np.mean(inside))
    return out


def log_variance_shares() -> pd.DataFrame:
    """Energy is a product of independent factors, so Var(log E) = sum Var(log factor).
    Each factor's share equals its first-order (= total) Sobol index on the log scale."""
    rng = np.random.default_rng(SEED)
    a, n = E.ASSUMPTIONS, 200_000
    rows = []
    for label, keys in [("Method B (MFU not reported)", ["mfu", "load", "overhead_dc", "pue", "ci_rel_sd"]),
                        ("Method A", ["load", "overhead_dc", "pue", "ci_rel_sd"])]:
        v = {}
        for k in keys:
            s = np.clip(rng.normal(1, a[k][1], n), 0.2, None) if k == "ci_rel_sd" else E.draw(a[k], n, rng)
            v[k] = np.var(np.log10(1 / s if k == "mfu" else s))
        tot_e = sum(val for k, val in v.items() if k != "ci_rel_sd")
        tot_c = sum(v.values())
        for k, val in v.items():
            rows.append(dict(method=label, factor=k,
                             share_energy=np.nan if k == "ci_rel_sd" else val / tot_e,
                             share_co2=val / tot_c))
    return pd.DataFrame(rows)


def sobol_check(seed=SEED) -> pd.DataFrame:
    """Saltelli/Sobol indices on linear-scale energy for one Method-B run (robustness check)."""
    from SALib.analyze import sobol
    from SALib.sample import sobol as sobol_sample
    # SALib 'triang' bounds are [lower, upper, peak position as a fraction of the range];
    # 'lognorm' bounds are [mean, sd] of the natural log.
    kind, p = E.ASSUMPTIONS["mfu"]
    mfu_b, mfu_d = ([p[0], p[1]], "lognorm") if kind == "lognormal" else \
        ([p[0], p[2], (p[1] - p[0]) / (p[2] - p[0])], "triang")
    prob = {"num_vars": 4, "names": ["mfu", "load", "overhead", "pue"],
            "bounds": [mfu_b, [0.6, 1.0], [1.3, 1.9], [1.08, 1.60, (1.20 - 1.08) / 0.52]],
            "dists": [mfu_d, "unif", "unif", "triang"]}
    X = sobol_sample.sample(prob, 2 ** 14, seed=seed)
    Y = E.energy_method_b(1e25, 989e12, X[:, 0], 700, X[:, 1], X[:, 2], X[:, 3])
    s = sobol.analyze(prob, Y, seed=seed)
    return pd.DataFrame({"factor": prob["names"], "S1": s["S1"], "S1_conf": s["S1_conf"],
                         "ST": s["ST"], "ST_conf": s["ST_conf"]})


def one_at_a_time(d: pd.DataFrame) -> pd.DataFrame:
    """Change in summed median energy of all models when one factor is fixed at its bound."""
    x = d[d.method != ""].reset_index(drop=True)
    base, *_ = run_mc(x, n=2000)
    base_tot = np.median(base, 1).sum()
    rows = []
    for k in ["mfu", "load", "overhead_dc", "pue"]:
        kind, p = E.ASSUMPTIONS[k]
        if kind == "lognormal":       # 5th / 95th percentiles stand in for bounds
            lo, hi = np.exp(p[0] - 1.645 * p[1]), np.exp(p[0] + 1.645 * p[1])
        else:
            lo, hi = p[0], p[-1]
        for tag, val in [("low", round(float(lo), 3)), ("high", round(float(hi), 3))]:
            a = dict(E.ASSUMPTIONS)
            a[k] = ("const", val)
            e, *_ = run_mc(x, n=2000, assumptions=a)
            rows.append(dict(factor=k, setting=tag, value=val,
                             total_energy_change_pct=100 * (np.median(e, 1).sum() / base_tot - 1)))
    return pd.DataFrame(rows)


def set_mfu(d: pd.DataFrame, which: str = "calibrated"):
    """Main analysis uses the data-calibrated MFU (LOO check: 92% coverage of 90% intervals vs 74% for
    the literature prior). The literature prior stays available as a sensitivity case."""
    if which == "calibrated":
        imp = d.loc[d.independent_ab & d.implied_mfu.between(0.02, 1.0), "implied_mfu"]
        E.ASSUMPTIONS["mfu"] = E.calibrated_mfu(imp)
    else:
        E.ASSUMPTIONS["mfu"] = LITERATURE_MFU
    return E.ASSUMPTIONS["mfu"]


LITERATURE_MFU = E.ASSUMPTIONS["mfu"]
MEASURED_POWER = ROOT / "data/validation/measured_system_power.csv"


def tpu_spec(exclude: str | None = None):
    """Log-normal TPU system-power factor (system W per chip / TDP) fitted to Google's measured runs.
    `exclude` drops one Epoch model (leave-one-out validation)."""
    t = pd.read_csv(MEASURED_POWER)
    if exclude is not None:
        t = t[t.epoch_model_name != exclude]
    lr = np.log(t.ratio_to_tdp.to_numpy(float))
    return ("lognormal", (float(lr.mean()), float(lr.std(ddof=1)), 0.3, 1.5))


def set_tpu():
    if MEASURED_POWER.exists():
        E.ASSUMPTIONS["tpu_system"] = tpu_spec()
    return E.ASSUMPTIONS["tpu_system"]


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    d = load()
    ts = set_tpu()
    print(f"TPU system-power factor: lognormal, median {np.exp(ts[1][0]):.3f} x TDP, sd_ln {ts[1][1]:.3f}")
    cal = mfu_calibration_check(d)        # uses the literature prior + LOO calibration
    cal.to_csv(TAB / "mfu_calibration_loo.csv", index=False)
    print("MFU calibration (LOO):\n", cal.round(3).to_string(index=False))
    # out-of-sample A vs B: literature prior (not fitted to these records)
    ab = ab_agreement(d)
    ab.to_csv(TAB / "method_a_vs_b.csv", index=False)
    st_ab = agreement_stats(ab.log10_ratio_B_A, ab.A_in_B_90pi)
    # sensitivity: totals under the literature prior, before switching
    lit, *_ = estimates(d)
    spec = set_mfu(d, "calibrated")
    mu, sg = spec[1][0], spec[1][1]
    print(f"calibrated MFU: lognormal, median {np.exp(mu):.3f}, 90% range {np.exp(mu - 1.645 * sg):.3f}-{np.exp(mu + 1.645 * sg):.3f}")
    pd.DataFrame([dict(mfu="calibrated lognormal", mu_ln=mu, sigma_ln=sg, median=np.exp(mu)),
                  dict(mfu="literature triangular(0.15,0.35,0.55)", mu_ln=np.nan, sigma_ln=np.nan, median=0.35)]
                 ).to_csv(TAB / "mfu_distribution.csv", index=False)
    est, e, c = estimates(d)
    est.to_csv(PROC / "estimates.csv", index=False)
    np.save(PROC / "mc_energy_kwh_sample.npy", e[:, :1000].astype(np.float32))  # thinned draws
    np.save(PROC / "mc_co2_t_sample.npy", c[:, :1000].astype(np.float32))

    ep = est[est.epoch_energy_mwh.gt(0)]
    st_ep = agreement_stats(np.log10(ep.energy_mwh_p50 / ep.epoch_energy_mwh))
    st_ep["share_epoch_inside_our_90pi"] = float(((ep.energy_mwh_p05 <= ep.epoch_energy_mwh) &
                                                  (ep.epoch_energy_mwh <= ep.energy_mwh_p95)).mean())
    pd.DataFrame([dict(comparison="Method B (literature MFU prior) vs Method A, independent records", **st_ab),
                  dict(comparison="This study (median) vs Epoch provisioned-power energy", **st_ep)]
                 ).to_csv(TAB / "agreement_summary.csv", index=False)

    st_lit = agreement_stats(np.log10(lit.energy_mwh_p50 / est.energy_mwh_p50))
    print(f"literature-prior vs calibrated estimates: median ratio {st_lit['median_ratio']:.2f}")
    pd.DataFrame([dict(comparison="literature-prior MFU vs calibrated MFU (all models)", **st_lit)]
                 ).to_csv(TAB / "sensitivity_mfu_prior.csv", index=False)
    log_variance_shares().to_csv(TAB / "sensitivity_log_variance.csv", index=False)
    sobol_check().to_csv(TAB / "sensitivity_sobol.csv", index=False)
    one_at_a_time(d).to_csv(TAB / "sensitivity_oat.csv", index=False)

    print("estimated models:", len(est), "| method A:", (est.method == "A").sum(), "B:", (est.method == "B").sum())
    print("A vs B:", {k: round(v, 3) for k, v in st_ab.items()})
    print("vs Epoch:", {k: round(v, 3) for k, v in st_ep.items()})
    top = est.sort_values("energy_mwh_p50", ascending=False).head(12)
    print(top[["model", "pub_date", "hw", "method", "energy_mwh_p05", "energy_mwh_p50", "energy_mwh_p95",
               "co2_t_p50", "epoch_energy_mwh"]].round(0).to_string(index=False))
    assert len(est) == int((d.method != "").sum()) and est.energy_mwh_p50.notna().all()


if __name__ == "__main__":
    main()
