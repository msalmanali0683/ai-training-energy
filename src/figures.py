"""Paper figures (PNG 300 dpi + PDF) from processed data and result tables.

Run last:  python -P src/figures.py
Palette: validated categorical slots (blue, orange, aqua, yellow) + neutral gray; direct labels
are added wherever a low-contrast slot is used.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, TAB, FIG = ROOT / "data/processed", ROOT / "results/tables", ROOT / "results/figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def year_axis(ax, step=2):
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        if ax.get_xscale() == "linear" and 2000 < ax.get_xlim()[0] < 2100:
            year_axis(ax, 4 if ax.get_xlim()[1] - ax.get_xlim()[0] > 14 else 2)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fit_line(ax, x, y, color, label=None):
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, 10 ** (a + b * xs), color=color, lw=2, label=label)


def fig_trends(d, est, dt):
    base = d[~d.is_finetune & d.flop.gt(0)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    ax = axs[0]
    o, f = base[~base.frontier], base[base.frontier]
    ax.scatter(o.year_frac, o.flop, s=10, color=GRAY, alpha=.6, lw=0, label="Other notable")
    ax.scatter(f.year_frac, f.flop, s=14, color=BLUE, lw=.4, edgecolor="white", label="Frontier")
    fit_line(ax, f.year_frac, np.log10(f.flop), BLUE)
    r = dt[(dt.quantity == "training compute (FLOP)") & (dt.group == "frontier")].iloc[0]
    ax.text(.03, .95, f"Frontier: ×{r.growth_x_per_yr:.1f}/yr\ndoubling {r.doubling_months:.1f} mo "
            f"[{r.doubling_ci_lo:.1f}–{r.doubling_ci_hi:.1f}]", transform=ax.transAxes, va="top", color=INK)
    ax.set(yscale="log", ylabel="Training compute (FLOP)", title="a  Training compute")
    ax.legend(loc="lower right")

    ax = axs[1]
    e = est[~est.is_finetune]
    for sub, col, lab, s in [(e[~e.frontier], GRAY, "Other notable", 10), (e[e.frontier], BLUE, "Frontier", 14)]:
        ax.errorbar(sub.year_frac, sub.energy_mwh_p50,
                    yerr=[sub.energy_mwh_p50 - sub.energy_mwh_p05, sub.energy_mwh_p95 - sub.energy_mwh_p50],
                    fmt="none", ecolor=col, alpha=.35, lw=.7)
        ax.scatter(sub.year_frac, sub.energy_mwh_p50, s=s, color=col, lw=.4, edgecolor="white", label=lab)
    ef = e[e.frontier]
    fit_line(ax, ef.year_frac, np.log10(ef.energy_mwh_p50), BLUE)
    r = dt[(dt.quantity == "training energy (kWh)") & (dt.group == "frontier")].iloc[0]
    ax.text(.03, .95, f"Frontier: ×{r.growth_x_per_yr:.1f}/yr\ndoubling {r.doubling_months:.1f} mo "
            f"[{r.doubling_ci_lo:.1f}–{r.doubling_ci_hi:.1f}]", transform=ax.transAxes, va="top", color=INK)
    ax.set(yscale="log", ylabel="Training energy (MWh), median & 90% interval", title="b  Training energy")
    ax.legend(loc="lower right")
    for a in axs:
        a.set_xlabel("Publication date")
    save(fig, "fig1_trends")


def fig_efficiency(est, hw, dt):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    ax = axs[0]
    for prec, col in [("FP32", GRAY), ("Tensor FP16/BF16", BLUE)]:
        s = hw[hw.precision == prec]
        r = dt[dt.quantity == f"hardware FLOP/s per W ({prec})"].iloc[0]
        ax.scatter(s.year, s.flops_per_w, s=12, color=col, lw=.4, edgecolor="white",
                   label=f"{prec} (doubling {r.doubling_months / 12:.1f} yr)")
        fit_line(ax, s.year, np.log10(s.flops_per_w), col)
    ax.set(yscale="log", xlabel="Accelerator release date", ylabel="Peak FLOP/s per W (TDP)", title="a  Accelerator efficiency")
    ax.legend(loc="lower right")
    ax = axs[1]
    e = est[~est.is_finetune & est.flop_run.gt(0)]
    jpf = e.energy_mwh_p50 * 3.6e9 / e.flop_run
    for m, col, lab in [(~e.frontier, GRAY, "Other notable"), (e.frontier, BLUE, "Frontier")]:
        ax.scatter(e.year_frac[m], jpf[m], s=12, color=col, lw=.4, edgecolor="white", label=lab)
    fit_line(ax, e.year_frac[e.frontier], np.log10(jpf[e.frontier]), BLUE)
    r = dt[(dt.quantity == "energy per FLOP (J/FLOP)") & (dt.group == "frontier")].iloc[0]
    ax.text(.97, .95, f"Frontier: halving {-r.doubling_months:.0f} mo", transform=ax.transAxes, ha="right", va="top", color=INK)
    ax.set(yscale="log", xlabel="Publication date", ylabel="Effective energy per FLOP (J/FLOP)", title="b  Realised energy per FLOP")
    ax.legend(loc="lower left")
    save(fig, "fig2_efficiency")


def loglog_compare(ax, x, y, xlabel, ylabel, title, note):
    lo, hi = np.nanmin([x.min(), y.min()]) / 2, np.nanmax([x.max(), y.max()]) * 2
    xs = np.array([lo, hi])
    ax.fill_between(xs, xs / 2, xs * 2, color=GRID, alpha=.6, lw=0, label="within 2×")
    ax.plot(xs, xs, color=INK2, lw=1, label="1:1")
    ax.scatter(x, y, s=14, color=BLUE, lw=.4, edgecolor="white")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel=xlabel, ylabel=ylabel, title=title)
    ax.text(.03, .97, note, transform=ax.transAxes, va="top", color=INK)
    ax.legend(loc="lower right")


def fig_validation(ab, est, agree, val=None):
    n = 3 if val is not None and len(val) else 2
    fig, axs = plt.subplots(1, n, figsize=(3.4 * n, 3.2), constrained_layout=True)
    r = agree.iloc[0]
    loglog_compare(axs[0], ab.energy_A_mwh, ab.energy_B_mwh, "Method A energy (MWh)", "Method B energy (MWh)",
                   "a  Method B vs Method A", f"n = {int(r.n)} independent records\nmedian ratio {r.median_ratio:.2f}\nwithin 2×: {100 * r.within_2x:.0f}%")
    ep = est[est.epoch_energy_mwh.gt(0)]
    r = agree.iloc[1]
    loglog_compare(axs[1], ep.epoch_energy_mwh, ep.energy_mwh_p50, "Epoch provisioned power × time (MWh)", "This study, median (MWh)",
                   "b  vs Epoch AI power field", f"n = {int(r.n)}\nmedian ratio {r.median_ratio:.2f}")
    if n == 3:
        loglog_compare(axs[2], val.reported, val.estimated, "Reported (MWh)", "This study, median (MWh)",
                       "c  vs reported values", f"n = {len(val)}\nmedian ratio {np.median(val.estimated / val.reported):.2f}")
    save(fig, "fig3_validation")


def fig_sensitivity(lv, oat):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    names = {"mfu": "MFU", "load": "Power load factor", "overhead_dc": "Server overhead", "pue": "PUE", "ci_rel_sd": "Grid intensity"}
    s = lv[lv.method.str.startswith("Method B")].assign(f=lambda t: t.factor.map(names)).sort_values("share_co2")
    axs[0].barh(s.f, 100 * s.share_co2, color=BLUE, height=.6)
    for y, v in enumerate(100 * s.share_co2):
        axs[0].text(v + 1, y, f"{v:.0f}%", va="center", color=INK)
    axs[0].set(xlabel="Share of variance of log CO₂ (%)", title="a  Variance shares (Method B)")
    axs[0].set_xlim(0, 100 * s.share_co2.max() * 1.2)
    axs[0].grid(axis="y", visible=False)
    o = oat.assign(f=lambda t: t.factor.map(names))
    order = o.groupby("f").total_energy_change_pct.apply(lambda v: v.abs().max()).sort_values().index
    for i, f in enumerate(order):
        for _, r in o[o.f == f].iterrows():
            col = AQUA if r.setting == "low" else ORANGE
            v = r.total_energy_change_pct
            axs[1].barh(i, v, color=col, height=.6)
            axs[1].text(v + (2 if v > 0 else -2), i, f"{r.setting} {r.value:g}", va="center",
                        ha="left" if v > 0 else "right", color=INK, fontsize=7.5)
    axs[1].set_yticks(range(len(order)), order)
    axs[1].axvline(0, color=INK2, lw=1)
    axs[1].set(xlabel="Change in total estimated energy (%)", title="b  One-at-a-time bounds")
    axs[1].grid(axis="y", visible=False)
    lim = o.total_energy_change_pct.abs().max() * 1.45
    axs[1].set_xlim(-lim * .6, lim)
    save(fig, "fig4_sensitivity")


def fig_forecast(fc):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    cols = {"Business as usual": BLUE, "Slowdown after 2026": AQUA, "Efficient & clean": YELLOW, "Dirty grid": ORANGE}
    for i, r in fc.reset_index(drop=True).iterrows():
        ax.plot([r.energy_gwh_p05, r.energy_gwh_p95], [i, i], color=cols[r.scenario], lw=2)
        ax.scatter(r.energy_gwh_p50, i, s=40, color=cols[r.scenario], edgecolor="white", lw=1, zorder=3)
        ax.text(r.energy_gwh_p95 * 1.15, i, f"{r.energy_gwh_p50 / 1e3:.1f} TWh, {r.avg_power_mw_p50 / 1e3:.1f} GW\n{r.co2_kt_p50 / 1e3:.1f} Mt CO₂",
                va="center", color=INK, fontsize=7.5)
    ax.set_yticks(range(len(fc)), fc.scenario)
    ax.invert_yaxis()
    ax.set(xscale="log", xlabel="Energy of largest training run in 2030 (GWh)", title="a  2030 scenarios (median, 90% interval)")
    ax.set_xlim(fc.energy_gwh_p05.min() / 2, fc.energy_gwh_p95.max() * 12)
    ax.grid(axis="y", visible=False)
    ax = axs[1]
    lc = pd.read_csv(TAB / "location_experiment.csv").sort_values("co2_kt_bau_2030_run")
    ax.barh(lc.country, lc.co2_kt_bau_2030_run / 1e3, color=BLUE, height=.6)
    for y, (v, g) in enumerate(zip(lc.co2_kt_bau_2030_run / 1e3, lc.ci_g_kwh)):
        ax.text(v + .2, y, f"{v:.1f}  ({g:.0f} g/kWh)", va="center", color=INK, fontsize=7.5)
    ax.set(xlabel="CO₂ of BAU 2030 run (Mt)", title="b  Same run, different grids")
    ax.set_xlim(0, lc.co2_kt_bau_2030_run.max() / 1e3 * 1.45)
    ax.grid(axis="y", visible=False)
    save(fig, "fig5_forecast_location")


def fig_shap(sh, cv):
    names = {"log_flop": "log training compute", "log_params": "log parameters", "log_data": "log dataset size",
             "year": "publication year", "finetune": "fine-tune", "industry": "industry org.", "academia": "academic org.",
             "dom_language": "language domain", "dom_vision": "vision domain", "dom_multimodal": "multimodal",
             "dom_games": "games domain", "dom_bio": "biology domain", "us": "US organisation", "china": "China organisation"}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for ax, fs, title in [(axs[0], "with_compute", "a  Compute known"), (axs[1], "no_compute", "b  Compute unknown")]:
        s = sh[sh.feature_set == fs].head(7).iloc[::-1]
        ax.barh(s.feature.map(names), s.mean_abs_shap, color=BLUE, height=.6)
        c = cv[(cv.feature_set == fs) & (cv.model == s.model.iloc[0])].iloc[0]
        ax.set(xlabel="mean |SHAP| (log10 MWh)", title=f"{title}\n{s.model.iloc[0]}; CV within 2×: {100 * c.within_2x:.0f}%")
        ax.grid(axis="y", visible=False)
    save(fig, "fig6_shap")


def main():
    d = pd.read_csv(PROC / "models_energy.csv", parse_dates=["pub_date"])
    est = pd.read_csv(PROC / "estimates.csv", parse_dates=["pub_date"])
    dt = pd.read_csv(TAB / "doubling_times.csv")
    fig_trends(d, est, dt)
    fig_efficiency(est, pd.read_csv(TAB / "hardware_efficiency.csv"), dt)
    val = pd.read_csv(TAB / "validation_points.csv") if (TAB / "validation_points.csv").exists() else None
    fig_validation(pd.read_csv(TAB / "method_a_vs_b.csv"), est, pd.read_csv(TAB / "agreement_summary.csv"), val)
    fig_sensitivity(pd.read_csv(TAB / "sensitivity_log_variance.csv"), pd.read_csv(TAB / "sensitivity_oat.csv"))
    fig_forecast(pd.read_csv(TAB / "forecast_2030.csv"))
    fig_shap(pd.read_csv(TAB / "shap_importance.csv"), pd.read_csv(TAB / "ml_cv_scores.csv"))
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
