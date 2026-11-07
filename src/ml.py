"""ML imputation of training energy from public metadata, with SHAP and conformal intervals.

Target: log10 median energy (MWh) from the Monte Carlo estimates.
Two feature sets: 'with_compute' (training FLOP known) and 'no_compute' (parameters/data/year only).
Run after estimate.py:  python -P src/ml.py
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, TAB = ROOT / "data/processed", ROOT / "results/tables"
SEED = 42
FEATURES = {
    "with_compute": ["log_flop", "log_params", "log_data", "year", "finetune", "industry", "academia",
                     "dom_language", "dom_vision", "dom_multimodal", "dom_games", "dom_bio", "us", "china"],
    "no_compute": ["log_params", "log_data", "year", "finetune", "industry", "academia",
                   "dom_language", "dom_vision", "dom_multimodal", "dom_games", "dom_bio", "us", "china"],
}


def features(d: pd.DataFrame) -> pd.DataFrame:
    dom = d.domain.fillna("").str.lower()
    return pd.DataFrame({
        "log_flop": np.log10(d.flop_run.where(d.flop_run > 0)),
        "log_params": np.log10(d.params.where(d.params > 0)),
        "log_data": np.log10(d.dataset_size.where(d.dataset_size > 0)),
        "year": d.year_frac, "finetune": d.is_finetune.astype(int),
        "industry": d.org_type.eq("Industry").astype(int), "academia": d.org_type.eq("Academia").astype(int),
        "dom_language": dom.str.contains("language").astype(int), "dom_vision": dom.str.contains("vision|image").astype(int),
        "dom_multimodal": dom.str.contains("multimodal").astype(int), "dom_games": dom.str.contains("games").astype(int),
        "dom_bio": dom.str.contains("biology|medicine").astype(int),
        "us": d.country.eq("United States").astype(int), "china": d.country.eq("China").astype(int),
    }, index=d.index)


def models():
    import lightgbm as lgb
    import xgboost as xgb
    return {
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, random_state=SEED, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=400, num_leaves=15, learning_rate=0.05, min_child_samples=10,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1),
    }


def impute_median(Xtr, Xte):
    med = Xtr.median()
    return Xtr.fillna(med), Xte.fillna(med)


def physics_baseline(Xtr, ytr, Xte):
    """log E = log FLOP + (a + b*year): energy-per-FLOP trend fitted on the training fold."""
    jpf = ytr - Xtr.log_flop
    b, a = np.polyfit(Xtr.year, jpf, 1)
    return Xte.log_flop + a + b * Xte.year


def score(y, p):
    err = p - y
    return dict(mae_log10=np.mean(np.abs(err)), rmse_log10=np.sqrt(np.mean(err ** 2)),
                within_2x=np.mean(np.abs(err) <= np.log10(2)), within_3x=np.mean(np.abs(err) <= np.log10(3)),
                r2=1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2),
                median_ape_pct=100 * np.median(np.abs(10 ** err - 1)))


def cross_validate(X, y, fs, n_splits=5, n_repeats=5):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    names = list(models()) + (["Physics baseline"] if fs == "with_compute" else [])
    preds = {m: np.zeros((n_repeats, len(y))) for m in names}
    for k, (tr, te) in enumerate(rkf.split(X)):
        r = k // n_splits
        Xtr, Xte = impute_median(X.iloc[tr], X.iloc[te])
        for m, est in models().items():
            preds[m][r, te] = est.fit(Xtr, y.iloc[tr]).predict(Xte)
        if fs == "with_compute":
            preds["Physics baseline"][r, te] = physics_baseline(Xtr, y.iloc[tr], Xte)
    rows = []
    for m, P in preds.items():
        s = pd.DataFrame([score(y.to_numpy(), P[r]) for r in range(n_repeats)])
        rows.append(dict(feature_set=fs, model=m, n=len(y), **{f"{c}": s[c].mean() for c in s},
                         **{f"{c}_sd": s[c].std() for c in ["mae_log10", "within_2x", "r2"]}))
    return pd.DataFrame(rows), preds


def main():
    import shap
    TAB.mkdir(parents=True, exist_ok=True)
    allm = pd.read_csv(PROC / "models_energy.csv", parse_dates=["pub_date"])
    est = pd.read_csv(PROC / "estimates.csv", parse_dates=["pub_date"])
    est_y = np.log10(est.set_index("model").energy_mwh_p50.groupby(level=0).first())
    allm["y"] = allm.model.map(est_y)
    X_all = features(allm)

    out, shap_rows, imputed = [], [], []
    for fs, cols in FEATURES.items():
        need = ["log_flop"] if fs == "with_compute" else ["log_params"]
        lab = allm.y.notna() & X_all[need].notna().all(axis=1)
        X, y = X_all.loc[lab, cols], allm.loc[lab, "y"]
        cv, preds = cross_validate(X, y, fs)
        out.append(cv)
        best = cv[cv.model != "Physics baseline"].sort_values("mae_log10").iloc[0].model

        # split-conformal 90% interval from out-of-fold absolute residuals of the best model
        resid = np.abs(preds[best].mean(0) - y.to_numpy())
        qhat = np.quantile(resid, min(1.0, np.ceil(0.9 * (len(y) + 1)) / len(y)))
        est_m = models()[best]
        Xf = X.fillna(X.median())
        est_m.fit(Xf, y)

        if best == "Ridge":
            sc, rg = est_m.steps[0][1], est_m.steps[1][1]
            sv = shap.LinearExplainer(rg, sc.transform(Xf)).shap_values(sc.transform(Xf))
        else:
            sv = shap.TreeExplainer(est_m).shap_values(Xf)
        np.save(PROC / f"shap_values_{fs}.npy", sv)
        Xf.to_csv(PROC / f"shap_X_{fs}.csv", index=False)
        for c, v in zip(cols, np.abs(sv).mean(0)):
            shap_rows.append(dict(feature_set=fs, model=best, feature=c, mean_abs_shap=v))

        # impute for models without an estimate but with the needed features
        tgt = allm.y.isna() & X_all[need].notna().all(axis=1)
        if fs == "no_compute":
            tgt &= X_all.log_flop.isna()          # models with compute are handled by with_compute
        Xt = X_all.loc[tgt, cols].fillna(X.median())
        p = est_m.predict(Xt) if len(Xt) else np.array([])
        imputed.append(pd.DataFrame({"model": allm.loc[tgt, "model"], "pub_date": allm.loc[tgt, "pub_date"],
                                     "feature_set": fs, "ml_model": best, "pred_log10_mwh": p,
                                     "pi90_lo_mwh": 10 ** (p - qhat), "pred_mwh": 10 ** p, "pi90_hi_mwh": 10 ** (p + qhat),
                                     "conformal_halfwidth_log10": qhat, "is_finetune": allm.loc[tgt, "is_finetune"],
                                     "frontier": allm.loc[tgt, "frontier"]}))
        print(f"[{fs}] n={len(y)} best={best} conformal 90% half-width = x{10 ** qhat:.2f}; imputed {len(Xt)} models")

    cv = pd.concat(out, ignore_index=True)
    cv.to_csv(TAB / "ml_cv_scores.csv", index=False)
    pd.DataFrame(shap_rows).sort_values(["feature_set", "mean_abs_shap"], ascending=[True, False]).to_csv(TAB / "shap_importance.csv", index=False)
    imp = pd.concat(imputed, ignore_index=True)
    imp.to_csv(PROC / "imputed_energy.csv", index=False)
    pd.set_option("display.width", 200)
    print(cv[["feature_set", "model", "n", "mae_log10", "within_2x", "within_3x", "r2", "median_ape_pct"]].round(3).to_string(index=False))
    print(pd.read_csv(TAB / "shap_importance.csv").groupby("feature_set").head(6).round(3).to_string(index=False))
    print("coverage: estimated", int(allm.y.notna().sum()), "+ imputed", len(imp), "=", int(allm.y.notna().sum()) + len(imp), "of", len(allm))


if __name__ == "__main__":
    main()
