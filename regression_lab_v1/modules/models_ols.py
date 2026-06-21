from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Iterable, Dict, Any


def prepare_model_data(df: pd.DataFrame, y_var: str, x_vars: Iterable[str]) -> pd.DataFrame:
    cols = [y_var] + list(x_vars)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Variables no encontradas: {missing}")
    data = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return data


def run_ols(
    df: pd.DataFrame,
    y_var: str,
    x_vars: Iterable[str],
    add_constant: bool = True,
    cov_type: str = "HAC",
    hac_lags: int = 3,
) -> Dict[str, Any]:
    x_vars = list(x_vars)
    data = prepare_model_data(df, y_var, x_vars)
    y = data[y_var]
    X = data[x_vars]
    if add_constant:
        X = sm.add_constant(X, has_constant="add")
    if cov_type == "HAC":
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
    elif cov_type == "HC1":
        res = sm.OLS(y, X).fit(cov_type="HC1")
    elif cov_type == "HC3":
        res = sm.OLS(y, X).fit(cov_type="HC3")
    else:
        res = sm.OLS(y, X).fit()
    pred = pd.Series(res.predict(X), index=data.index, name=f"pred_{y_var}")
    return {"results": res, "data": data, "X": X, "y": y, "pred": pred}


def coef_table(results) -> pd.DataFrame:
    return pd.DataFrame({
        "variable": results.params.index,
        "coef": results.params.values,
        "std_err": results.bse.values,
        "t_stat": results.tvalues.values,
        "p_value": results.pvalues.values,
    }).assign(
        signif=lambda d: np.select(
            [d["p_value"] < 0.01, d["p_value"] < 0.05, d["p_value"] < 0.10],
            ["***", "**", "*"],
            default=""
        )
    )


def model_metrics(results, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    common = y_true.index.intersection(y_pred.index)
    err = y_true.loc[common] - y_pred.loc[common]
    rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan
    return {
        "nobs": float(getattr(results, "nobs", np.nan)),
        "rmse": rmse,
        "r2": float(getattr(results, "rsquared", np.nan)),
        "r2_adj": float(getattr(results, "rsquared_adj", np.nan)),
        "aic": float(getattr(results, "aic", np.nan)),
        "bic": float(getattr(results, "bic", np.nan)),
    }
