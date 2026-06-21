from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Dict, Any
import statsmodels.api as sm

try:
    from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
    LINEARMODELS_AVAILABLE = True
except Exception:
    LINEARMODELS_AVAILABLE = False


def prepare_panel_data(
    df: pd.DataFrame,
    y_var: str,
    x_vars: Iterable[str],
    entity_col: str = "Banco",
    date_col: str = "Fecha",
) -> pd.DataFrame:
    x_vars = list(x_vars)
    required = [entity_col, date_col, y_var] + x_vars
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Variables no encontradas: {missing}")
    data = df[required].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    for c in [y_var] + x_vars:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_values([entity_col, date_col]).set_index([entity_col, date_col])
    return data


def run_panel_model(
    df: pd.DataFrame,
    y_var: str,
    x_vars: Iterable[str],
    model_kind: str = "Two-way FE",
    add_constant: bool = False,
    cov_type: str = "clustered",
    cluster_entity: bool = True,
    entity_col: str = "Banco",
    date_col: str = "Fecha",
) -> Dict[str, Any]:
    if not LINEARMODELS_AVAILABLE:
        raise ImportError("Falta instalar linearmodels. Ejecuta: pip install linearmodels")
    x_vars = list(x_vars)
    data = prepare_panel_data(df, y_var, x_vars, entity_col, date_col)
    y = data[y_var]
    X = data[x_vars]
    if add_constant:
        X = sm.add_constant(X, has_constant="add")

    if model_kind == "Pooled OLS":
        model = PooledOLS(y, X, check_rank=False)
    elif model_kind == "Random Effects":
        model = RandomEffects(y, X, check_rank=False)
    else:
        entity_effects = model_kind in ["FE entidad", "Two-way FE"]
        time_effects = model_kind in ["FE tiempo", "Two-way FE"]
        model = PanelOLS(
            y,
            X,
            entity_effects=entity_effects,
            time_effects=time_effects,
            drop_absorbed=True,
            check_rank=False,
        )

    fit_kwargs = {}
    if cov_type == "clustered":
        fit_kwargs = {"cov_type": "clustered", "cluster_entity": cluster_entity}
    elif cov_type == "robust":
        fit_kwargs = {"cov_type": "robust"}
    else:
        fit_kwargs = {"cov_type": "unadjusted"}
    res = model.fit(**fit_kwargs)

    params = res.params
    common_x = [c for c in X.columns if c in params.index]
    pred = X[common_x].mul(params[common_x], axis=1).sum(axis=1).rename(f"pred_{y_var}")

    return {"results": res, "data": data, "X": X, "y": y, "pred": pred}


def coef_table_panel(results) -> pd.DataFrame:
    params = results.params
    std = results.std_errors
    tstats = results.tstats
    pvals = results.pvalues
    out = pd.DataFrame({
        "variable": params.index,
        "coef": params.values,
        "std_err": std.reindex(params.index).values,
        "t_stat": tstats.reindex(params.index).values,
        "p_value": pvals.reindex(params.index).values,
    })
    out["signif"] = np.select(
        [out["p_value"] < 0.01, out["p_value"] < 0.05, out["p_value"] < 0.10],
        ["***", "**", "*"],
        default=""
    )
    return out


def panel_metrics(results, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    common = y_true.index.intersection(y_pred.index)
    err = y_true.loc[common] - y_pred.loc[common]
    rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan
    r2 = getattr(results, "rsquared", np.nan)
    return {
        "nobs": float(getattr(results, "nobs", np.nan)),
        "rmse": rmse,
        "r2": float(r2) if r2 is not None else np.nan,
        "r2_adj": np.nan,
        "aic": np.nan,
        "bic": np.nan,
    }
