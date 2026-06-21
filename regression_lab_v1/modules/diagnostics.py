from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Optional

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "variable": df.columns,
        "missing": df.isna().sum().values,
        "missing_pct": (df.isna().mean().values * 100).round(2),
        "dtype": [str(df[c].dtype) for c in df.columns],
    })
    return out.sort_values("missing_pct", ascending=False)


def descriptive_summary(df: pd.DataFrame, cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    cols = list(cols or df.select_dtypes(include=[np.number]).columns)
    if not cols:
        return pd.DataFrame()
    return df[cols].describe().T.reset_index().rename(columns={"index": "variable"})


def correlation_matrix(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr()


def adf_tests(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.nunique() <= 1 or len(s) < 12:
            rows.append({"variable": col, "adf_stat": np.nan, "p_value": np.nan, "conclusion": "muestra insuficiente"})
            continue
        try:
            res = adfuller(s, autolag="AIC")
            p = res[1]
            rows.append({
                "variable": col,
                "adf_stat": round(res[0], 4),
                "p_value": round(p, 4),
                "conclusion": "estacionaria" if p < 0.05 else "no estacionaria",
            })
        except Exception as exc:
            rows.append({"variable": col, "adf_stat": np.nan, "p_value": np.nan, "conclusion": f"error: {exc}"})
    return pd.DataFrame(rows)


def vif_table(df: pd.DataFrame, x_vars: Iterable[str], add_constant: bool = True) -> pd.DataFrame:
    x_vars = [c for c in x_vars if c in df.columns]
    if len(x_vars) < 2:
        return pd.DataFrame()
    X = df[x_vars].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty or len(X) <= len(x_vars):
        return pd.DataFrame({"variable": x_vars, "VIF": np.nan, "alerta": "muestra insuficiente"})
    if add_constant:
        X_eval = sm.add_constant(X, has_constant="add")
    else:
        X_eval = X
    rows = []
    for i, col in enumerate(X_eval.columns):
        try:
            val = variance_inflation_factor(X_eval.values, i)
        except Exception:
            val = np.nan
        rows.append({"variable": col, "VIF": val})
    out = pd.DataFrame(rows)
    out["VIF"] = out["VIF"].replace([np.inf, -np.inf], np.nan).round(3)
    out["alerta"] = np.where(out["VIF"] >= 10, "alto", np.where(out["VIF"] >= 5, "moderado", "ok"))
    return out


def residual_diagnostics_ols(results, X) -> pd.DataFrame:
    rows = []
    resid = results.resid
    try:
        rows.append({"test": "Durbin-Watson", "stat": float(durbin_watson(resid)), "p_value": np.nan, "lectura": "cerca de 2 sugiere baja autocorrelación"})
    except Exception as exc:
        rows.append({"test": "Durbin-Watson", "stat": np.nan, "p_value": np.nan, "lectura": f"error: {exc}"})
    try:
        bp = het_breuschpagan(resid, X)
        rows.append({"test": "Breusch-Pagan", "stat": float(bp[0]), "p_value": float(bp[1]), "lectura": "p<0.05 sugiere heterocedasticidad"})
    except Exception as exc:
        rows.append({"test": "Breusch-Pagan", "stat": np.nan, "p_value": np.nan, "lectura": f"error: {exc}"})
    try:
        bg = acorr_breusch_godfrey(results, nlags=min(12, max(1, int(len(resid) / 5))))
        rows.append({"test": "Breusch-Godfrey", "stat": float(bg[0]), "p_value": float(bg[1]), "lectura": "p<0.05 sugiere autocorrelación"})
    except Exception as exc:
        rows.append({"test": "Breusch-Godfrey", "stat": np.nan, "p_value": np.nan, "lectura": f"error: {exc}"})
    return pd.DataFrame(rows)


def panel_balance(df: pd.DataFrame, entity_col: str = "Banco", date_col: str = "Fecha") -> pd.DataFrame:
    if entity_col not in df.columns or date_col not in df.columns:
        return pd.DataFrame()
    counts = df.groupby(entity_col)[date_col].nunique().reset_index(name="n_periodos")
    counts["panel_balanceado"] = counts["n_periodos"].eq(counts["n_periodos"].max())
    return counts
