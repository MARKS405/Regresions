from __future__ import annotations

import pandas as pd
from typing import Iterable, Optional


def apply_counterfactual(
    df: pd.DataFrame,
    pred_col: str,
    params: pd.Series,
    variables_to_remove: Iterable[str],
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    variables_to_remove = list(variables_to_remove)
    output_col = output_col or f"{pred_col}_cf"
    out[output_col] = out[pred_col]
    for v in variables_to_remove:
        if v in out.columns and v in params.index:
            out[output_col] = out[output_col] - out[v] * params[v]
    return out


def contribution_table(df: pd.DataFrame, params: pd.Series, x_vars: Iterable[str]) -> pd.DataFrame:
    rows = []
    for v in x_vars:
        if v in df.columns and v in params.index:
            contrib = df[v] * params[v]
            rows.append({
                "variable": v,
                "coef": params[v],
                "contrib_promedio": contrib.mean(),
                "contrib_abs_promedio": contrib.abs().mean(),
                "contrib_min": contrib.min(),
                "contrib_max": contrib.max(),
            })
    return pd.DataFrame(rows).sort_values("contrib_abs_promedio", ascending=False) if rows else pd.DataFrame()
