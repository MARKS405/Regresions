from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def _reconstruct_group(g: pd.DataFrame, target_var: str, dep_var: str, pred_col: str) -> pd.Series:
    g = g.copy()
    pred = g[pred_col]
    actual = g[target_var] if target_var in g.columns else pd.Series(np.nan, index=g.index)

    if dep_var == target_var:
        return pred
    if dep_var == f"ln_{target_var}":
        return np.exp(pred)
    if dep_var == f"d_{target_var}":
        out = pd.Series(np.nan, index=g.index, dtype=float)
        if actual.dropna().empty:
            return out
        first_idx = actual.dropna().index[0]
        running = actual.loc[first_idx]
        for idx in g.index:
            if idx == first_idx or pd.isna(pred.loc[idx]):
                out.loc[idx] = running
            else:
                running = running + pred.loc[idx]
                out.loc[idx] = running
        return out
    if dep_var == f"d_ln_{target_var}":
        out = pd.Series(np.nan, index=g.index, dtype=float)
        if actual.dropna().empty:
            return out
        first_idx = actual.dropna().index[0]
        running = actual.loc[first_idx]
        for idx in g.index:
            if idx == first_idx or pd.isna(pred.loc[idx]):
                out.loc[idx] = running
            else:
                running = running * np.exp(pred.loc[idx])
                out.loc[idx] = running
        return out
    if dep_var == f"d_{target_var}_12":
        lag12 = actual.shift(12)
        return lag12 + pred
    if dep_var == f"d_ln_{target_var}_12":
        lag12 = actual.shift(12)
        return lag12 * np.exp(pred)
    # Fallback: return model scale if no known inverse mapping.
    return pred


def reconstruct_target(
    df: pd.DataFrame,
    target_var: str,
    dep_var: str,
    pred_col: str,
    entity_col: Optional[str] = None,
    date_col: str = "Fecha",
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    output_col = output_col or f"estimado_{target_var}"
    if pred_col not in out.columns:
        out[output_col] = np.nan
        return out
    sort_cols = [c for c in [entity_col, date_col] if c and c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).copy()
    if entity_col and entity_col in out.columns:
        pieces = []
        for _, g in out.groupby(entity_col, sort=False):
            s = _reconstruct_group(g, target_var, dep_var, pred_col)
            pieces.append(s)
        out[output_col] = pd.concat(pieces).sort_index()
    else:
        out[output_col] = _reconstruct_group(out, target_var, dep_var, pred_col)
    return out
