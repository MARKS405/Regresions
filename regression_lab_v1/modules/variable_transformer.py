from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, List, Dict, Optional


def infer_numeric_columns(df: pd.DataFrame, date_col: str = "Fecha", entity_col: Optional[str] = None) -> List[str]:
    exclude = {date_col}
    if entity_col:
        exclude.add(entity_col)
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def infer_dummy_columns(df: pd.DataFrame, numeric_cols: Iterable[str]) -> List[str]:
    dummies = []
    for c in numeric_cols:
        vals = set(pd.Series(df[c]).dropna().unique().tolist())
        if vals and vals.issubset({0, 1, 0.0, 1.0}):
            dummies.append(c)
    return dummies


def _group_series(df: pd.DataFrame, col: str, entity_col: Optional[str]):
    if entity_col and entity_col in df.columns:
        return df.groupby(entity_col, sort=False)[col]
    return df[col]


def _shift(df: pd.DataFrame, col: str, entity_col: Optional[str], periods: int = 1):
    if entity_col and entity_col in df.columns:
        return df.groupby(entity_col, sort=False)[col].shift(periods)
    return df[col].shift(periods)


def _diff(df: pd.DataFrame, col: str, entity_col: Optional[str], periods: int = 1):
    if entity_col and entity_col in df.columns:
        return df.groupby(entity_col, sort=False)[col].diff(periods)
    return df[col].diff(periods)


def generate_event_dummies(
    df: pd.DataFrame,
    events: List[Dict[str, str]],
    date_col: str = "Fecha",
) -> pd.DataFrame:
    """Create event dummies from a list of dicts: {name,start,end}."""
    out = df.copy()
    if date_col not in out.columns:
        return out
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    for event in events:
        name = str(event.get("name", "event")).strip().replace(" ", "_")
        start = pd.to_datetime(event.get("start"), errors="coerce")
        end = pd.to_datetime(event.get("end"), errors="coerce")
        if name and not pd.isna(start) and not pd.isna(end):
            out[name] = ((out[date_col] >= start) & (out[date_col] <= end)).astype(int)
    return out


def generate_transformations(
    df: pd.DataFrame,
    date_col: str = "Fecha",
    entity_col: Optional[str] = None,
    dummy_vars: Optional[Iterable[str]] = None,
    include_ma: bool = False,
    include_lags_3_6_12: bool = False,
) -> pd.DataFrame:
    """Generate transformations for numeric variables.

    Logs are not generated for dummy variables. Invalid logs remain NaN.
    """
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    sort_cols = [c for c in [entity_col, date_col] if c and c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    numeric_cols = infer_numeric_columns(out, date_col=date_col, entity_col=entity_col)
    dummy_vars = set(dummy_vars or [])

    for v in numeric_cols:
        # Always generate simple diff and lag for numeric variables, including dummies.
        if f"{v}_lag1" not in out.columns:
            out[f"{v}_lag1"] = _shift(out, v, entity_col, 1)
        if f"d_{v}" not in out.columns:
            out[f"d_{v}"] = _diff(out, v, entity_col, 1)
        if f"d_{v}_lag1" not in out.columns:
            out[f"d_{v}_lag1"] = _shift(out, f"d_{v}", entity_col, 1)
        if f"d_{v}_12" not in out.columns:
            out[f"d_{v}_12"] = _diff(out, v, entity_col, 12)

        if include_lags_3_6_12:
            for lag in [3, 6, 12]:
                out[f"{v}_lag{lag}"] = _shift(out, v, entity_col, lag)
                out[f"d_{v}_lag{lag}"] = _shift(out, f"d_{v}", entity_col, lag)

        if include_ma:
            if entity_col and entity_col in out.columns:
                for w in [3, 6, 12]:
                    out[f"ma{w}_{v}"] = out.groupby(entity_col, sort=False)[v].transform(lambda s: s.rolling(w, min_periods=1).mean())
            else:
                for w in [3, 6, 12]:
                    out[f"ma{w}_{v}"] = out[v].rolling(w, min_periods=1).mean()

        if v in dummy_vars:
            continue

        positive = out[v].where(out[v] > 0)
        if f"ln_{v}" not in out.columns:
            out[f"ln_{v}"] = np.log(positive)
        if f"ln_{v}_lag1" not in out.columns:
            out[f"ln_{v}_lag1"] = _shift(out, f"ln_{v}", entity_col, 1)
        if f"d_ln_{v}" not in out.columns:
            out[f"d_ln_{v}"] = _diff(out, f"ln_{v}", entity_col, 1)
        if f"d_ln_{v}_lag1" not in out.columns:
            out[f"d_ln_{v}_lag1"] = _shift(out, f"d_ln_{v}", entity_col, 1)
        if f"d_ln_{v}_12" not in out.columns:
            out[f"d_ln_{v}_12"] = _diff(out, f"ln_{v}", entity_col, 12)

        if include_lags_3_6_12:
            for lag in [3, 6, 12]:
                out[f"ln_{v}_lag{lag}"] = _shift(out, f"ln_{v}", entity_col, lag)
                out[f"d_ln_{v}_lag{lag}"] = _shift(out, f"d_ln_{v}", entity_col, lag)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def create_interaction(df: pd.DataFrame, var_a: str, var_b: str, name: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    if var_a in out.columns and var_b in out.columns:
        col = name or f"{var_a}_x_{var_b}"
        out[col] = out[var_a] * out[var_b]
    return out
