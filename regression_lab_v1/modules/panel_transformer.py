from __future__ import annotations

import pandas as pd
from typing import Iterable, Optional


def transform_panel_notebook_format(
    df: pd.DataFrame,
    macro_vars: Optional[Iterable[str]] = None,
    variable_col: str = "Variable",
    entity_col: str = "Banco",
    date_col: str = "Fecha",
) -> pd.DataFrame:
    """Transform panel format used in the notebook:
    columns `Variable`, `Banco`, and dates as columns -> long panel Banco/Fecha/variables.
    Macro variables can be repeated/blank by Banco and are merged by Fecha.
    """
    required = {variable_col, entity_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas para panel: {missing}")

    macro_vars = list(macro_vars or [])
    id_vars = [variable_col, entity_col]
    df_long = df.melt(id_vars=id_vars, var_name=date_col, value_name="Valor")
    df_long[date_col] = pd.to_datetime(df_long[date_col], errors="coerce")
    df_long = df_long.dropna(subset=[date_col])
    df_long["Valor"] = pd.to_numeric(df_long["Valor"], errors="coerce")

    # Bank-specific variables: rows with Banco informed and variable not explicitly macro.
    banco_clean = df_long[entity_col].astype(str).str.strip()
    df_bank = df_long[(df_long[entity_col].notna()) & (banco_clean != "") & (~df_long[variable_col].isin(macro_vars))].copy()

    if df_bank.empty:
        raise ValueError("No se encontraron observaciones bancarias con Banco informado.")

    df_bank_wide = (
        df_bank.pivot_table(index=[entity_col, date_col], columns=variable_col, values="Valor", aggfunc="first")
        .reset_index()
    )
    df_bank_wide.columns.name = None

    if macro_vars:
        df_macro = df_long[df_long[variable_col].isin(macro_vars)].copy()
        if not df_macro.empty:
            df_macro_wide = (
                df_macro.pivot_table(index=date_col, columns=variable_col, values="Valor", aggfunc="first")
                .reset_index()
            )
            df_macro_wide.columns.name = None
            out = df_bank_wide.merge(df_macro_wide, on=date_col, how="left")
        else:
            out = df_bank_wide
    else:
        out = df_bank_wide

    out = out.sort_values([entity_col, date_col]).reset_index(drop=True)
    return out


def transform_panel_long_format(
    df: pd.DataFrame,
    entity_col: str = "Entidad",
    date_col: str = "Fecha",
    output_entity_col: str = "Banco",
    output_date_col: str = "Fecha",
    drop_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Transform panel format that is already tidy/long: one row per
    Entidad-Fecha, with variables already as columns (e.g. BD_tasas: Cdg, Fecha,
    Nro, Entidad, TEA_Tarjeta, ...). No melt/pivot is needed here — this only
    normalizes types, drops helper/id columns, and renames entity/date columns
    to match the convention the rest of the app expects (Banco/Fecha).

    - Values that are not numeric (e.g. "-" placeholders) become NaN.
    - Rows with a missing entity or an unparseable date are dropped.
    """
    required = {entity_col, date_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas para panel: {missing}")

    out = df.copy()

    drop_cols = [c for c in (drop_cols or []) if c in out.columns and c not in {entity_col, date_col}]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[entity_col] = out[entity_col].astype(str).str.strip()
    out = out[out[entity_col] != ""]
    out = out.dropna(subset=[date_col, entity_col])

    value_cols = [c for c in out.columns if c not in {entity_col, date_col}]
    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    rename_map = {}
    if entity_col != output_entity_col:
        rename_map[entity_col] = output_entity_col
    if date_col != output_date_col:
        rename_map[date_col] = output_date_col
    if rename_map:
        out = out.rename(columns=rename_map)

    out = out.sort_values([output_entity_col, output_date_col]).reset_index(drop=True)
    return out


def detect_panel_format(
    df: pd.DataFrame,
    variable_col: str = "Variable",
    entity_col_notebook: str = "Banco",
    entity_col_long: str = "Entidad",
    date_col: str = "Fecha",
) -> str:
    """Detect which of the two supported panel structures a raw sheet has:

    - "notebook": columns `Variable` + an entity column, dates as columns (wide).
      e.g. Data_Panel.xlsx / Datos_1.
    - "long": columns `Fecha` + an entity column, variables already as columns.
      e.g. BD_tasas.xlsx / BD.
    - "unknown": neither pattern matches.
    """
    cols = set(df.columns)
    has_entity = entity_col_notebook in cols or entity_col_long in cols

    if variable_col in cols and has_entity:
        return "notebook"
    if date_col in cols and has_entity:
        return "long"
    return "unknown"


def transform_panel_auto(
    df: pd.DataFrame,
    macro_vars: Optional[Iterable[str]] = None,
    variable_col: str = "Variable",
    entity_col_notebook: str = "Banco",
    entity_col_long: str = "Entidad",
    date_col: str = "Fecha",
    output_entity_col: str = "Banco",
    drop_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Detect the panel format and dispatch to the matching transformer."""
    fmt = detect_panel_format(df, variable_col, entity_col_notebook, entity_col_long, date_col)

    if fmt == "notebook":
        entity_col = entity_col_notebook if entity_col_notebook in df.columns else entity_col_long
        return transform_panel_notebook_format(
            df,
            macro_vars=macro_vars,
            variable_col=variable_col,
            entity_col=entity_col,
            date_col=date_col,
        )
    if fmt == "long":
        entity_col = entity_col_long if entity_col_long in df.columns else entity_col_notebook
        return transform_panel_long_format(
            df,
            entity_col=entity_col,
            date_col=date_col,
            output_entity_col=output_entity_col,
            drop_cols=drop_cols,
        )
    raise ValueError(
        "No se pudo detectar el formato del panel. Se esperaba columnas "
        f"'{variable_col}' + entidad ('{entity_col_notebook}'/'{entity_col_long}') para formato notebook, "
        f"o '{date_col}' + entidad con variables como columnas para formato long."
    )
