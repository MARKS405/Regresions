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
