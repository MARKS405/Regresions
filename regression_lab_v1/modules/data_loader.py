from __future__ import annotations

import pandas as pd
from typing import BinaryIO, List, Tuple


DATE_COLUMN_DEFAULT = "Fecha"
PANEL_REQUIRED_COLUMNS = {"Variable", "Banco"}


def get_sheet_names(uploaded_file: BinaryIO) -> List[str]:
    """Return Excel sheet names without persisting the uploaded file."""
    uploaded_file.seek(0)
    xls = pd.ExcelFile(uploaded_file)
    return xls.sheet_names


def load_excel_sheet(uploaded_file: BinaryIO, sheet_name: str, header: int = 0) -> pd.DataFrame:
    """Load a selected sheet from an uploaded Excel file."""
    uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header)
    df = df.dropna(how="all").copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_data_type(df: pd.DataFrame, date_col: str = DATE_COLUMN_DEFAULT) -> Tuple[str, str]:
    """Detect whether a dataframe looks like panel-notebook format or aggregate time series."""
    cols = set(df.columns)
    if PANEL_REQUIRED_COLUMNS.issubset(cols):
        return "Panel", "Se detectaron columnas `Variable` y `Banco`; parece panel estilo notebook."
    if date_col in cols:
        return "Agregada", f"Se detectó columna `{date_col}`; parece serie agregada."
    return "Indefinido", "No se detectó estructura estándar. Revisa columnas esperadas."


def validate_monthly_dates(df: pd.DataFrame, date_col: str = DATE_COLUMN_DEFAULT) -> pd.DataFrame:
    """Parse and sort a monthly date column."""
    if date_col not in df.columns:
        raise ValueError(f"No se encontró la columna de fecha `{date_col}`.")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return out
