from __future__ import annotations

import io
import json
from typing import Dict, Any
import pandas as pd
import nbformat as nbf


def dataframe_dict_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            clean_name = str(name)[:31].replace("/", "-").replace("\\", "-")
            if df is None:
                continue
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            df.to_excel(writer, index=True, sheet_name=clean_name)
    return output.getvalue()


def config_to_json_bytes(config: Dict[str, Any]) -> bytes:
    return json.dumps(config, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def generate_notebook_bytes(config: Dict[str, Any]) -> bytes:
    """Generate a reproducible notebook with code blocks for load/transform/model/plot."""
    nb = nbf.v4.new_notebook()
    nb.cells = []
    nb.cells.append(nbf.v4.new_markdown_cell("# Notebook reproducible generado por Econometric Regression Lab"))
    nb.cells.append(nbf.v4.new_markdown_cell("Este notebook replica la estructura de importación, transformación, regresión y gráfico objetivo vs estimado. Reemplaza `DATA_FILE.xlsx` por tu archivo."))
    imports = """
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
""".strip()
    nb.cells.append(nbf.v4.new_code_cell(imports))

    cfg_literal = json.dumps(config, indent=2, ensure_ascii=False, default=str)
    nb.cells.append(nbf.v4.new_code_cell(f"CONFIG = {cfg_literal}"))

    load_code = """
# 1. Importar datos
archivo = "DATA_FILE.xlsx"  # cambiar por la ruta real
sheet_name = CONFIG.get("sheet_name", 0)
raw = pd.read_excel(archivo, sheet_name=sheet_name)
raw.columns = [str(c).strip() for c in raw.columns]
raw.head()
""".strip()
    nb.cells.append(nbf.v4.new_code_cell(load_code))

    transform_code = """
# 2. Transformación base

def transformar_panel_notebook(df, macro_vars):
    df_long = df.melt(id_vars=["Variable", "Banco"], var_name="Fecha", value_name="Valor")
    df_long["Fecha"] = pd.to_datetime(df_long["Fecha"], errors="coerce")
    df_long["Valor"] = pd.to_numeric(df_long["Valor"], errors="coerce")
    banco_clean = df_long["Banco"].astype(str).str.strip()
    df_bank = df_long[(df_long["Banco"].notna()) & (banco_clean != "") & (~df_long["Variable"].isin(macro_vars))]
    df_bank_wide = df_bank.pivot_table(index=["Banco", "Fecha"], columns="Variable", values="Valor", aggfunc="first").reset_index()
    df_bank_wide.columns.name = None
    if macro_vars:
        df_macro = df_long[df_long["Variable"].isin(macro_vars)]
        df_macro_wide = df_macro.pivot_table(index="Fecha", columns="Variable", values="Valor", aggfunc="first").reset_index()
        df_macro_wide.columns.name = None
        return df_bank_wide.merge(df_macro_wide, on="Fecha", how="left")
    return df_bank_wide

def shift(df, col, entity_col=None, periods=1):
    return df.groupby(entity_col)[col].shift(periods) if entity_col else df[col].shift(periods)

def diff(df, col, entity_col=None, periods=1):
    return df.groupby(entity_col)[col].diff(periods) if entity_col else df[col].diff(periods)

def generar_transformaciones(df, date_col="Fecha", entity_col=None, dummy_vars=None):
    dummy_vars = set(dummy_vars or [])
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    sort_cols = [c for c in [entity_col, date_col] if c]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    numeric_cols = [c for c in out.columns if c not in [date_col, entity_col] and pd.api.types.is_numeric_dtype(out[c])]
    for v in numeric_cols:
        out[f"{v}_lag1"] = shift(out, v, entity_col, 1)
        out[f"d_{v}"] = diff(out, v, entity_col, 1)
        out[f"d_{v}_lag1"] = shift(out, f"d_{v}", entity_col, 1)
        out[f"d_{v}_12"] = diff(out, v, entity_col, 12)
        if v not in dummy_vars:
            out[f"ln_{v}"] = np.log(out[v].where(out[v] > 0))
            out[f"ln_{v}_lag1"] = shift(out, f"ln_{v}", entity_col, 1)
            out[f"d_ln_{v}"] = diff(out, f"ln_{v}", entity_col, 1)
            out[f"d_ln_{v}_lag1"] = shift(out, f"d_ln_{v}", entity_col, 1)
            out[f"d_ln_{v}_12"] = diff(out, f"ln_{v}", entity_col, 12)
    return out.replace([np.inf, -np.inf], np.nan)

if CONFIG["data_type"] == "Panel":
    base = transformar_panel_notebook(raw, CONFIG.get("macro_vars", []))
    entity_col = "Banco"
else:
    base = raw.copy()
    base["Fecha"] = pd.to_datetime(base["Fecha"], errors="coerce")
    base = base.sort_values("Fecha").reset_index(drop=True)
    entity_col = None

# Dummies de eventos
for ev in CONFIG.get("events", []):
    name = ev["name"]
    start = pd.to_datetime(ev["start"])
    end = pd.to_datetime(ev["end"])
    base[name] = ((base["Fecha"] >= start) & (base["Fecha"] <= end)).astype(int)

work = generar_transformaciones(base, date_col="Fecha", entity_col=entity_col, dummy_vars=CONFIG.get("dummy_vars", []))
work.head()
""".strip()
    nb.cells.append(nbf.v4.new_code_cell(transform_code))

    model_code = """
# 3. Regresión
y_var = CONFIG["dependent_variable"]
x_vars = CONFIG["regressors"]
add_constant = CONFIG.get("add_constant", True)

if CONFIG["data_type"] == "Panel":
    panel = work[["Banco", "Fecha", y_var] + x_vars].dropna().set_index(["Banco", "Fecha"])
    y = panel[y_var]
    X = panel[x_vars]
    if add_constant:
        X = sm.add_constant(X, has_constant="add")
    model_kind = CONFIG.get("model_kind", "Two-way FE")
    if model_kind == "Pooled OLS":
        model = PooledOLS(y, X, check_rank=False)
    elif model_kind == "Random Effects":
        model = RandomEffects(y, X, check_rank=False)
    else:
        model = PanelOLS(
            y, X,
            entity_effects=model_kind in ["FE entidad", "Two-way FE"],
            time_effects=model_kind in ["FE tiempo", "Two-way FE"],
            drop_absorbed=True,
            check_rank=False
        )
    results = model.fit(cov_type="clustered", cluster_entity=True)
    print(results.summary)
    params = results.params
    common_x = [c for c in X.columns if c in params.index]
    pred = X[common_x].mul(params[common_x], axis=1).sum(axis=1).rename(f"pred_{y_var}")
    pred_df = pred.reset_index()
    work = work.merge(pred_df, on=["Banco", "Fecha"], how="left")
else:
    data = work[[y_var] + x_vars].replace([np.inf, -np.inf], np.nan).dropna()
    y = data[y_var]
    X = data[x_vars]
    if add_constant:
        X = sm.add_constant(X, has_constant="add")
    cov_type = CONFIG.get("cov_type", "HAC")
    if cov_type == "HAC":
        results = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": CONFIG.get("hac_lags", 3)})
    elif cov_type in ["HC1", "HC3"]:
        results = sm.OLS(y, X).fit(cov_type=cov_type)
    else:
        results = sm.OLS(y, X).fit()
    print(results.summary())
    work.loc[data.index, f"pred_{y_var}"] = results.predict(X)
""".strip()
    nb.cells.append(nbf.v4.new_code_cell(model_code))

    plot_code = """
# 4. Reconstruir variable objetivo vs estimado

def reconstruir_grupo(g, target_var, dep_var, pred_col):
    g = g.copy()
    pred = g[pred_col]
    actual = g[target_var]
    if dep_var == target_var:
        return pred
    if dep_var == f"ln_{target_var}":
        return np.exp(pred)
    if dep_var == f"d_ln_{target_var}":
        out = pd.Series(np.nan, index=g.index)
        if actual.dropna().empty:
            return out
        first_idx = actual.dropna().index[0]
        running = actual.loc[first_idx]
        for idx in g.index:
            if idx == first_idx or pd.isna(pred.loc[idx]):
                out.loc[idx] = running
            else:
                running *= np.exp(pred.loc[idx])
                out.loc[idx] = running
        return out
    if dep_var == f"d_{target_var}":
        out = pd.Series(np.nan, index=g.index)
        if actual.dropna().empty:
            return out
        first_idx = actual.dropna().index[0]
        running = actual.loc[first_idx]
        for idx in g.index:
            if idx == first_idx or pd.isna(pred.loc[idx]):
                out.loc[idx] = running
            else:
                running += pred.loc[idx]
                out.loc[idx] = running
        return out
    return pred

target = CONFIG["target_variable"]
pred_col = f"pred_{CONFIG['dependent_variable']}"
out_col = f"estimado_{target}"
if CONFIG["data_type"] == "Panel":
    work[out_col] = work.groupby("Banco", group_keys=False).apply(lambda g: reconstruir_grupo(g.sort_values("Fecha"), target, y_var, pred_col))
    fig = px.line(work, x="Fecha", y=[target, out_col], facet_row="Banco", title=f"{target} real vs estimado")
else:
    work[out_col] = reconstruir_grupo(work.sort_values("Fecha"), target, y_var, pred_col)
    fig = px.line(work, x="Fecha", y=[target, out_col], title=f"{target} real vs estimado")
fig.show()
""".strip()
    nb.cells.append(nbf.v4.new_code_cell(plot_code))

    return nbf.writes(nb).encode("utf-8")
