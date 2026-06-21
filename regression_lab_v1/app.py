from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_loader import get_sheet_names, load_excel_sheet, detect_data_type, validate_monthly_dates
from modules.panel_transformer import transform_panel_notebook_format
from modules.variable_transformer import (
    infer_numeric_columns,
    infer_dummy_columns,
    generate_event_dummies,
    generate_transformations,
    create_interaction,
)
from modules.diagnostics import (
    missing_summary,
    descriptive_summary,
    correlation_matrix,
    adf_tests,
    vif_table,
    residual_diagnostics_ols,
    panel_balance,
)
from modules.models_ols import run_ols, coef_table, model_metrics
from modules.models_panel import run_panel_model, coef_table_panel, panel_metrics
from modules.reconstruction import reconstruct_target
from modules.counterfactuals import apply_counterfactual, contribution_table
from modules.interpretation import executive_interpretation
from modules.exporters import dataframe_dict_to_excel_bytes, config_to_json_bytes, generate_notebook_bytes


st.set_page_config(
    page_title="Econometric Regression Lab · BCRP",
    page_icon="📈",
    layout="wide",
)

BCRP_RED = "#8A1538"
BCRP_DARK = "#2B2B2B"
BCRP_LIGHT = "#F7F3F4"

st.markdown(
    f"""
    <style>
    .main {{background-color: #ffffff;}}
    .stTabs [data-baseweb="tab-list"] {{gap: 6px;}}
    .stTabs [data-baseweb="tab"] {{height: 44px; background-color: {BCRP_LIGHT}; border-radius: 8px 8px 0 0;}}
    .stTabs [aria-selected="true"] {{background-color: {BCRP_RED}; color: white;}}
    div[data-testid="stMetric"] {{background-color: {BCRP_LIGHT}; padding: 12px; border-radius: 10px;}}
    .small-note {{font-size: 0.88rem; color: #666;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Auth
# ---------------------------
def check_password() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    st.title("Econometric Regression Lab")
    st.caption("Acceso privado · Versión 1")
    pwd = st.text_input("Contraseña", type="password")
    expected = st.secrets.get("APP_PASSWORD", "bcrp_demo")
    if st.button("Ingresar"):
        if pwd == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    return False


if not check_password():
    st.stop()


# ---------------------------
# Helpers
# ---------------------------
def init_state():
    defaults = {
        "raw_df": None,
        "base_df": None,
        "work_df": None,
        "sheet_name": None,
        "data_type": None,
        "events": [],
        "models": [],
        "last_result": None,
        "last_config": None,
        "dummy_vars": [],
        "macro_vars": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def numeric_options(df: pd.DataFrame, exclude=()) -> list[str]:
    if df is None:
        return []
    exclude = set(exclude)
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def model_summary_text(result_obj, data_type: str) -> str:
    if result_obj is None:
        return ""
    try:
        if data_type == "Agregada":
            return result_obj.summary().as_text()
        return str(result_obj.summary)
    except Exception as exc:
        return f"No se pudo construir summary: {exc}"


def add_model_to_store(name: str, metrics: dict, config: dict, coef_df: pd.DataFrame, economic_score: int):
    row = {
        "modelo": name,
        "tipo": config.get("model_type"),
        "dependiente": config.get("dependent_variable"),
        "objetivo": config.get("target_variable"),
        "n_regresoras": len(config.get("regressors", [])),
        "score_economico": economic_score,
        "significativas_10pct": int((coef_df["p_value"] < 0.10).sum()) if coef_df is not None and not coef_df.empty else 0,
        **metrics,
        "config": config,
    }
    st.session_state["models"].append(row)


def render_metric_row(metrics: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("N obs.", f"{metrics.get('nobs', np.nan):.0f}" if pd.notna(metrics.get('nobs', np.nan)) else "—")
    c2.metric("RMSE", f"{metrics.get('rmse', np.nan):.4f}" if pd.notna(metrics.get('rmse', np.nan)) else "—")
    c3.metric("R²", f"{metrics.get('r2', np.nan):.3f}" if pd.notna(metrics.get('r2', np.nan)) else "—")
    c4.metric("R² adj.", f"{metrics.get('r2_adj', np.nan):.3f}" if pd.notna(metrics.get('r2_adj', np.nan)) else "—")
    c5.metric("AIC/BIC", f"{metrics.get('aic', np.nan):.1f} / {metrics.get('bic', np.nan):.1f}" if pd.notna(metrics.get('aic', np.nan)) else "—")


init_state()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown(f"<h2 style='color:{BCRP_RED}; margin-bottom:0'>BCRP</h2>", unsafe_allow_html=True)
    st.caption("Econometric Regression Lab · V1")
    st.divider()
    if st.button("Cerrar sesión y limpiar data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown("**Alcance V1**")
    st.markdown("Excel · mensual · OLS/HAC · Panel FE/RE · exportables")
    st.info("Los datos se mantienen solo en memoria de sesión.")

st.title("Econometric Regression Lab")
st.caption("Herramienta reproducible para regresiones económicas/financieras · Versión 1")

tabs = st.tabs([
    "1. Carga",
    "2. Transformaciones",
    "3. Descriptivo",
    "4. Diagnósticos",
    "5. Modelo",
    "6. Resultados",
    "7. Contrafactual",
    "8. Comparador",
    "9. Exportar",
])

# ---------------------------
# 1 Carga
# ---------------------------
with tabs[0]:
    st.subheader("Carga y configuración inicial")
    uploaded = st.file_uploader("Sube un archivo Excel", type=["xlsx", "xls"])
    if uploaded:
        try:
            sheets = get_sheet_names(uploaded)
            sheet = st.selectbox("Selecciona hoja", sheets, index=0)
            header = st.number_input("Fila de encabezado en Excel (0 = primera fila)", min_value=0, value=0, step=1)
            raw = load_excel_sheet(uploaded, sheet, header=int(header))
            detected_type, msg = detect_data_type(raw, date_col="Fecha")
            st.info(msg)
            data_type = st.radio(
                "Tipo de data",
                ["Agregada", "Panel"],
                index=0 if detected_type != "Panel" else 1,
                horizontal=True,
            )
            st.session_state["raw_df"] = raw
            st.session_state["sheet_name"] = sheet
            st.session_state["data_type"] = data_type
            st.write("Vista previa del archivo cargado")
            st.dataframe(raw.head(20), use_container_width=True)

            if data_type == "Panel":
                variables = sorted(raw["Variable"].dropna().astype(str).unique().tolist()) if "Variable" in raw.columns else []
                default_macro = [v for v in ["PBI", "TasaBCRP"] if v in variables]
                macro_vars = st.multiselect(
                    "Variables macro a replicar por fecha en todos los bancos",
                    variables,
                    default=default_macro,
                )
                st.session_state["macro_vars"] = macro_vars
                if st.button("Preparar panel"):
                    base = transform_panel_notebook_format(raw, macro_vars=macro_vars)
                    st.session_state["base_df"] = base
                    st.success("Panel transformado correctamente.")
                    st.dataframe(base.head(30), use_container_width=True)
            else:
                if st.button("Preparar serie agregada"):
                    base = validate_monthly_dates(raw, date_col="Fecha")
                    st.session_state["base_df"] = base
                    st.success("Serie agregada preparada correctamente.")
                    st.dataframe(base.head(30), use_container_width=True)
        except Exception as exc:
            st.error(f"Error al cargar/preparar datos: {exc}")
    else:
        st.warning("Sube un Excel para iniciar.")

# ---------------------------
# 2 Transformaciones
# ---------------------------
with tabs[1]:
    st.subheader("Transformaciones automáticas")
    base = st.session_state.get("base_df")
    if base is None:
        st.warning("Primero prepara la data en la pestaña Carga.")
    else:
        data_type = st.session_state.get("data_type")
        entity_col = "Banco" if data_type == "Panel" else None
        num_cols = infer_numeric_columns(base, date_col="Fecha", entity_col=entity_col)
        inferred = infer_dummy_columns(base, num_cols)
        dummy_vars = st.multiselect(
            "Marca variables dummy: no se les aplicará log",
            num_cols,
            default=inferred,
        )
        st.session_state["dummy_vars"] = dummy_vars

        with st.expander("Crear dummies de eventos por rango de fechas", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            ev_name = col1.text_input("Nombre de dummy", value="Evento")
            ev_start = col2.date_input("Inicio", value=date(2020, 5, 1))
            ev_end = col3.date_input("Fin", value=date(2020, 8, 1))
            if st.button("Agregar evento"):
                event = {"name": ev_name.strip().replace(" ", "_"), "start": str(ev_start), "end": str(ev_end)}
                st.session_state["events"].append(event)
                st.success(f"Evento agregado: {event}")
            if st.session_state["events"]:
                st.dataframe(pd.DataFrame(st.session_state["events"]), use_container_width=True)
                if st.button("Limpiar eventos"):
                    st.session_state["events"] = []
                    st.rerun()

        include_ma = st.checkbox("Agregar medias móviles ma3, ma6, ma12", value=False)
        include_lags = st.checkbox("Agregar lags 3, 6 y 12", value=False)

        with st.expander("Crear interacción manual", expanded=False):
            opts = numeric_options(base, exclude=["Fecha", "Banco"])
            ia = st.selectbox("Variable A", opts, index=0 if opts else None)
            ib = st.selectbox("Variable B", opts, index=1 if len(opts) > 1 else 0 if opts else None)
            inter_name = st.text_input("Nombre de interacción", value=f"{ia}_x_{ib}" if opts else "")

        if st.button("Generar base transformada"):
            try:
                base_events = generate_event_dummies(base, st.session_state["events"], date_col="Fecha")
                if ia and ib and inter_name:
                    base_events = create_interaction(base_events, ia, ib, inter_name)
                work = generate_transformations(
                    base_events,
                    date_col="Fecha",
                    entity_col=entity_col,
                    dummy_vars=dummy_vars + [e["name"] for e in st.session_state["events"]],
                    include_ma=include_ma,
                    include_lags_3_6_12=include_lags,
                )
                st.session_state["work_df"] = work
                st.success(f"Base transformada generada: {work.shape[0]} filas y {work.shape[1]} columnas.")
            except Exception as exc:
                st.error(f"Error generando transformaciones: {exc}")

        work = st.session_state.get("work_df")
        if work is not None:
            st.write("Variables disponibles después del tratamiento")
            st.dataframe(pd.DataFrame({"variable": work.columns, "dtype": [str(work[c].dtype) for c in work.columns]}), use_container_width=True)
            st.dataframe(work.head(30), use_container_width=True)

# ---------------------------
# 3 Descriptivo
# ---------------------------
with tabs[2]:
    st.subheader("Análisis descriptivo")
    work = st.session_state.get("work_df")
    if work is None:
        st.warning("Primero genera la base transformada.")
    else:
        entity_col = "Banco" if st.session_state.get("data_type") == "Panel" else None
        cols = numeric_options(work, exclude=["Fecha", "Banco"])
        selected = st.multiselect("Variables para graficar", cols, default=cols[: min(4, len(cols))])
        if selected:
            plot_df = work[[c for c in ["Fecha", entity_col] if c] + selected].copy()
            if entity_col:
                long = plot_df.melt(id_vars=["Fecha", entity_col], value_vars=selected, var_name="variable", value_name="valor")
                fig = px.line(long, x="Fecha", y="valor", color="variable", facet_row=entity_col, title="Series por banco")
            else:
                long = plot_df.melt(id_vars=["Fecha"], value_vars=selected, var_name="variable", value_name="valor")
                fig = px.line(long, x="Fecha", y="valor", color="variable", title="Series temporales")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Estadísticos descriptivos**")
        st.dataframe(descriptive_summary(work, selected or cols[:20]), use_container_width=True)

        st.markdown("**Matriz de correlación**")
        corr_vars = st.multiselect("Variables para correlación", cols, default=cols[: min(8, len(cols))])
        corr = correlation_matrix(work, corr_vars)
        if not corr.empty:
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlación")
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("**Missing values**")
        st.dataframe(missing_summary(work).head(50), use_container_width=True)

        if entity_col:
            st.markdown("**Balance del panel**")
            st.dataframe(panel_balance(work, entity_col=entity_col, date_col="Fecha"), use_container_width=True)

# ---------------------------
# 4 Diagnosticos
# ---------------------------
with tabs[3]:
    st.subheader("Diagnósticos econométricos")
    work = st.session_state.get("work_df")
    if work is None:
        st.warning("Primero genera la base transformada.")
    else:
        cols = numeric_options(work, exclude=["Fecha", "Banco"])
        adf_vars = st.multiselect("Variables para ADF", cols, default=cols[: min(6, len(cols))])
        if st.button("Correr ADF"):
            st.dataframe(adf_tests(work, adf_vars), use_container_width=True)

        vif_vars = st.multiselect("Variables para VIF", cols, default=[])
        if st.button("Calcular VIF"):
            st.dataframe(vif_table(work, vif_vars, add_constant=True), use_container_width=True)

        st.info("Los diagnósticos de residuales se mostrarán automáticamente después de correr un modelo OLS.")

# ---------------------------
# 5 Modelo
# ---------------------------
with tabs[4]:
    st.subheader("Configuración y estimación del modelo")
    work = st.session_state.get("work_df")
    if work is None:
        st.warning("Primero genera la base transformada.")
    else:
        data_type = st.session_state.get("data_type")
        entity_col = "Banco" if data_type == "Panel" else None
        cols = numeric_options(work, exclude=["Fecha", "Banco"])
        original_cols = numeric_options(st.session_state.get("base_df"), exclude=["Fecha", "Banco"])

        col1, col2 = st.columns(2)
        target_var = col1.selectbox("Variable objetivo original", original_cols, index=0 if original_cols else None)
        dep_default = f"d_ln_{target_var}" if target_var and f"d_ln_{target_var}" in cols else target_var
        dep_index = cols.index(dep_default) if dep_default in cols else 0
        y_var = col2.selectbox("Variable dependiente del modelo", cols, index=dep_index if cols else None)

        x_candidates = [c for c in cols if c != y_var]
        x_vars = st.multiselect("Regresoras", x_candidates)
        add_constant = st.checkbox("Incluir constante", value=(data_type == "Agregada"))
        economic_score = st.slider("Sentido económico de la especificación", min_value=1, max_value=5, value=3, help="Puntaje manual para el comparador; 5 = mayor coherencia económica.")
        model_name = st.text_input("Nombre del modelo", value="Modelo 1")

        if data_type == "Agregada":
            model_type = st.selectbox("Modelo", ["OLS"])
            cov_type = st.selectbox("Errores estándar", ["HAC", "HC1", "HC3", "No robusto"])
            hac_lags = st.number_input("Maxlags HAC", min_value=1, max_value=24, value=3, step=1)
            run = st.button("Correr regresión agregada")
            if run:
                try:
                    cov_map = {"No robusto": "nonrobust", "HAC": "HAC", "HC1": "HC1", "HC3": "HC3"}
                    out = run_ols(work, y_var, x_vars, add_constant=add_constant, cov_type=cov_map[cov_type], hac_lags=int(hac_lags))
                    res = out["results"]
                    coefs = coef_table(res)
                    metrics = model_metrics(res, out["y"], out["pred"])

                    pred_col = f"pred_{y_var}"
                    model_df = work.copy()
                    model_df[pred_col] = np.nan
                    model_df.loc[out["pred"].index, pred_col] = out["pred"]
                    model_df = reconstruct_target(model_df, target_var, y_var, pred_col, entity_col=None, date_col="Fecha")

                    config = {
                        "sheet_name": st.session_state.get("sheet_name"),
                        "data_type": data_type,
                        "frequency": "monthly",
                        "target_variable": target_var,
                        "dependent_variable": y_var,
                        "regressors": x_vars,
                        "model_type": model_type,
                        "cov_type": cov_map[cov_type],
                        "hac_lags": int(hac_lags),
                        "add_constant": add_constant,
                        "dummy_vars": st.session_state.get("dummy_vars", []),
                        "events": st.session_state.get("events", []),
                        "macro_vars": st.session_state.get("macro_vars", []),
                    }
                    st.session_state["last_result"] = {
                        "type": data_type,
                        "result_obj": res,
                        "coef_df": coefs,
                        "metrics": metrics,
                        "model_data": out["data"],
                        "X": out["X"],
                        "y": out["y"],
                        "pred": out["pred"],
                        "model_df": model_df,
                        "pred_col": pred_col,
                        "target_var": target_var,
                        "dep_var": y_var,
                        "x_vars": x_vars,
                    }
                    st.session_state["last_config"] = config
                    add_model_to_store(model_name, metrics, config, coefs, economic_score)
                    st.success("Modelo estimado correctamente.")
                except Exception as exc:
                    st.error(f"Error al estimar modelo: {exc}")
        else:
            model_kind = st.selectbox("Modelo panel", ["Pooled OLS", "FE entidad", "FE tiempo", "Two-way FE", "Random Effects"], index=3)
            cov_type = st.selectbox("Covarianza panel", ["clustered", "robust", "unadjusted"], index=0)
            cluster_entity = st.checkbox("Cluster por entidad", value=True)
            run = st.button("Correr modelo panel")
            if run:
                try:
                    out = run_panel_model(
                        work,
                        y_var,
                        x_vars,
                        model_kind=model_kind,
                        add_constant=add_constant,
                        cov_type=cov_type,
                        cluster_entity=cluster_entity,
                        entity_col="Banco",
                        date_col="Fecha",
                    )
                    res = out["results"]
                    coefs = coef_table_panel(res)
                    metrics = panel_metrics(res, out["y"], out["pred"])

                    pred_col = f"pred_{y_var}"
                    pred_df = out["pred"].reset_index().rename(columns={0: pred_col, out["pred"].name: pred_col})
                    model_df = work.merge(pred_df, on=["Banco", "Fecha"], how="left")
                    model_df = reconstruct_target(model_df, target_var, y_var, pred_col, entity_col="Banco", date_col="Fecha")

                    config = {
                        "sheet_name": st.session_state.get("sheet_name"),
                        "data_type": data_type,
                        "frequency": "monthly",
                        "target_variable": target_var,
                        "dependent_variable": y_var,
                        "regressors": x_vars,
                        "model_type": "Panel",
                        "model_kind": model_kind,
                        "cov_type": cov_type,
                        "cluster_entity": cluster_entity,
                        "add_constant": add_constant,
                        "dummy_vars": st.session_state.get("dummy_vars", []),
                        "events": st.session_state.get("events", []),
                        "macro_vars": st.session_state.get("macro_vars", []),
                    }
                    st.session_state["last_result"] = {
                        "type": data_type,
                        "result_obj": res,
                        "coef_df": coefs,
                        "metrics": metrics,
                        "model_data": out["data"],
                        "X": out["X"],
                        "y": out["y"],
                        "pred": out["pred"],
                        "model_df": model_df,
                        "pred_col": pred_col,
                        "target_var": target_var,
                        "dep_var": y_var,
                        "x_vars": x_vars,
                    }
                    st.session_state["last_config"] = config
                    add_model_to_store(model_name, metrics, config, coefs, economic_score)
                    st.success("Modelo panel estimado correctamente.")
                except Exception as exc:
                    st.error(f"Error al estimar panel: {exc}")

# ---------------------------
# 6 Resultados
# ---------------------------
with tabs[5]:
    st.subheader("Resultados del último modelo")
    last = st.session_state.get("last_result")
    if not last:
        st.warning("Corre un modelo en la pestaña Modelo.")
    else:
        render_metric_row(last["metrics"])
        st.markdown("**Coeficientes**")
        st.dataframe(last["coef_df"], use_container_width=True)
        st.markdown("**Summary**")
        st.text(model_summary_text(last["result_obj"], last["type"]))

        st.markdown("**Interpretación ejecutiva**")
        st.write(executive_interpretation(
            last["coef_df"],
            target_var=last["target_var"],
            dep_var=last["dep_var"],
            dummy_vars=st.session_state.get("dummy_vars", []),
        ))

        model_df = last["model_df"]
        target = last["target_var"]
        est_col = f"estimado_{target}"
        if est_col in model_df.columns and target in model_df.columns:
            if last["type"] == "Panel":
                plot_long = model_df[["Banco", "Fecha", target, est_col]].melt(
                    id_vars=["Banco", "Fecha"], value_vars=[target, est_col], var_name="serie", value_name="valor"
                )
                fig = px.line(plot_long, x="Fecha", y="valor", color="serie", facet_row="Banco", title=f"{target} real vs estimado")
            else:
                fig = px.line(model_df, x="Fecha", y=[target, est_col], title=f"{target} real vs estimado")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Contribución promedio por regresora**")
        params = last["result_obj"].params
        contrib = contribution_table(model_df, params, last["x_vars"])
        st.dataframe(contrib, use_container_width=True)

        if last["type"] == "Agregada":
            st.markdown("**Diagnósticos de residuales OLS**")
            st.dataframe(residual_diagnostics_ols(last["result_obj"], last["X"]), use_container_width=True)

# ---------------------------
# 7 Contrafactual
# ---------------------------
with tabs[6]:
    st.subheader("Contrafactuales")
    last = st.session_state.get("last_result")
    if not last:
        st.warning("Corre un modelo antes de crear contrafactuales.")
    else:
        remove_vars = st.multiselect("Variables a excluir del estimado", last["x_vars"])
        cf_name = st.text_input("Nombre del escenario", value="sin_variables")
        if st.button("Construir contrafactual"):
            try:
                model_df = last["model_df"].copy()
                params = last["result_obj"].params
                pred_col = last["pred_col"]
                cf_pred_col = f"{pred_col}_{cf_name}"
                cf_df = apply_counterfactual(model_df, pred_col, params, remove_vars, output_col=cf_pred_col)
                cf_df = reconstruct_target(
                    cf_df,
                    last["target_var"],
                    last["dep_var"],
                    cf_pred_col,
                    entity_col="Banco" if last["type"] == "Panel" else None,
                    date_col="Fecha",
                    output_col=f"estimado_{last['target_var']}_{cf_name}",
                )
                st.session_state["last_result"]["model_df"] = cf_df
                st.success("Contrafactual construido.")
                target = last["target_var"]
                cols_plot = [target, f"estimado_{target}", f"estimado_{target}_{cf_name}"]
                if last["type"] == "Panel":
                    plot_long = cf_df[["Banco", "Fecha"] + cols_plot].melt(
                        id_vars=["Banco", "Fecha"], value_vars=cols_plot, var_name="serie", value_name="valor"
                    )
                    fig = px.line(plot_long, x="Fecha", y="valor", color="serie", facet_row="Banco", title=f"Contrafactual: {cf_name}")
                else:
                    fig = px.line(cf_df, x="Fecha", y=cols_plot, title=f"Contrafactual: {cf_name}")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(cf_df[[c for c in ["Banco", "Fecha"] if c in cf_df.columns] + cols_plot].dropna(how="all"), use_container_width=True)
            except Exception as exc:
                st.error(f"Error construyendo contrafactual: {exc}")

# ---------------------------
# 8 Comparador
# ---------------------------
with tabs[7]:
    st.subheader("Comparador de modelos de la sesión")
    models = st.session_state.get("models", [])
    if not models:
        st.warning("Aún no hay modelos guardados en la sesión.")
    else:
        comp = pd.DataFrame([{k: v for k, v in m.items() if k != "config"} for m in models])
        # Orden sugerido: sentido económico, significancia y menor RMSE primero; AIC/BIC y R2 como segunda lectura.
        comp_sorted = comp.sort_values(
            by=["score_economico", "significativas_10pct", "rmse", "aic", "r2_adj"],
            ascending=[False, False, True, True, False],
            na_position="last",
        )
        st.dataframe(comp_sorted, use_container_width=True)
        st.caption("Orden: sentido económico y significancia primero; luego menor RMSE; después AIC/BIC y R² ajustado.")

# ---------------------------
# 9 Exportar
# ---------------------------
with tabs[8]:
    st.subheader("Exportar resultados")
    work = st.session_state.get("work_df")
    last = st.session_state.get("last_result")
    config = st.session_state.get("last_config")

    if work is not None:
        st.download_button(
            "Descargar base tratada Excel",
            data=dataframe_dict_to_excel_bytes({"base_tratada": work}),
            file_name="base_tratada_regression_lab.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if last:
        sheets = {
            "coeficientes": last["coef_df"],
            "modelo_data": last["model_data"].reset_index(),
            "estimado": last["model_df"],
        }
        st.download_button(
            "Descargar resultados del último modelo Excel",
            data=dataframe_dict_to_excel_bytes(sheets),
            file_name="resultados_modelo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if config:
            st.download_button(
                "Descargar configuración JSON",
                data=config_to_json_bytes(config),
                file_name="config_modelo.json",
                mime="application/json",
            )
            st.download_button(
                "Descargar notebook reproducible .ipynb",
                data=generate_notebook_bytes(config),
                file_name="notebook_reproducible.ipynb",
                mime="application/x-ipynb+json",
            )
    else:
        st.info("Cuando corras un modelo, aquí aparecerán los exportables de summary, data del modelo, config y notebook.")
