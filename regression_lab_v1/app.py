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


def safe_num(val, digits=4, pct=False):
    try:
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return "—"
        if pct:
            return f"{val:.{digits}%}"
        return f"{val:.{digits}f}"
    except Exception:
        return "—"


def significance_stars(p):
    try:
        if p < 0.01:
            return '***'
        if p < 0.05:
            return '**'
        if p < 0.10:
            return '*'
    except Exception:
        pass
    return ''


def prettify_coef_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if 'variable' in out.columns:
        out = out.rename(columns={'variable': 'Variable'})
    if 'coef' in out.columns:
        out['Coef.'] = out['coef'].map(lambda x: round(x, 4) if pd.notna(x) else x)
    if 'std_err' in out.columns:
        out['Err. Est.'] = out['std_err'].map(lambda x: round(x, 4) if pd.notna(x) else x)
    if 't_stat' in out.columns:
        out['t-stat'] = out['t_stat'].map(lambda x: round(x, 3) if pd.notna(x) else x)
    if 'p_value' in out.columns:
        out['p-valor'] = out['p_value'].map(lambda x: round(x, 4) if pd.notna(x) else x)
        out['Sig.'] = out['p_value'].map(significance_stars)
    if 'ci_low' in out.columns:
        out['LI 95%'] = out['ci_low'].map(lambda x: round(x, 4) if pd.notna(x) else x)
    if 'ci_high' in out.columns:
        out['LS 95%'] = out['ci_high'].map(lambda x: round(x, 4) if pd.notna(x) else x)
    preferred = [c for c in ['Variable', 'Coef.', 'Err. Est.', 't-stat', 'p-valor', 'Sig.', 'LI 95%', 'LS 95%'] if c in out.columns]
    return out[preferred]


def summary_snapshot(result_obj, metrics: dict, data_type: str) -> dict:
    snap = {
        'Observaciones': metrics.get('nobs'),
        'RMSE': metrics.get('rmse'),
        'R²': metrics.get('r2'),
        'R² ajustado': metrics.get('r2_adj'),
        'AIC': metrics.get('aic'),
        'BIC': metrics.get('bic'),
    }
    try:
        if hasattr(result_obj, 'fvalue'):
            snap['F-stat'] = float(result_obj.fvalue) if result_obj.fvalue is not None else np.nan
    except Exception:
        pass
    try:
        if hasattr(result_obj, 'f_pvalue'):
            snap['Prob(F)'] = float(result_obj.f_pvalue) if result_obj.f_pvalue is not None else np.nan
    except Exception:
        pass
    try:
        if hasattr(result_obj, 'loglik'):
            snap['Log-likelihood'] = float(result_obj.loglik)
    except Exception:
        pass
    try:
        if data_type == 'Panel' and hasattr(result_obj, 'entity_info'):
            ent = result_obj.entity_info
            if isinstance(ent, pd.Series):
                snap['Entidades'] = ent.get('total', np.nan)
    except Exception:
        pass
    return snap


def plot_model_comparison(model_df: pd.DataFrame, target: str, series_cols: list[str], panel: bool, title: str):
    cols_plot = [c for c in series_cols if c in model_df.columns]
    nice_names = {target: target, f'estimado_{target}': 'Estimado', **{c: c.replace(f'estimado_{target}_', '').replace('_', ' ').title() for c in cols_plot}}
    color_map = {
        target: '#2B2B2B',
        f'estimado_{target}': '#1f77b4',
    }
    # assign BCRP red to first extra scenario, then violet/orange if needed
    extras = [c for c in cols_plot if c not in [target, f'estimado_{target}']]
    extra_palette = ['#8A1538', '#9467bd', '#ff7f0e', '#2ca02c']
    for c, col in zip(extras, extra_palette):
        color_map[c] = col

    if panel:
        plot_long = model_df[['Banco', 'Fecha'] + cols_plot].melt(
            id_vars=['Banco', 'Fecha'], value_vars=cols_plot, var_name='serie', value_name='valor'
        )
        bancos = plot_long['Banco'].dropna().unique().tolist()
        n_cols = 2 if len(bancos) > 1 else 1
        height = max(520, 320 * math.ceil(len(bancos) / n_cols))
        fig = px.line(
            plot_long,
            x='Fecha', y='valor', color='serie',
            facet_col='Banco', facet_col_wrap=n_cols,
            color_discrete_map=color_map,
            title=title,
            labels={'valor': target, 'serie': 'Serie'}
        )
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
    else:
        fig = px.line(
            model_df,
            x='Fecha', y=cols_plot,
            color_discrete_map=color_map,
            title=title,
            labels={'value': target, 'variable': 'Serie'}
        )
        height = 520

    rename_map = {k: nice_names.get(k, k) for k in cols_plot}
    fig.for_each_trace(lambda tr: tr.update(name=rename_map.get(tr.name, tr.name), line=dict(width=2.6)))
    for tr in fig.data:
        if tr.name in ['Estimado']:
            tr.update(line=dict(dash='dash', width=2.5))
        if tr.name not in [nice_names.get(target, target), 'Estimado']:
            tr.update(line=dict(dash='dot', width=2.7))
    fig.update_layout(
        template='plotly_white',
        height=height,
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        hovermode='x unified',
    )
    return fig


def render_presentable_summary(last: dict):
    st.markdown('**Resumen ejecutivo del modelo**')
    snap = summary_snapshot(last['result_obj'], last['metrics'], last['type'])
    cols = st.columns(4)
    items = list(snap.items())
    for i, (k, v) in enumerate(items[:8]):
        txt = safe_num(v, digits=4) if isinstance(v, (float, int, np.floating, np.integer)) else str(v)
        cols[i % 4].metric(k, txt if txt != 'nan' else '—')

    st.markdown('**Tabla de coeficientes**')
    pretty = prettify_coef_table(last['coef_df'])
    st.dataframe(pretty, use_container_width=True, hide_index=True)

    with st.expander('Ver summary técnico completo', expanded=False):
        st.text(model_summary_text(last['result_obj'], last['type']))


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

            col_a, col_b, col_c = st.columns([1.4, 1.2, 1])
            if entity_col:
                graph_mode = col_a.selectbox(
                    "Formato del gráfico",
                    [
                        "Variables separadas · bancos como líneas",
                        "Bancos separados · variables como líneas",
                    ],
                    index=0,
                )
            else:
                graph_mode = "Variables separadas"
            independent_y = col_b.checkbox("Escala Y independiente", value=True)
            normalize_100 = col_c.checkbox("Normalizar índice 100", value=False)

            if entity_col:
                long = plot_df.melt(id_vars=["Fecha", entity_col], value_vars=selected, var_name="variable", value_name="valor")
                long = long.sort_values([entity_col, "variable", "Fecha"])
                if normalize_100:
                    def _index_100(s: pd.Series) -> pd.Series:
                        valid = s.dropna()
                        if valid.empty or valid.iloc[0] == 0:
                            return s * np.nan
                        return s / valid.iloc[0] * 100
                    long["valor"] = long.groupby([entity_col, "variable"], sort=False)["valor"].transform(_index_100)
                    y_title = "Índice 100"
                else:
                    y_title = "valor"

                if graph_mode.startswith("Variables separadas"):
                    n_cols = 2 if len(selected) > 1 else 1
                    height = max(520, 330 * math.ceil(len(selected) / n_cols))
                    fig = px.line(
                        long,
                        x="Fecha",
                        y="valor",
                        color=entity_col,
                        facet_col="variable",
                        facet_col_wrap=n_cols,
                        title="Series por variable",
                        labels={"valor": y_title},
                    )
                else:
                    bancos = long[entity_col].nunique()
                    height = max(520, 280 * bancos)
                    fig = px.line(
                        long,
                        x="Fecha",
                        y="valor",
                        color="variable",
                        facet_row=entity_col,
                        title="Series por banco",
                        labels={"valor": y_title},
                    )
            else:
                long = plot_df.melt(id_vars=["Fecha"], value_vars=selected, var_name="variable", value_name="valor")
                long = long.sort_values(["variable", "Fecha"])
                if normalize_100:
                    def _index_100_ts(s: pd.Series) -> pd.Series:
                        valid = s.dropna()
                        if valid.empty or valid.iloc[0] == 0:
                            return s * np.nan
                        return s / valid.iloc[0] * 100
                    long["valor"] = long.groupby("variable", sort=False)["valor"].transform(_index_100_ts)
                    y_title = "Índice 100"
                else:
                    y_title = "valor"
                n_cols = 2 if len(selected) > 1 else 1
                height = max(520, 330 * math.ceil(len(selected) / n_cols))
                fig = px.line(
                    long,
                    x="Fecha",
                    y="valor",
                    color="variable",
                    facet_col="variable",
                    facet_col_wrap=n_cols,
                    title="Series temporales",
                    labels={"valor": y_title},
                )

            if independent_y:
                fig.update_yaxes(matches=None, showticklabels=True)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig.update_layout(
                height=height,
                margin=dict(l=40, r=30, t=70, b=45),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
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
        render_presentable_summary(last)

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
            fig = plot_model_comparison(
                model_df=model_df,
                target=target,
                series_cols=[target, est_col],
                panel=(last["type"] == "Panel"),
                title=f"{target} real vs estimado",
            )
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
                fig = plot_model_comparison(
                    model_df=cf_df,
                    target=target,
                    series_cols=cols_plot,
                    panel=(last["type"] == "Panel"),
                    title=f"Contrafactual: {target} · {cf_name}",
                )
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
