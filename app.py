from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dss_arima_core import (
    DEFAULT_TEST_MONTHS,
    TARGET_MAPE,
    TARGET_SERVICE,
    analyze_sku,
    base_kpis,
    batch_analyze,
    batch_summary,
    data_quality_summary,
    dynamic_targets,
    error_ranking,
    load_workbook_data,
    replenishment_for_sku,
    report_excel_bytes,
    sku_catalog,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "BD_DSS_12_MESES.xlsx"
PRECOMP_RESULTS_PATH = BASE_DIR / "data" / "precomputed_results.csv"
PRECOMP_DETAILS_PATH = BASE_DIR / "data" / "precomputed_details.json"
PRECOMP_SUMMARY_PATH = BASE_DIR / "data" / "precomputed_summary.json"

st.set_page_config(
    page_title="DSS ARIMA · Inventarios Metalmecánicos",
    page_icon="📦",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.15rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {background:#F7FAFC;border:1px solid #DCE6EE;padding:14px;border-radius:12px;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {color:#334155 !important;}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {color:#0F172A !important;}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {color:#475569 !important;}
.small-note {font-size:0.88rem;color:#5F6F7F;}
.ok {color:#137333;font-weight:700}.warn {color:#A15C00;font-weight:700}.bad {color:#B3261E;font-weight:700}
</style>
""",
    unsafe_allow_html=True,
)

st.title("DSS de Pronóstico y Reposición de Inventarios")
st.caption("ARIMA · Punto de reorden · Stock de seguridad · KPIs · Alertas · Base oficial de 12 meses")


@st.cache_resource(show_spinner=False)
def load_default_assets():
    wb = load_workbook_data(DEFAULT_DATA_PATH)
    result = pd.read_csv(PRECOMP_RESULTS_PATH)
    with PRECOMP_DETAILS_PATH.open(encoding="utf-8") as f:
        details = json.load(f)
    with PRECOMP_SUMMARY_PATH.open(encoding="utf-8") as f:
        summary = json.load(f)
    return wb, result, details, summary


@st.cache_resource(show_spinner=False)
def load_custom_workbook(raw: bytes):
    return load_workbook_data(raw)


with st.sidebar:
    st.header("Acceso")
    role = st.selectbox("Perfil de usuario", ["Gerencia", "Compras", "Almacén", "Analista"])
    st.caption("Control por perfil del prototipo académico; no sustituye autenticación corporativa.")
    st.divider()
    uploaded = st.file_uploader(
        "Base Excel alternativa (.xlsx)",
        type=["xlsx"],
        help="Opcional. Si no subes nada, el DSS abre automáticamente la base oficial de 12 meses.",
    )

if uploaded is None:
    wb, precomputed_result, precomputed_details, precomputed_summary = load_default_assets()
    use_precomputed = True
    source_label = "Base oficial incluida · 12 meses"
    base_key = "DEFAULT_12M"
else:
    raw = uploaded.getvalue()
    try:
        wb = load_custom_workbook(raw)
    except Exception as e:
        st.error(f"No se pudo cargar la base: {e}")
        st.stop()
    precomputed_result = pd.DataFrame()
    precomputed_details = {}
    precomputed_summary = {}
    use_precomputed = False
    source_label = f"Base cargada por usuario · {uploaded.name}"
    base_key = hashlib.sha256(raw).hexdigest()

kpi = base_kpis(wb)
quality = data_quality_summary(wb)
targets = dynamic_targets(wb)
catalog = sku_catalog(wb)

allowed = {
    "Gerencia": ["Inicio", "Pronóstico ARIMA", "KPIs", "Análisis completo", "Reportes"],
    "Compras": ["Inicio", "Pronóstico ARIMA", "Reposición", "Alertas", "Análisis completo", "Reportes"],
    "Almacén": ["Inicio", "Reposición", "Alertas", "KPIs"],
    "Analista": ["Inicio", "Datos", "Pronóstico ARIMA", "Reposición", "Alertas", "KPIs", "Análisis completo", "Reportes"],
}
page = st.sidebar.radio("Módulo", allowed[role])

st.sidebar.divider()
st.sidebar.write(f"**Fuente:** {source_label}")
st.sidebar.write(f"**Base:** {quality['skus']} SKU · {quality['months']} meses · {quality['rows']:,} registros")
st.sidebar.write(f"**Periodo:** {quality['period_start']:%b %Y} – {quality['period_end']:%b %Y}")
st.sidebar.write(f"**Validación:** rolling-origin de {DEFAULT_TEST_MONTHS} meses")
st.sidebar.write("**Meta predictiva:** MAPE < 15 %")
st.sidebar.write("**Meta servicio:** ≥ 99.3 %")


def sku_selector(label="SKU"):
    opts = catalog["SKU"].astype(str).tolist()
    return st.selectbox(
        label,
        opts,
        format_func=lambda x: f"{x} · {catalog.loc[catalog['SKU'].eq(x), 'Producto'].iloc[0]}",
    )


@st.cache_data(show_spinner="Calculando ARIMA para la base cargada...")
def analyze_custom_cached(sku: str, key: str):
    # key fuerza la invalidación del caché cuando cambia la base subida.
    return analyze_sku(wb, sku, test_months=DEFAULT_TEST_MONTHS, forecast_months=3)


def detail_from_precomputed(sku: str):
    d = precomputed_details[sku]
    return {
        "series": pd.Series(d["series_values"], index=pd.to_datetime(d["series_index"])),
        "validation_index": pd.to_datetime(d["validation_index"]),
        "forecast": pd.Series(d["forecast_values"], index=pd.to_datetime(d["forecast_index"])),
        "best": {
            "order": tuple(d["order"]),
            "mode": d["mode"],
            "actual": np.asarray(d["actual_test"], dtype=float),
            "pred": np.asarray(d["pred_test"], dtype=float),
            "mape": float(d["mape"]),
            "wmape": float(d["wmape"]),
            "mae": float(d["mae"]),
            "rmse": float(d["rmse"]),
        },
        "diagnostics": d["diagnostics"],
        "top_candidates": pd.DataFrame(d["top_candidates"]),
        "replenishment": d["replenishment"],
    }


def get_analysis(sku: str):
    if use_precomputed:
        return detail_from_precomputed(sku)
    return analyze_custom_cached(sku, base_key)


# ---------------- Inicio ----------------
if page == "Inicio":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demanda total", f"{kpi['demand']:,.0f} und")
    c2.metric("Stockouts históricos", f"{kpi['stockouts']}")
    c3.metric("Demanda insatisfecha", f"{kpi['unmet']:,.0f} und")
    c4.metric("Nivel de servicio", f"{kpi['service']*100:.2f}%", delta=f"Meta {TARGET_SERVICE*100:.1f}%")

    st.subheader("Flujo de decisión del DSS")
    st.markdown("**Datos → preparación de SKU → ARIMA → demanda en lead time → stock de seguridad → punto de reorden → alertas/KPIs → decisión de compras**")

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown("#### Línea base de la ventana cargada")
        st.dataframe(
            pd.DataFrame({
                "Indicador": ["Demanda total", "Ventas atendidas", "Demanda insatisfecha", "Stockouts", "Nivel de servicio"],
                "Valor": [f"{kpi['demand']:.0f}", f"{kpi['sales']:.0f}", f"{kpi['unmet']:.0f}", f"{kpi['stockouts']}", f"{kpi['service']*100:.2f}%"],
            }),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Las metas operativas se recalculan proporcionalmente para la ventana de 12 meses sin alterar los criterios originales de la tesis.")
    with col2:
        st.markdown("#### Calidad de datos")
        st.dataframe(
            pd.DataFrame({
                "Control": ["Registros", "SKU", "Meses", "Duplicados SKU-mes", "Faltantes críticos"],
                "Resultado": [quality['rows'], quality['skus'], quality['months'], quality['duplicates_sku_month'], quality['missing_critical']],
            }),
            hide_index=True,
            use_container_width=True,
        )
        if quality['duplicates_sku_month'] == 0 and quality['missing_critical'] == 0:
            st.success("La base cumple los controles estructurales principales.")

    if use_precomputed:
        st.success("La base oficial y los resultados de los 30 SKU ya están precalculados. El usuario puede navegar directamente sin volver a ejecutar Google Colab.")

# ---------------- Datos ----------------
elif page == "Datos":
    st.subheader("Datos históricos y selección de SKU")
    st.write("El DSS valida la estructura antes de utilizar el motor predictivo.")
    st.dataframe(catalog, use_container_width=True, hide_index=True)
    st.download_button("Descargar catálogo CSV", catalog.to_csv(index=False).encode("utf-8-sig"), "catalogo_sku.csv", "text/csv")

# ---------------- Pronóstico ----------------
elif page == "Pronóstico ARIMA":
    st.subheader("Pronóstico ARIMA por SKU")
    st.caption(f"La base de 12 meses se valida mediante rolling-origin de {DEFAULT_TEST_MONTHS} meses. Con 12 observaciones se restringe la complejidad de ARIMA para evitar sobreajuste.")
    sku = sku_selector()
    a = get_analysis(sku)
    best = a["best"]
    series = a["series"]
    forecast = a["forecast"]
    dg = a["diagnostics"]

    mode_label = {"none": "Serie original", "log1p": "Transformación log1p"}.get(best["mode"], best["mode"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ARIMA seleccionado", str(tuple(best["order"])))
    c2.metric("MAPE rolling-origin", f"{best['mape']:.2f}%", delta="Cumple meta" if best['mape'] < TARGET_MAPE else "Meta pendiente")
    c3.metric("WMAPE", f"{best['wmape']:.2f}%")
    c4.metric("Tratamiento", mode_label)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers", name="Demanda histórica"))
    fig.add_trace(go.Scatter(x=a["validation_index"], y=best["pred"], mode="lines+markers", name="Predicción test", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Forecast", line=dict(dash="dash")))
    fig.update_layout(height=430, xaxis_title="Periodo", yaxis_title="Demanda (unidades)", legend_orientation="h", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Candidatos evaluados")
    st.dataframe(a["top_candidates"].style.format({"MAPE_%":"{:.2f}", "WMAPE_%":"{:.2f}", "MAE":"{:.2f}", "RMSE":"{:.2f}"}), hide_index=True, use_container_width=True)
    st.markdown("#### Diagnóstico de la serie")
    cv = dg.get("cv")
    cv_txt = "N/D" if cv is None or not np.isfinite(cv) else f"{cv:.2f}"
    st.write(f"Observaciones: **{int(dg['n'])}** · CV: **{cv_txt}** · Meses con cero: **{dg['zero_ratio']*100:.1f}%** · Rango: **{dg['min']:.0f}–{dg['max']:.0f} und**")

# ---------------- Reposición ----------------
elif page == "Reposición":
    st.subheader("Reposición: punto de reorden y stock de seguridad")
    sku = sku_selector()
    a = get_analysis(sku)
    fm1 = float(a["forecast"].iloc[0])
    rep = a.get("replenishment") if use_precomputed else replenishment_for_sku(wb, sku, fm1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast próximo mes", f"{fm1:.1f} und")
    c2.metric("Demanda durante lead time", f"{float(rep['demanda_lead_time']):.1f} und")
    c3.metric("Stock seguridad", f"{float(rep['stock_seguridad']):.0f} und")
    c4.metric("Punto de reorden", f"{float(rep['punto_reorden']):.0f} und")

    c5, c6, c7 = st.columns(3)
    stock = rep.get("stock_actual")
    suggested = rep.get("compra_sugerida")
    c5.metric("Stock actual", "N/D" if stock is None or not np.isfinite(float(stock)) else f"{float(stock):.0f} und")
    c6.metric("Lead time", f"{float(rep['lead_time']):.0f} días")
    c7.metric("Compra sugerida", "N/D" if suggested is None or not np.isfinite(float(suggested)) else f"{float(suggested):.0f} und")

    estado = rep.get("estado", "")
    if estado == "REPOSICIÓN INMEDIATA":
        st.error(f"🔴 {estado}: el stock actual alcanzó o cayó por debajo del punto de reorden.")
    elif estado == "VIGILAR":
        st.warning("🟠 VIGILAR: el stock se encuentra próximo al punto de reorden.")
    elif estado == "OK":
        st.success("🟢 OK: inventario por encima del umbral preventivo.")
    else:
        st.info(str(estado))
    st.caption("La recomendación apoya la decisión; la orden de compra continúa siendo responsabilidad del usuario.")

# ---------------- Alertas ----------------
elif page == "Alertas":
    st.subheader("Alertas y recomendaciones")
    if use_precomputed and not precomputed_result.empty:
        cols = ["SKU", "Producto", "ABC", "XYZ", "Stock_Actual", "Punto_Reorden", "Estado", "Compra_Sugerida"]
        st.dataframe(precomputed_result[[c for c in cols if c in precomputed_result.columns]].sort_values("Estado"), hide_index=True, use_container_width=True)
        st.caption("Las alertas corresponden al forecast ARIMA precalculado con la base oficial de 12 meses.")
    else:
        cat = catalog.copy()
        if {"Stock_Actual", "RO_Base"}.issubset(cat.columns):
            prelim = cat[["SKU", "Producto", "Clase ABC", "Clase XYZ", "Stock_Actual", "RO_Base", "Stockouts_Historicos"]].copy()
            prelim["Alerta preliminar"] = np.where(prelim["Stock_Actual"] <= prelim["RO_Base"], "REVISAR / REPONER", "OK")
            st.dataframe(prelim, hide_index=True, use_container_width=True)
        st.info("Con una base alternativa, ejecuta el análisis completo para actualizar todas las alertas con ARIMA.")

# ---------------- KPIs ----------------
elif page == "KPIs":
    st.subheader("KPIs de la ventana de 12 meses")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stockouts base", f"{kpi['stockouts']}", delta=f"Meta derivada ≤ {targets['stockouts_target']}")
    c2.metric("Demanda insatisfecha", f"{kpi['unmet']:.0f}", delta=f"Meta derivada ≤ {targets['unmet_target']}")
    c3.metric("Nivel de servicio", f"{kpi['service']*100:.2f}%", delta="Meta ≥ 99.3%")

    st.markdown("#### Coherencia con la tesis")
    st.write("La reducción objetivo de stockouts se mantiene en **≥30 %**. La reducción proporcional de demanda insatisfecha conserva el criterio histórico 98→74. El nivel de servicio mantiene la meta de **≥99.3 %** y el MAPE la meta de **<15 %**.")
    st.info("Los KPIs de esta pantalla son línea base de 12 meses; no se presentan como impacto real posterior a una implementación empresarial.")

# ---------------- Análisis completo ----------------
elif page == "Análisis completo":
    st.subheader("Análisis completo de los 30 SKU")
    if use_precomputed:
        result = precomputed_result.copy()
        summary = precomputed_summary
        st.success("Resultados precalculados: no es necesario ejecutar nuevamente el motor para consultar la base oficial.")
    else:
        if "custom_batch_result" not in st.session_state:
            st.info("Esta es una base alternativa. Para obtener resultados de todos los SKU debes ejecutar el análisis una sola vez; luego quedará en caché durante la sesión.")
            if st.button("Ejecutar análisis de los 30 SKU", type="primary"):
                with st.spinner("Procesando los 30 SKU..."):
                    st.session_state["custom_batch_result"] = batch_analyze(wb, only_critical=False, test_months=DEFAULT_TEST_MONTHS)
        result = st.session_state.get("custom_batch_result", pd.DataFrame())
        summary = batch_summary(result) if not result.empty else {}

    if isinstance(result, pd.DataFrame) and not result.empty:
        display_cols = [c for c in result.columns if c not in ["Actual_Test", "Pred_Test"]]
        st.dataframe(result[display_cols], hide_index=True, use_container_width=True)
        if summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ejecución técnica", f"{summary['tasa_ejecucion_pct']:.1f}%", delta=summary['cp4_estado_tecnico'])
            c2.metric("SKU con MAPE <15%", f"{summary['sku_mape_lt15']}/{summary['sku_analizados']}")
            c3.metric("MAPE promedio", f"{summary['mape_promedio']:.2f}%", delta=summary['estado_meta_predictiva'])
            c4.metric("WMAPE global", f"{summary['wmape_global']:.2f}%")
            st.caption("El logro técnico del motor y la meta predictiva son criterios distintos. El DSS reporta ambos sin modificar resultados para forzar la meta.")

            st.markdown("#### SKU que más elevan el error")
            rank = result.dropna(subset=["MAPE_%"]).sort_values("MAPE_%", ascending=False)
            rank_cols = [c for c in ["SKU", "Producto", "MAPE_%", "WMAPE_%", "MAE", "RMSE", "Tratamiento", "ARIMA", "CV_Demanda", "Porc_Ceros"] if c in rank.columns]
            st.dataframe(rank[rank_cols].head(10).style.format({"MAPE_%":"{:.2f}", "WMAPE_%":"{:.2f}", "MAE":"{:.2f}", "RMSE":"{:.2f}"}), hide_index=True, use_container_width=True)

        st.download_button("Descargar resultados CSV", result[display_cols].to_csv(index=False).encode("utf-8-sig"), "resultado_dss_arima_12m.csv", "text/csv")

# ---------------- Reportes ----------------
elif page == "Reportes":
    st.subheader("Reportes")
    st.download_button("Línea base CSV", pd.DataFrame([kpi]).to_csv(index=False).encode("utf-8-sig"), "kpis_linea_base_12m.csv", "text/csv")
    st.download_button("Catálogo SKU CSV", catalog.to_csv(index=False).encode("utf-8-sig"), "catalogo_sku_12m.csv", "text/csv")
    if use_precomputed:
        st.download_button("Resultados precalculados CSV", precomputed_result.to_csv(index=False).encode("utf-8-sig"), "resultado_dss_arima_12m.csv", "text/csv")
    else:
        result = st.session_state.get("custom_batch_result", pd.DataFrame())
        if not result.empty:
            st.download_button("Resultados de la base cargada CSV", result.to_csv(index=False).encode("utf-8-sig"), "resultado_dss_arima_custom.csv", "text/csv")
        else:
            st.caption("Ejecuta primero el análisis completo para generar el reporte de una base alternativa.")

st.divider()
st.caption("DSS académico de pronóstico y reposición · Proyecto autónomo para Visual Studio/VS Code y despliegue web estable.")
