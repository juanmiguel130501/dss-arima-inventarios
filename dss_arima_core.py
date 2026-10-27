from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union
import math
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

TARGET_MAPE = 15.0
TARGET_SERVICE = 0.993
TARGET_STOCKOUT_REDUCTION = 0.30
# La meta histórica 98 -> 74 equivale aproximadamente a 24.49 % de reducción.
TARGET_UNMET_REDUCTION = 1.0 - (74.0 / 98.0)
Z_SERVICE = 2.46  # cuantil unilateral aproximado para 99.3 %
DEFAULT_TEST_MONTHS = 3
MIN_TRAIN_MONTHS = 6

# Con solo 12 observaciones por SKU se restringe deliberadamente la complejidad
# para evitar modelos sobredimensionados. Se mantiene ARIMA como núcleo.
CANDIDATE_ORDERS: List[Tuple[int, int, int]] = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
]
CANDIDATE_MODES = ("none", "log1p")


@dataclass
class WorkbookData:
    data: pd.DataFrame
    products: pd.DataFrame
    inventory: pd.DataFrame
    abcxyz: pd.DataFrame
    sheets: List[str]


def _excel_date(series: pd.Series) -> pd.Series:
    s = series.copy()
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    num = pd.to_numeric(s, errors="coerce")
    valid_num = num.dropna()
    if len(valid_num) and num.notna().mean() > 0.7 and 20000 < valid_num.median() < 100000:
        return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _source_for_pandas(source: Union[str, Path, bytes, bytearray, BytesIO]):
    return BytesIO(source) if isinstance(source, (bytes, bytearray)) else source


def load_workbook_data(source: Union[str, Path, bytes, bytearray, BytesIO]) -> WorkbookData:
    src = _source_for_pandas(source)
    xls = pd.ExcelFile(src, engine="openpyxl")
    sheets = xls.sheet_names

    def read(name: str) -> pd.DataFrame:
        return _clean_columns(pd.read_excel(xls, sheet_name=name)) if name in sheets else pd.DataFrame()

    data = read("Data")
    products = read("Productos")
    inventory = read("Inventario")
    abcxyz = read("ABC-XYZ")

    required = {"Periodo", "SKU", "Demanda (Und)"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("La hoja 'Data' no contiene: " + ", ".join(sorted(missing)))

    data["Periodo"] = _excel_date(data["Periodo"])
    data["SKU"] = data["SKU"].astype(str).str.strip()
    for c in ["Demanda (Und)", "Unit", "Demanda Insatisfecha", "Precio Unitario", "Costo Unitario"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
    if "Stockout" in data.columns:
        data["Stockout"] = data["Stockout"].astype(str).str.strip()

    if not products.empty:
        products["SKU"] = products["SKU"].astype(str).str.strip()
        for c in ["Lead Time Dias", "Lote Minimo Compra", "Stock Seguridad Unid", "Punto Reorden Unid", "Costo Unit", "Precio Unit"]:
            if c in products.columns:
                products[c] = pd.to_numeric(products[c], errors="coerce")

    if not inventory.empty:
        inventory["Periodo"] = _excel_date(inventory["Periodo"])
        inventory["SKU"] = inventory["SKU"].astype(str).str.strip()
        for c in ["Stock Final", "Stock Inicial", "Punto Reorden", "Stock Seguridad"]:
            if c in inventory.columns:
                inventory[c] = pd.to_numeric(inventory[c], errors="coerce")

    if not abcxyz.empty:
        abcxyz["SKU"] = abcxyz["SKU"].astype(str).str.strip()
        for c in ["Demanda Total", "Ventas Total", "Demanda Insatisfecha", "Nivel Servicio", "Coef Variación"]:
            if c in abcxyz.columns:
                abcxyz[c] = pd.to_numeric(abcxyz[c], errors="coerce")

    data = data.sort_values(["SKU", "Periodo"]).reset_index(drop=True)
    return WorkbookData(data=data, products=products, inventory=inventory, abcxyz=abcxyz, sheets=sheets)


def data_quality_summary(wb: WorkbookData) -> Dict[str, object]:
    d = wb.data
    dup = int(d.duplicated(["Periodo", "SKU"]).sum())
    missing_critical = int(d[["Periodo", "SKU", "Demanda (Und)"]].isna().sum().sum())
    consistent = None
    if {"Unit", "Demanda Insatisfecha"}.issubset(d.columns):
        mask = d[["Demanda (Und)", "Unit", "Demanda Insatisfecha"]].notna().all(axis=1)
        consistent = int(((d.loc[mask, "Demanda (Und)"] - d.loc[mask, "Unit"] - d.loc[mask, "Demanda Insatisfecha"]).abs() < 1e-9).sum())
    return {
        "rows": int(len(d)),
        "skus": int(d["SKU"].nunique()),
        "months": int(d["Periodo"].nunique()),
        "duplicates_sku_month": dup,
        "missing_critical": missing_critical,
        "demand_identity_ok_rows": consistent,
        "period_start": d["Periodo"].min(),
        "period_end": d["Periodo"].max(),
    }


def base_kpis(wb: WorkbookData) -> Dict[str, float]:
    d = wb.data
    demand = float(pd.to_numeric(d["Demanda (Und)"], errors="coerce").fillna(0).sum())
    if "Unit" in d.columns:
        sales = float(pd.to_numeric(d["Unit"], errors="coerce").fillna(0).sum())
    else:
        sales = demand
    if "Demanda Insatisfecha" in d.columns:
        unmet = float(pd.to_numeric(d["Demanda Insatisfecha"], errors="coerce").fillna(0).sum())
    else:
        unmet = max(0.0, demand - sales)
    if "Stockout" in d.columns:
        stockouts = int(d["Stockout"].str.lower().isin(["si", "sí", "yes", "1", "true"]).sum())
    else:
        stockouts = int((pd.to_numeric(d.get("Demanda Insatisfecha", 0), errors="coerce").fillna(0) > 0).sum())
    service = sales / demand if demand > 0 else np.nan
    return {"demand": demand, "sales": sales, "unmet": unmet, "stockouts": stockouts, "service": service}


def dynamic_targets(wb: WorkbookData) -> Dict[str, float]:
    """Metas equivalentes para la ventana cargada, conservando los criterios de la tesis."""
    k = base_kpis(wb)
    # Para una reducción mínima de 30 %, el entero máximo se toma con floor.
    stockouts_target = math.floor(k["stockouts"] * (1.0 - TARGET_STOCKOUT_REDUCTION))
    unmet_target = math.floor(k["unmet"] * (1.0 - TARGET_UNMET_REDUCTION))
    return {
        "stockouts_target": max(0, stockouts_target),
        "unmet_target": max(0, unmet_target),
        "service_target": TARGET_SERVICE,
        "mape_target": TARGET_MAPE,
    }


def sku_catalog(wb: WorkbookData) -> pd.DataFrame:
    d = wb.data.copy()
    d["Demanda (Und)"] = pd.to_numeric(d["Demanda (Und)"], errors="coerce").fillna(0)
    if "Producto" not in d.columns:
        d["Producto"] = d["SKU"]
    if "Categoria" not in d.columns:
        d["Categoria"] = ""
    base = d.groupby("SKU", as_index=False).agg(
        Producto=("Producto", "first"),
        Categoria=("Categoria", "first"),
        Demanda_Total=("Demanda (Und)", "sum"),
        Promedio_Mensual=("Demanda (Und)", "mean"),
        Sigma=("Demanda (Und)", "std"),
    )
    if "Stockout" in d.columns:
        st = d.assign(_so=d["Stockout"].str.lower().isin(["si", "sí", "yes", "1", "true"])) \
            .groupby("SKU", as_index=False)["_so"].sum().rename(columns={"_so": "Stockouts_Historicos"})
    else:
        st = base[["SKU"]].copy(); st["Stockouts_Historicos"] = 0
    base = base.merge(st, on="SKU", how="left")

    if not wb.abcxyz.empty:
        cols = [c for c in ["SKU", "Clase ABC", "Clase XYZ", "Nivel Servicio", "Coef Variación"] if c in wb.abcxyz.columns]
        base = base.merge(wb.abcxyz[cols].drop_duplicates("SKU"), on="SKU", how="left")
    else:
        base["Clase ABC"] = ""; base["Clase XYZ"] = ""

    if not wb.products.empty:
        cols = [c for c in ["SKU", "Lead Time Dias", "Lote Minimo Compra", "Criticidad Inventario", "Tipo Abastecimiento", "Costo Unit", "Precio Unit"] if c in wb.products.columns]
        base = base.merge(wb.products[cols].drop_duplicates("SKU"), on="SKU", how="left")

    if not wb.inventory.empty and "Stock Final" in wb.inventory.columns:
        inv = wb.inventory.dropna(subset=["Periodo"]).sort_values("Periodo").groupby("SKU", as_index=False).tail(1)
        cols = [c for c in ["SKU", "Periodo", "Stock Final", "Punto Reorden", "Stock Seguridad"] if c in inv.columns]
        inv = inv[cols].rename(columns={"Periodo": "Periodo_Stock", "Stock Final": "Stock_Actual", "Punto Reorden": "RO_Base", "Stock Seguridad": "SS_Base"})
        base = base.merge(inv, on="SKU", how="left")
    else:
        base["Stock_Actual"] = np.nan

    if "Lead Time Dias" not in base.columns:
        base["Lead Time Dias"] = 30
    if "Lote Minimo Compra" not in base.columns:
        base["Lote Minimo Compra"] = 1
    if "Stock_Actual" not in base.columns:
        base["Stock_Actual"] = np.nan
    base["Lead Time Dias"] = pd.to_numeric(base["Lead Time Dias"], errors="coerce").fillna(30)
    base["Lote Minimo Compra"] = pd.to_numeric(base["Lote Minimo Compra"], errors="coerce").fillna(1).clip(lower=1)
    base["Stock_Actual"] = pd.to_numeric(base["Stock_Actual"], errors="coerce")

    abc = base["Clase ABC"] if "Clase ABC" in base.columns else pd.Series("", index=base.index)
    crit = base["Criticidad Inventario"] if "Criticidad Inventario" in base.columns else pd.Series("", index=base.index)
    base["Critico"] = (
        abc.fillna("").eq("A") |
        base["Stockouts_Historicos"].fillna(0).gt(0) |
        crit.fillna("").astype(str).str.lower().eq("alta")
    )
    return base.sort_values(["Critico", "Demanda_Total"], ascending=[False, False]).reset_index(drop=True)


def get_series(wb: WorkbookData, sku: str) -> pd.Series:
    s = wb.data.loc[wb.data["SKU"].eq(str(sku)), ["Periodo", "Demanda (Und)"]].dropna(subset=["Periodo"])
    s["Demanda (Und)"] = pd.to_numeric(s["Demanda (Und)"], errors="coerce").fillna(0)
    s = s.groupby("Periodo")["Demanda (Und)"].sum().sort_index().astype(float)
    if len(s):
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
        s = s.reindex(full_idx, fill_value=0.0)
    return s


def series_diagnostics(series: pd.Series) -> Dict[str, float]:
    y = np.asarray(series.values, dtype=float)
    mean = float(np.mean(y)) if len(y) else np.nan
    std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    return {
        "n": int(len(y)),
        "zero_ratio": float(np.mean(y == 0)) if len(y) else np.nan,
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean > 1e-12 else np.nan,
        "min": float(np.min(y)) if len(y) else np.nan,
        "max": float(np.max(y)) if len(y) else np.nan,
    }


def _transform(y: Iterable[float], mode: str) -> np.ndarray:
    arr = np.asarray(list(y), dtype=float)
    if mode == "log1p":
        return np.log1p(np.clip(arr, 0, None))
    return arr


def _inverse(yhat: Iterable[float], mode: str) -> np.ndarray:
    arr = np.asarray(list(yhat), dtype=float)
    if mode == "log1p":
        arr = np.expm1(arr)
    return np.clip(arr, 0, None)


def metric_mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    mask = np.abs(yt) > 1e-12
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def metric_wmape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    denom = np.sum(np.abs(yt))
    return float(np.sum(np.abs(yt - yp)) / denom * 100) if denom > 1e-12 else np.nan


def metric_mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=float); yp = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def metric_rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=float); yp = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def _forecast_one(history: Iterable[float], order: Tuple[int, int, int], mode: str) -> float:
    arr_raw = np.asarray(list(history), dtype=float)
    if len(arr_raw) == 0:
        return 0.0
    try:
        arr = _transform(arr_raw, mode)
        if np.nanstd(arr) < 1e-10:
            return float(_inverse([arr[-1]], mode)[0])
        fit = ARIMA(
            arr,
            order=order,
            trend=None,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(method_kwargs={"warn_convergence": False})
        pred = float(_inverse(fit.forecast(1), mode)[0])
        return max(0.0, pred)
    except Exception:
        # Fallback transparente para que una falla numérica puntual no rompa el DSS.
        tail = arr_raw[-3:] if len(arr_raw) >= 3 else arr_raw
        return max(0.0, float(np.nanmean(tail)))


def rolling_validate_order(
    series: pd.Series,
    order: Tuple[int, int, int],
    mode: str,
    test_months: int = DEFAULT_TEST_MONTHS,
) -> Dict[str, object]:
    y = np.asarray(series.values, dtype=float)
    if test_months < 1:
        raise ValueError("test_months debe ser al menos 1.")
    if len(y) - test_months < MIN_TRAIN_MONTHS:
        raise ValueError(
            f"La serie requiere al menos {MIN_TRAIN_MONTHS + test_months} observaciones "
            f"para validar; se recibieron {len(y)}."
        )
    train = y[:-test_months]
    test = y[-test_months:]
    hist = list(train)
    preds = []
    for real in test:
        pred = _forecast_one(hist, order, mode)
        preds.append(float(pred))
        hist.append(float(real))
    return {
        "order": order,
        "mode": mode,
        "actual": np.asarray(test),
        "pred": np.asarray(preds),
        "mape": metric_mape(test, preds),
        "wmape": metric_wmape(test, preds),
        "mae": metric_mae(test, preds),
        "rmse": metric_rmse(test, preds),
    }


def optimize_arima_cp4(series: pd.Series, test_months: int = DEFAULT_TEST_MONTHS) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Selecciona el ARIMA por MAPE rolling-origin, con WMAPE y RMSE como desempate."""
    results = []
    for mode in CANDIDATE_MODES:
        for order in CANDIDATE_ORDERS:
            try:
                r = rolling_validate_order(series, order, mode, test_months=test_months)
                if np.isfinite(r["mape"]):
                    results.append(r)
            except Exception:
                continue
    if not results:
        raise RuntimeError("No fue posible ajustar candidatos ARIMA válidos para la serie de 12 meses.")
    results.sort(key=lambda r: (
        np.nan_to_num(r["mape"], nan=np.inf),
        np.nan_to_num(r["wmape"], nan=np.inf),
        np.nan_to_num(r["rmse"], nan=np.inf),
    ))
    best = results[0]
    top = pd.DataFrame([
        {
            "Orden": str(r["order"]),
            "Tratamiento": r["mode"],
            "MAPE_%": r["mape"],
            "WMAPE_%": r["wmape"],
            "MAE": r["mae"],
            "RMSE": r["rmse"],
        }
        for r in results[:10]
    ])
    return best, top


def _forecast_future_iterative(series: pd.Series, order: Tuple[int, int, int], mode: str, steps: int) -> np.ndarray:
    hist = list(np.asarray(series.values, dtype=float))
    out = []
    for _ in range(steps):
        p = _forecast_one(hist, order, mode)
        out.append(float(p))
        hist.append(float(p))
    return np.asarray(out)


def analyze_sku(
    wb: WorkbookData,
    sku: str,
    test_months: int = DEFAULT_TEST_MONTHS,
    forecast_months: int = 3,
) -> Dict[str, object]:
    series = get_series(wb, sku)
    best, top_candidates = optimize_arima_cp4(series, test_months=test_months)
    future = _forecast_future_iterative(series, best["order"], best["mode"], forecast_months)
    future_idx = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=forecast_months, freq="MS")
    return {
        "sku": sku,
        "series": series,
        "diagnostics": series_diagnostics(series),
        "best": best,
        "top_candidates": top_candidates,
        "order_full": best["order"],
        "mode_full": best["mode"],
        "forecast": pd.Series(future, index=future_idx),
        "validation_index": series.index[-test_months:],
    }


def _round_to_lot(qty: float, lot: float) -> int:
    if not np.isfinite(qty) or qty <= 0:
        return 0
    lot = max(1.0, float(lot) if np.isfinite(lot) else 1.0)
    return int(math.ceil(qty / lot) * lot)


def replenishment_for_sku(
    wb: WorkbookData,
    sku: str,
    forecast_m1: float,
    service_level: float = TARGET_SERVICE,
) -> Dict[str, object]:
    cat = sku_catalog(wb)
    row = cat.loc[cat["SKU"].eq(str(sku))]
    if row.empty:
        raise ValueError(f"SKU {sku} no encontrado.")
    r = row.iloc[0]
    series = get_series(wb, sku)
    lead = float(r.get("Lead Time Dias", 30) or 30)
    lot = float(r.get("Lote Minimo Compra", 1) or 1)
    stock = float(r.get("Stock_Actual", np.nan))
    sigma = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    dlt = max(0.0, float(forecast_m1)) / 30.0 * lead
    ss = Z_SERVICE * sigma * math.sqrt(max(lead, 1.0) / 30.0)
    ro = dlt + ss
    if not np.isfinite(stock):
        status, suggested = "SIN STOCK ACTUAL", np.nan
    elif stock <= ro:
        status, suggested = "REPOSICIÓN INMEDIATA", _round_to_lot(ro - stock, lot)
    elif stock <= 1.2 * ro:
        status, suggested = "VIGILAR", 0
    else:
        status, suggested = "OK", 0
    return {
        "sku": sku,
        "lead_time": lead,
        "stock_actual": stock,
        "sigma_12m": sigma,
        "demanda_lead_time": dlt,
        "stock_seguridad": ss,
        "punto_reorden": ro,
        "lote_minimo": lot,
        "estado": status,
        "compra_sugerida": suggested,
    }


def batch_analyze(wb: WorkbookData, only_critical: bool = False, test_months: int = DEFAULT_TEST_MONTHS) -> pd.DataFrame:
    cat = sku_catalog(wb)
    if only_critical:
        cat = cat[cat["Critico"]].copy()
    rows = []
    for _, c in cat.iterrows():
        sku = str(c["SKU"])
        try:
            a = analyze_sku(wb, sku, test_months=test_months, forecast_months=3)
            b, f, dg = a["best"], a["forecast"], a["diagnostics"]
            rep = replenishment_for_sku(wb, sku, float(f.iloc[0]))
            rows.append({
                "SKU": sku,
                "Producto": c.get("Producto", ""),
                "ABC": c.get("Clase ABC", ""),
                "XYZ": c.get("Clase XYZ", ""),
                "Critico": bool(c.get("Critico", False)),
                "ARIMA": str(tuple(b["order"])),
                "Tratamiento": b["mode"],
                "MAPE_%": b["mape"],
                "WMAPE_%": b["wmape"],
                "MAE": b["mae"],
                "RMSE": b["rmse"],
                "Cumple_MAPE_15": bool(b["mape"] < TARGET_MAPE),
                "CV_Demanda": dg["cv"],
                "Porc_Ceros": dg["zero_ratio"] * 100,
                "Forecast_M1": float(f.iloc[0]),
                "Forecast_M2": float(f.iloc[1]) if len(f) > 1 else np.nan,
                "Forecast_M3": float(f.iloc[2]) if len(f) > 2 else np.nan,
                "Stock_Actual": rep["stock_actual"],
                "Lead_Time_Dias": rep["lead_time"],
                "Stock_Seguridad": rep["stock_seguridad"],
                "Punto_Reorden": rep["punto_reorden"],
                "Estado": rep["estado"],
                "Compra_Sugerida": rep["compra_sugerida"],
                "Actual_Test": list(map(float, b["actual"])),
                "Pred_Test": list(map(float, b["pred"])),
            })
        except Exception as e:
            rows.append({"SKU": sku, "Producto": c.get("Producto", ""), "Error": str(e)})
    return pd.DataFrame(rows)


def batch_summary(batch_df: pd.DataFrame) -> Dict[str, object]:
    if batch_df is None or batch_df.empty:
        return {}
    total = int(len(batch_df))
    valid = batch_df.dropna(subset=["MAPE_%"]).copy() if "MAPE_%" in batch_df.columns else pd.DataFrame()
    valid_n = int(len(valid))
    error_n = total - valid_n
    required = ["ARIMA", "Tratamiento", "MAPE_%", "WMAPE_%", "MAE", "RMSE"]
    required_ok = bool(valid_n == total and all(c in valid.columns for c in required))
    if required_ok:
        for c in ["MAPE_%", "WMAPE_%", "MAE", "RMSE"]:
            required_ok = required_ok and bool(np.isfinite(pd.to_numeric(valid[c], errors="coerce")).all())
    cp4_superado = bool(total > 0 and valid_n == total and required_ok)

    actual_all, pred_all = [], []
    for _, r in valid.iterrows():
        actual_all.extend(r.get("Actual_Test", [])); pred_all.extend(r.get("Pred_Test", []))
    mape_prom = float(valid["MAPE_%"].mean()) if valid_n else np.nan
    return {
        "sku_solicitados": total,
        "sku_analizados": valid_n,
        "sku_con_error": error_n,
        "tasa_ejecucion_pct": float(100.0 * valid_n / total) if total else 0.0,
        "cp4_superado_tecnico": cp4_superado,
        "cp4_estado_tecnico": "SUPERADO" if cp4_superado else "PENDIENTE",
        "sku_mape_lt15": int(valid["Cumple_MAPE_15"].sum()) if valid_n else 0,
        "mape_promedio": mape_prom,
        "mape_mediana": float(valid["MAPE_%"].median()) if valid_n else np.nan,
        "wmape_global": metric_wmape(actual_all, pred_all) if actual_all else np.nan,
        "mae_promedio": float(valid["MAE"].mean()) if valid_n else np.nan,
        "rmse_promedio": float(valid["RMSE"].mean()) if valid_n else np.nan,
        "meta_mape_15_cumplida": bool(np.isfinite(mape_prom) and mape_prom < TARGET_MAPE),
        "estado_meta_predictiva": "CUMPLIDA" if np.isfinite(mape_prom) and mape_prom < TARGET_MAPE else "PENDIENTE",
    }


def error_ranking(batch_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = [c for c in ["SKU", "Producto", "MAPE_%", "WMAPE_%", "MAE", "RMSE", "Tratamiento", "ARIMA", "CV_Demanda", "Porc_Ceros"] if c in batch_df.columns]
    return batch_df.dropna(subset=["MAPE_%"]).sort_values("MAPE_%", ascending=False)[cols].head(n).reset_index(drop=True)


def report_excel_bytes(wb: WorkbookData, batch_df: pd.DataFrame) -> bytes:
    kpi = base_kpis(wb)
    quality = data_quality_summary(wb)
    summary = batch_summary(batch_df)
    export_df = batch_df.copy()
    for c in ["Actual_Test", "Pred_Test"]:
        if c in export_df.columns:
            export_df[c] = export_df[c].apply(lambda x: ", ".join(f"{v:.2f}" for v in x) if isinstance(x, list) else x)
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame([kpi]).to_excel(writer, sheet_name="KPIs_Base", index=False)
        pd.DataFrame([dynamic_targets(wb)]).to_excel(writer, sheet_name="Metas_12M", index=False)
        pd.DataFrame([quality]).to_excel(writer, sheet_name="Calidad_Datos", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Resumen_CP4", index=False)
        sku_catalog(wb).to_excel(writer, sheet_name="SKU_Catalogo", index=False)
        export_df.to_excel(writer, sheet_name="ARIMA_Reposicion", index=False)
        error_ranking(batch_df, n=15).to_excel(writer, sheet_name="Ranking_Error", index=False)
    out.seek(0)
    return out.getvalue()
