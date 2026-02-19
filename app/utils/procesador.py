# =============================================================
# ARCHIVO: app/utils/procesador.py
# DESCRIPCIÓN: Procesa los datos del cliente en tiempo real
#              sin importar cuántas columnas extras tenga
# =============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def procesar_datos(clientes: pd.DataFrame, ventas: pd.DataFrame) -> dict:
    """
    Recibe los datos del cliente y genera todos los análisis.
    Ignora columnas extras automáticamente.
    Retorna un diccionario con todos los resultados.
    """

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/outputs", exist_ok=True)

    # ── COLUMNAS MÍNIMAS REQUERIDAS ────────────────────
    # Tomamos solo lo que necesitamos, ignoramos el resto
    cols_clientes = ["cliente_id", "tipo_cliente"]
    cols_ventas   = ["venta_id", "cliente_id", "producto", "precio", "fecha_venta"]

    clientes = clientes[
        [c for c in cols_clientes if c in clientes.columns]
    ].copy()

    ventas = ventas[
        [c for c in cols_ventas if c in ventas.columns]
    ].copy()

    # ── CONVERTIR FECHAS ───────────────────────────────
    ventas["fecha_venta"] = pd.to_datetime(ventas["fecha_venta"], errors="coerce")
    ventas = ventas.dropna(subset=["fecha_venta"])

    # ── MERGE ──────────────────────────────────────────
    df = ventas.merge(clientes, on="cliente_id", how="left")

    # ── ENCODING ──────────────────────────────────────
    le = LabelEncoder()
    df["producto_encoded"] = le.fit_transform(df["producto"].fillna("Otro"))

    # ── MÓDULO 1: RENTABILIDAD ─────────────────────────
    rentabilidad = (
        df.groupby("producto")
        .agg(
            total_ingresos  = ("precio", "sum"),
            total_ventas    = ("venta_id", "count"),
            precio_promedio = ("precio", "mean"),
            precio_maximo   = ("precio", "max"),
            precio_minimo   = ("precio", "min"),
        )
        .sort_values("total_ingresos", ascending=False)
        .round(2)
        .reset_index()
    )
    rentabilidad["participacion_%"] = (
        rentabilidad["total_ingresos"] / rentabilidad["total_ingresos"].sum() * 100
    ).round(2)

    # ── MÓDULO 2: PERFIL DE CLIENTE ────────────────────
    perfil = df.groupby("cliente_id").agg(
        total_gastado      = ("precio", "sum"),
        gasto_promedio     = ("precio", "mean"),
        gasto_maximo       = ("precio", "max"),
        num_compras        = ("venta_id", "count"),
        mes_ultima_compra  = ("fecha_venta", lambda x: x.max().month),
        dias_desde_ultima  = ("fecha_venta", lambda x: (
            pd.Timestamp.now() - x.max()
        ).days)
    ).reset_index()

    if "tipo_cliente" in clientes.columns:
        perfil = perfil.merge(
            clientes[["cliente_id", "tipo_cliente"]],
            on="cliente_id", how="left"
        )
    else:
        perfil["tipo_cliente"] = "general"

    # ── MÓDULO 3: CLIENTES RECURRENTES ────────────────
    perfil["prob_volver_a_comprar"] = (
        perfil["num_compras"] / perfil["num_compras"].max()
    ).round(2)

    recurrentes = perfil.sort_values(
        "prob_volver_a_comprar", ascending=False
    )

    # ── MÓDULO 4: CLIENTES EN RIESGO ──────────────────
    en_riesgo = perfil[
        (perfil["dias_desde_ultima"] > 90) &
        (perfil["num_compras"] >= 2)
    ].copy()
    en_riesgo["prob_no_volver"] = 1.0

    # ── MÓDULO 5: RECOMENDACIONES ──────────────────────
    if "tipo_cliente" in df.columns:
        recomendaciones = (
            df.groupby(["tipo_cliente", "producto"])
            .size()
            .reset_index(name="veces_comprado")
            .sort_values(["tipo_cliente", "veces_comprado"], ascending=[True, False])
            .groupby("tipo_cliente")
            .first()
            .reset_index()
        )
    else:
        recomendaciones = rentabilidad[["producto", "total_ventas"]].head(3)

    # ── MÓDULO 6: TENDENCIA MENSUAL ────────────────────
    df["mes"] = df["fecha_venta"].dt.to_period("M").astype(str)
    tendencia = df.groupby("mes")["precio"].sum().reset_index()

    # ── GUARDAR RESULTADOS ─────────────────────────────
    df.to_csv("data/processed/ventas_con_encoding.csv", index=False)
    rentabilidad.to_csv("data/outputs/productos_rentables.csv", index=False)
    recurrentes.to_csv("data/outputs/clientes_recurrentes.csv", index=False)
    en_riesgo.to_csv("data/outputs/clientes_en_riesgo.csv", index=False)
    recomendaciones.to_csv("data/outputs/recomendaciones_producto.csv", index=False)

    return {
        "ventas"          : df,
        "rentabilidad"    : rentabilidad,
        "recurrentes"     : recurrentes,
        "en_riesgo"       : en_riesgo,
        "recomendaciones" : recomendaciones,
        "tendencia"       : tendencia,
        "total_ingresos"  : df["precio"].sum(),
        "total_clientes"  : df["cliente_id"].nunique(),
        "total_ventas"    : len(df)
    }