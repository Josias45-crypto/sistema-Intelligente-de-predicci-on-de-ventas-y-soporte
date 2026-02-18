
# PEA 1 — Operaciones con Pandas
import pandas as pd
import numpy as np

# ── CARGAR DATOS ───────────────────────────────────
print(" Cargando datos...")
clientes = pd.read_csv("data/raw/clientes.csv", parse_dates=["fecha_registro"])
ventas   = pd.read_csv("data/raw/ventas.csv",   parse_dates=["fecha_venta"])
tickets  = pd.read_csv("data/raw/tickets.csv",  parse_dates=["fecha_ticket"])

print(f"    Clientes : {len(clientes):,}")
print(f"    Ventas   : {len(ventas):,}")
print(f"    Tickets  : {len(tickets):,}")

# ── MERGE: Unir ventas con clientes ───────────────
print("\n Uniendo tablas...")
df_ventas = ventas.merge(clientes, on="cliente_id", how="left")
df_tickets = tickets.merge(clientes, on="cliente_id", how="left")

print(f"   Tabla ventas+clientes  : {df_ventas.shape}")
print(f"   Tabla tickets+clientes : {df_tickets.shape}")

# ── PREGUNTA 1: ¿Qué producto vende más? ──────────
print("\n TOP productos más vendidos:")
top_productos = (
    df_ventas.groupby("producto")
    .agg(
        total_ventas=("venta_id", "count"),
        ingreso_total=("precio", "sum"),
        precio_promedio=("precio", "mean")
    )
    .sort_values("ingreso_total", ascending=False)
    .round(2)
)
print(top_productos)

# ── PREGUNTA 2: ¿Qué tipo de cliente gasta más? ───
print("\n👥 Gasto por tipo de cliente:")
gasto_cliente = (
    df_ventas.groupby("tipo_cliente")
    .agg(
        total_ventas=("venta_id", "count"),
        ingreso_total=("precio", "sum"),
        gasto_promedio=("precio", "mean")
    )
    .sort_values("ingreso_total", ascending=False)
    .round(2)
)
print(gasto_cliente)

# ── PREGUNTA 3: Tendencia de ventas por mes ────────
print("\n Tendencia de ventas mensual:")
df_ventas["mes"] = df_ventas["fecha_venta"].dt.to_period("M")
tendencia_mensual = (
    df_ventas.groupby("mes")
    .agg(
        ventas=("venta_id", "count"),
        ingresos=("precio", "sum")
    )
    .round(2)
)
print(tendencia_mensual.head(10))

# ── PREGUNTA 4: Window Function — Gasto móvil 48h ─
print("\n Calculando gasto promedio móvil por cliente (48h)...")
df_ventas = df_ventas.sort_values(["cliente_id", "fecha_venta"])
df_ventas = df_ventas.set_index("fecha_venta")

gasto_48h = (
    df_ventas.groupby("cliente_id")["precio"]
    .rolling("48h", min_periods=1)
    .mean()
    .reset_index()
    .rename(columns={"precio": "gasto_promedio_48h"})
)

df_ventas = df_ventas.reset_index()
df_ventas["gasto_promedio_48h"] = gasto_48h["gasto_promedio_48h"].values
print(" Window function calculada")
print(df_ventas[["cliente_id", "fecha_venta", "precio", "gasto_promedio_48h"]].head(10))

# ── PREGUNTA 5: ¿Qué producto genera más tickets? ─
print("\n Tickets por tipo de problema:")
tickets_resumen = (
    df_tickets.groupby("tipo_problema")
    .agg(
        total_tickets=("ticket_id", "count"),
        horas_promedio=("horas_resolucion", "mean"),
        tasa_resolucion=("resuelto", "mean")
    )
    .sort_values("total_tickets", ascending=False)
    .round(2)
)
print(tickets_resumen)

# ── GUARDAR RESULTADOS ─────────────────────────────
import os
os.makedirs("data/processed", exist_ok=True)

df_ventas.to_csv("data/processed/ventas_procesadas.csv", index=False)
top_productos.to_csv("data/processed/top_productos.csv")
gasto_cliente.to_csv("data/processed/gasto_por_cliente.csv")

print("\n Análisis Pandas completado")
print(" Resultados guardados en data/processed/")