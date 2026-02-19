# PEA 1 — Operaciones con NumPy
# Objetivo: Estadísticas avanzadas y detección de anomalías en ventas

import numpy as np
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# ── CARGAR DATOS ───────────────────────────────────
print(" Cargando datos procesados...")
df = pd.read_csv("data/processed/ventas_procesadas.csv")
precios = df["precio"].values

# ── 1. ESTADÍSTICAS AVANZADAS ──────────────────────
print("\n Estadísticas de ventas:")
print(f"   Promedio    : S/. {np.mean(precios):,.2f}")
print(f"   Mediana     : S/. {np.median(precios):,.2f}")
print(f"   Desv. Est.  : S/. {np.std(precios):,.2f}")
print(f"   Mínimo      : S/. {np.min(precios):,.2f}")
print(f"   Máximo      : S/. {np.max(precios):,.2f}")
print(f"   Total       : S/. {np.sum(precios):,.2f}")

# ── 2. DETECCIÓN DE PRECIOS ATÍPICOS ──────────────
print("\n Detectando precios atípicos...")
Q1 = np.percentile(precios, 25)
Q3 = np.percentile(precios, 75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[(df["precio"] < limite_inferior) | (df["precio"] > limite_superior)]

print(f"   Rango normal : S/. {limite_inferior:,.2f} — S/. {limite_superior:,.2f}")
print(f"   Ventas atípicas encontradas: {len(outliers)}")
print(outliers[["venta_id", "producto", "marca", "precio"]].head())

# ── 3. INGRESO PROMEDIO POR PRODUCTO ──────────────
print("\n Ingreso promedio por producto (vectorizado):")
productos = df["producto"].values
precios_arr = df["precio"].values

productos_unicos = np.unique(productos)
for prod in productos_unicos:
    mask = productos == prod
    promedio = np.mean(precios_arr[mask])
    total = np.sum(precios_arr[mask])
    print(f"   {prod:<20} Promedio: S/. {promedio:,.2f}  |  Total: S/. {total:,.2f}")

# ── 4. TARGET ENCODING VECTORIZADO ────────────────
print("\n Target Encoding de productos (vectorizado)...")

# Target: precio normalizado como indicador de valor
target = (precios_arr - np.min(precios_arr)) / (np.max(precios_arr) - np.min(precios_arr))

cats_unicas, indices = np.unique(productos, return_inverse=True)
suma_por_cat = np.bincount(indices, weights=target)
count_por_cat = np.bincount(indices)

SMOOTH = 10
media_global = np.mean(target)
encoding = (suma_por_cat + SMOOTH * media_global) / (count_por_cat + SMOOTH)

df["producto_encoded"] = encoding[indices]

print("   Encoding por producto:")
for i, prod in enumerate(cats_unicas):
    print(f"   {prod:<20} encoding: {encoding[i]:.4f}")

# ── GUARDAR ────────────────────────────────────────
df.to_csv("data/processed/ventas_con_encoding.csv", index=False)
outliers.to_csv("data/processed/ventas_atipicas.csv", index=False)

print("\n Análisis NumPy completado")
print(" Resultados guardados en data/processed/")