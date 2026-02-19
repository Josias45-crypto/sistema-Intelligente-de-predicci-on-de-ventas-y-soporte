# src/modelo_sklearn.py
# PEA 2 — Scikit-learn
# Sistema Inteligente de Análisis Comercial
# Adaptable a cualquier negocio que maneje ventas y clientes

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os

os.makedirs("data/outputs", exist_ok=True)

print("=" * 60)
print("  SISTEMA INTELIGENTE DE ANÁLISIS COMERCIAL")
print("  Powered by Scikit-learn")
print("=" * 60)

# ── CARGAR DATOS ───────────────────────────────────
print("\n Cargando datos...")
ventas   = pd.read_csv("data/processed/ventas_con_encoding.csv", parse_dates=["fecha_venta"])
clientes = pd.read_csv("data/raw/clientes.csv", parse_dates=["fecha_registro"])

# ══════════════════════════════════════════════════
# MÓDULO 1 — PRODUCTOS MÁS RENTABLES
# ══════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  MÓDULO 1: Productos más rentables")
print("═" * 60)

rentabilidad = (
    ventas.groupby("producto")
    .agg(
        total_ingresos  = ("precio", "sum"),
        total_ventas    = ("venta_id", "count"),
        precio_promedio = ("precio", "mean"),
        precio_maximo   = ("precio", "max"),
        precio_minimo   = ("precio", "min"),
    )
    .sort_values("total_ingresos", ascending=False)
    .round(2)
)

rentabilidad["participacion_%"] = (
    rentabilidad["total_ingresos"] / rentabilidad["total_ingresos"].sum() * 100
).round(2)

print(rentabilidad.to_string())
rentabilidad.to_csv("data/outputs/productos_rentables.csv")
print("\n Productos rentables identificados")

# ══════════════════════════════════════════════════
# MÓDULO 2 — CONSTRUIR PERFIL DE CLIENTE
# ══════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  MÓDULO 2: Perfil de clientes")
print("═" * 60)

perfil = ventas.groupby("cliente_id").agg(
    total_gastado       = ("precio", "sum"),
    gasto_promedio      = ("precio", "mean"),
    gasto_maximo        = ("precio", "max"),
    num_compras         = ("venta_id", "count"),
    producto_favorito   = ("producto_encoded", "mean"),
    mes_ultima_compra   = ("fecha_venta", lambda x: x.max().month),
    dias_entre_compras  = ("fecha_venta", lambda x: (
        (x.max() - x.min()).days / max(len(x) - 1, 1)
    )),
    dias_desde_ultima   = ("fecha_venta", lambda x: (
        pd.Timestamp("2024-12-31") - x.max()
    ).days)
).reset_index()

# Agregar tipo de cliente
perfil = perfil.merge(
    clientes[["cliente_id", "tipo_cliente", "ciudad"]],
    on="cliente_id", how="left"
)

le = LabelEncoder()
perfil["tipo_cliente_cod"] = le.fit_transform(perfil["tipo_cliente"].fillna("particular"))

print(f" Perfil construido para {len(perfil):,} clientes")

# ══════════════════════════════════════════════════
# MÓDULO 3 — PREDECIR QUIÉN VOLVERÁ A COMPRAR
# ══════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  MÓDULO 3: Predicción de clientes recurrentes")
print("═" * 60)

perfil["volvio_a_comprar"] = (perfil["num_compras"] > 1).astype(int)

FEATURES = [
    "total_gastado", "gasto_promedio", "gasto_maximo",
    "num_compras", "producto_favorito", "mes_ultima_compra",
    "dias_entre_compras", "dias_desde_ultima", "tipo_cliente_cod"
]

X = perfil[FEATURES].fillna(0)
y = perfil["volvio_a_comprar"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

modelo_recurrente = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_recurrente.fit(X_train_sc, y_train)

y_pred = modelo_recurrente.predict(X_test_sc)
y_prob = modelo_recurrente.predict_proba(X_test_sc)[:, 1]

print(f"\n Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# Top clientes con mayor probabilidad de volver
perfil_test = perfil.iloc[X_test.index].copy()
perfil_test["prob_volver_a_comprar"] = y_prob

top_recurrentes = (
    perfil_test
    .sort_values("prob_volver_a_comprar", ascending=False)
    [["cliente_id", "ciudad", "tipo_cliente", "total_gastado", "prob_volver_a_comprar"]]
    .head(10)
    .round(2)
)
print("\n Top 10 clientes con mayor probabilidad de volver:")
print(top_recurrentes.to_string(index=False))

# ══════════════════════════════════════════════════
# MÓDULO 4 — DETECTAR CLIENTES EN RIESGO
# ══════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  MÓDULO 4: Clientes en riesgo de no volver")
print("═" * 60)

perfil["en_riesgo"] = (
    (perfil["dias_desde_ultima"] > 180) &
    (perfil["num_compras"] >= 2)
).astype(int)

modelo_riesgo = GradientBoostingClassifier(n_estimators=100, random_state=42)
X2 = perfil[FEATURES].fillna(0)
y2 = perfil["en_riesgo"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)
X2_train_sc = scaler.fit_transform(X2_train)
X2_test_sc  = scaler.transform(X2_test)

modelo_riesgo.fit(X2_train_sc, y2_train)
prob_riesgo = modelo_riesgo.predict_proba(X2_test_sc)[:, 1]

perfil_riesgo = perfil.iloc[X2_test.index].copy()
perfil_riesgo["prob_no_volver"] = prob_riesgo

clientes_riesgo = (
    perfil_riesgo[perfil_riesgo["prob_no_volver"] > 0.5]
    .sort_values("prob_no_volver", ascending=False)
    [["cliente_id", "ciudad", "tipo_cliente", "total_gastado",
      "dias_desde_ultima", "prob_no_volver"]]
    .head(10)
    .round(2)
)
print(f"\n  Clientes en riesgo detectados: {len(clientes_riesgo)}")
print(clientes_riesgo.to_string(index=False))

# ══════════════════════════════════════════════════
# MÓDULO 5 — RECOMENDAR PRODUCTO A CADA CLIENTE
# ══════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  MÓDULO 5: Recomendación de productos")
print("═" * 60)

# ← CORRECCIÓN: usar .map() en lugar de .merge() para evitar conflicto de columnas
ventas_tipo = ventas.copy()
ventas_tipo["tipo_cliente"] = ventas_tipo["cliente_id"].map(
    clientes.set_index("cliente_id")["tipo_cliente"]
)

recomendaciones = (
    ventas_tipo.groupby(["tipo_cliente", "producto"])
    .agg(veces_comprado=("venta_id", "count"))
    .reset_index()
    .sort_values(["tipo_cliente", "veces_comprado"], ascending=[True, False])
    .groupby("tipo_cliente")
    .first()
    .reset_index()
    [["tipo_cliente", "producto", "veces_comprado"]]
)

print("\n Producto recomendado por tipo de cliente:")
print(recomendaciones.to_string(index=False))

# ── GUARDAR TODOS LOS RESULTADOS ───────────────────
perfil_test.sort_values("prob_volver_a_comprar", ascending=False).to_csv(
    "data/outputs/clientes_recurrentes.csv", index=False
)
clientes_riesgo.to_csv(
    "data/outputs/clientes_en_riesgo.csv", index=False
)
recomendaciones.to_csv(
    "data/outputs/recomendaciones_producto.csv", index=False
)

print("\n" + "=" * 60)
print("   SISTEMA COMERCIAL COMPLETADO")
print("   Resultados en data/outputs/:")
print("     - productos_rentables.csv")
print("     - clientes_recurrentes.csv")
print("     - clientes_en_riesgo.csv")
print("     - recomendaciones_producto.csv")
print("=" * 60)
