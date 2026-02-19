# src/reporte_final.py
# Reporte visual final del sistema
# Genera gráficas para presentar a la empresa

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("data/outputs/graficas", exist_ok=True)

print("=" * 60)
print("  GENERANDO REPORTE VISUAL FINAL")
print("=" * 60)

# ── CARGAR RESULTADOS ──────────────────────────────
ventas        = pd.read_csv("data/processed/ventas_con_encoding.csv", parse_dates=["fecha_venta"])
clientes      = pd.read_csv("data/raw/clientes.csv")
rentabilidad  = pd.read_csv("data/outputs/productos_rentables.csv")
recurrentes   = pd.read_csv("data/outputs/clientes_recurrentes.csv")
en_riesgo     = pd.read_csv("data/outputs/clientes_en_riesgo.csv")
recomendacion = pd.read_csv("data/outputs/recomendaciones_producto.csv")
prediccion    = pd.read_csv("data/outputs/prediccion_proxima_semana.csv")

# ══════════════════════════════════════════════════
# GRÁFICA 1 — Productos más rentables
# ══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
colores = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
bars = ax.barh(
    rentabilidad["producto"],
    rentabilidad["total_ingresos"],
    color=colores
)
ax.set_title(" Productos más rentables", fontsize=14, fontweight="bold")
ax.set_xlabel("Ingresos Totales (S/.)")
for bar, val in zip(bars, rentabilidad["total_ingresos"]):
    ax.text(bar.get_width() + 10000, bar.get_y() + bar.get_height()/2,
            f"S/. {val:,.0f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("data/outputs/graficas/1_productos_rentables.png", dpi=150)
plt.close()
print(" Gráfica 1 generada")

# ══════════════════════════════════════════════════
# GRÁFICA 2 — Tendencia de ventas mensual
# ══════════════════════════════════════════════════
ventas["mes"] = ventas["fecha_venta"].dt.to_period("M").astype(str)
tendencia = ventas.groupby("mes")["precio"].sum().reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(tendencia["mes"], tendencia["precio"],
        marker="o", color="#3498db", linewidth=2)
ax.fill_between(range(len(tendencia)), tendencia["precio"],
                alpha=0.1, color="#3498db")
ax.set_xticks(range(len(tendencia)))
ax.set_xticklabels(tendencia["mes"], rotation=45, ha="right")
ax.set_title(" Tendencia de Ventas Mensual", fontsize=14, fontweight="bold")
ax.set_ylabel("Ingresos (S/.)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("data/outputs/graficas/2_tendencia_mensual.png", dpi=150)
plt.close()
print(" Gráfica 2 generada")

# ══════════════════════════════════════════════════
# GRÁFICA 3 — Ventas por tipo de cliente
# ══════════════════════════════════════════════════
ventas["tipo_cliente"] = ventas["cliente_id"].map(
    clientes.set_index("cliente_id")["tipo_cliente"]
)
por_tipo = ventas.groupby("tipo_cliente")["precio"].sum()

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(
    por_tipo.values,
    labels=por_tipo.index,
    autopct="%1.1f%%",
    colors=["#2ecc71", "#3498db", "#e74c3c"],
    startangle=90
)
ax.set_title("👥 Ingresos por Tipo de Cliente", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/outputs/graficas/3_ingresos_por_tipo.png", dpi=150)
plt.close()
print(" Gráfica 3 generada")

# ══════════════════════════════════════════════════
# GRÁFICA 4 — Top clientes potenciales
# ══════════════════════════════════════════════════
top10 = recurrentes.sort_values(
    "prob_volver_a_comprar", ascending=False
).head(10)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    range(len(top10)),
    top10["prob_volver_a_comprar"] * 100,
    color="#2ecc71"
)
ax.set_xticks(range(len(top10)))
ax.set_xticklabels(
    [f"Cliente {int(c)}" for c in top10["cliente_id"]],
    rotation=45, ha="right"
)
ax.set_title(" Top 10 Clientes con Mayor Probabilidad de Volver", fontsize=13, fontweight="bold")
ax.set_ylabel("Probabilidad (%)")
ax.set_ylim(0, 110)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.0f}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("data/outputs/graficas/4_clientes_potenciales.png", dpi=150)
plt.close()
print(" Gráfica 4 generada")

# ══════════════════════════════════════════════════
# GRÁFICA 5 — Resumen ejecutivo
# ══════════════════════════════════════════════════
fig = plt.figure(figsize=(12, 6))
fig.patch.set_facecolor("#1a1a2e")

ax = fig.add_subplot(111)
ax.set_facecolor("#1a1a2e")
ax.axis("off")

titulo = "SISTEMA INTELIGENTE DE ANÁLISIS COMERCIAL"
ax.text(0.5, 0.92, titulo, ha="center", va="center",
        fontsize=16, fontweight="bold", color="white",
        transform=ax.transAxes)

ax.text(0.5, 0.82, "— Resumen Ejecutivo —", ha="center",
        fontsize=11, color="#95a5a6", transform=ax.transAxes)

# Métricas clave
metricas = [
    (" Ingresos Totales",
     f"S/. {ventas['precio'].sum():,.0f}"),
    (" Producto Estrella",
     rentabilidad.iloc[0]["producto"]),
    (" Total Clientes",
     f"{ventas['cliente_id'].nunique():,}"),
    ("  Clientes en Riesgo",
     f"{len(en_riesgo)}"),
    (" Predicción Próx. Semana",
     f"S/. {prediccion.iloc[0]['ingreso_estimado']:,.0f}"),
    (" Próximo Producto Top",
     prediccion.iloc[0]["producto_mas_vendido"]),
]

x_pos = [0.1, 0.4, 0.7, 0.1, 0.4, 0.7]
y_pos = [0.58, 0.58, 0.58, 0.28, 0.28, 0.28]

for (label, valor), x, y in zip(metricas, x_pos, y_pos):
    ax.text(x, y + 0.08, label, ha="left", fontsize=9,
            color="#95a5a6", transform=ax.transAxes)
    ax.text(x, y, valor, ha="left", fontsize=13,
            fontweight="bold", color="#2ecc71", transform=ax.transAxes)

ax.text(0.5, 0.05,
        "Desarrollado con Python | Pandas · NumPy · Scikit-learn · PyTorch",
        ha="center", fontsize=8, color="#7f8c8d", transform=ax.transAxes)

plt.tight_layout()
plt.savefig("data/outputs/graficas/5_resumen_ejecutivo.png",
            dpi=150, facecolor="#1a1a2e")
plt.close()
print(" Gráfica 5 generada — Resumen ejecutivo")

print("\n" + "=" * 60)
print("   REPORTE FINAL COMPLETADO")
print("  Gráficas en: data/outputs/graficas/")
print("     1_productos_rentables.png")
print("     2_tendencia_mensual.png")
print("     3_ingresos_por_tipo.png")
print("     4_clientes_potenciales.png")
print("     5_resumen_ejecutivo.png")
print("=" * 60)