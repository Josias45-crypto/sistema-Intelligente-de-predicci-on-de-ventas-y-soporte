# =============================================================
# ARCHIVO: app/components/inicio.py
# DESCRIPCIÓN: Página de inicio del dashboard
# =============================================================

import streamlit as st
import pandas as pd

def mostrar_inicio():
    st.title("🧠 Sistema Inteligente de Análisis Comercial")
    st.caption("Transforma tus datos de ventas en decisiones inteligentes")
    st.divider()

    # ── CARGAR DATOS ───────────────────────────────────────
    try:
        ventas       = pd.read_csv("data/processed/ventas_con_encoding.csv", parse_dates=["fecha_venta"])
        recurrentes  = pd.read_csv("data/outputs/clientes_recurrentes.csv")
        en_riesgo    = pd.read_csv("data/outputs/clientes_en_riesgo.csv")
        rentabilidad = pd.read_csv("data/outputs/productos_rentables.csv")
        prediccion   = pd.read_csv("data/outputs/prediccion_proxima_semana.csv")
    except FileNotFoundError:
        st.error("⚠️ Primero ejecuta todos los scripts de src/ para generar los datos.")
        st.code("python src/generar_datos.py\npython src/analisis_pandas.py\npython src/analisis_numpy.py\npython src/modelo_sklearn.py\npython src/modelo_pytorch.py")
        return

    # ── MÉTRICAS PRINCIPALES ───────────────────────────────
    st.subheader("📈 Resumen del Negocio")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Ingresos Totales",
            f"S/. {ventas['precio'].sum():,.0f}"
        )
    with col2:
        st.metric(
            "👥 Total Clientes",
            f"{ventas['cliente_id'].nunique():,}"
        )
    with col3:
        st.metric(
            "⚠️ Clientes en Riesgo",
            f"{len(en_riesgo)}"
        )
    with col4:
        st.metric(
            "🔮 Predicción Próx. Semana",
            f"S/. {prediccion.iloc[0]['ingreso_estimado']:,.0f}"
        )

    st.divider()

    # ── PRODUCTO ESTRELLA ──────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Producto Estrella")
        producto_top = rentabilidad.iloc[0]
        st.success(f"**{producto_top['producto']}**")
        st.write(f"Ingresos totales: **S/. {producto_top['total_ingresos']:,.2f}**")
        st.write(f"Total vendido: **{producto_top['total_ventas']} unidades**")

    with col2:
        st.subheader("🔮 Próxima Semana")
        pred = prediccion.iloc[0]
        st.info(f"**Producto esperado:** {pred['producto_mas_vendido']}")
        st.info(f"**Cliente activo:** {pred['tipo_cliente_activo']}")
        st.info(f"**Ingresos estimados:** S/. {pred['ingreso_estimado']:,.2f}")