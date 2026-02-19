import streamlit as st
import pandas as pd
import plotly.express as px

def mostrar_ventas():
    st.title("📊 Análisis de Ventas")
    st.divider()

    try:
        ventas   = pd.read_csv("data/processed/ventas_con_encoding.csv", parse_dates=["fecha_venta"])
        clientes = pd.read_csv("data/raw/clientes.csv")
        rentabilidad = pd.read_csv("data/outputs/productos_rentables.csv")
    except FileNotFoundError:
        st.error("⚠️ Ejecuta primero los scripts de src/")
        return

    # ── MÉTRICAS ───────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Ingresos Totales", f"S/. {ventas['precio'].sum():,.0f}")
    with col2:
        st.metric("🛒 Total Ventas", f"{len(ventas):,}")
    with col3:
        st.metric("💻 Precio Promedio", f"S/. {ventas['precio'].mean():,.0f}")

    st.divider()

    # ── GRÁFICA 1: Productos más rentables ─────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Productos más rentables")
        fig = px.bar(
            rentabilidad,
            x="total_ingresos",
            y="producto",
            orientation="h",
            color="total_ingresos",
            color_continuous_scale="Blues",
            text_auto=".2s"
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Participación por producto")
        fig2 = px.pie(
            rentabilidad,
            values="total_ingresos",
            names="producto",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # ── GRÁFICA 2: Tendencia mensual ───────────────────
    st.subheader("📅 Tendencia de ventas mensual")
    ventas["mes"] = ventas["fecha_venta"].dt.to_period("M").astype(str)
    tendencia = ventas.groupby("mes")["precio"].sum().reset_index()

    fig3 = px.line(
        tendencia,
        x="mes", y="precio",
        markers=True,
        color_discrete_sequence=["#2ecc71"]
    )
    fig3.update_layout(height=350, xaxis_tickangle=45)
    st.plotly_chart(fig3, use_container_width=True)

    # ── TABLA ──────────────────────────────────────────
    st.subheader("📋 Detalle por producto")
    st.dataframe(rentabilidad, use_container_width=True)