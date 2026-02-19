import streamlit as st
import pandas as pd
import plotly.express as px

def mostrar_recomendaciones():
    st.title("💡 Recomendaciones")
    st.caption("Qué producto ofrecer según el tipo de cliente")
    st.divider()

    try:
        recomendaciones = pd.read_csv("data/outputs/recomendaciones_producto.csv")
        rentabilidad    = pd.read_csv("data/outputs/productos_rentables.csv")
    except FileNotFoundError:
        st.error("⚠️ Ejecuta primero los scripts de src/")
        return

    st.subheader("🎯 Producto recomendado por tipo de cliente")
    for _, row in recomendaciones.iterrows():
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**{row['tipo_cliente'].upper()}**")
            with col2:
                st.success(f"💻 {row['producto']} — {row['veces_comprado']} compras")

    st.divider()
    st.subheader("📊 Rentabilidad por producto")
    fig = px.bar(
        rentabilidad,
        x="producto",
        y="participacion_%",
        color="participacion_%",
        color_continuous_scale="Greens",
        text_auto=".1f"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)