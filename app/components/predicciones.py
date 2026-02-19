import streamlit as st
import pandas as pd
import plotly.express as px

def mostrar_predicciones():
    st.title("🤖 Predicciones con IA")
    st.divider()

    try:
        prediccion  = pd.read_csv("data/outputs/prediccion_proxima_semana.csv")
        recurrentes = pd.read_csv("data/outputs/clientes_recurrentes.csv")
    except FileNotFoundError:
        st.error("⚠️ Ejecuta primero los scripts de src/")
        return

    # ── PREDICCIÓN PRÓXIMA SEMANA ──────────────────────
    st.subheader("🔮 Predicción próxima semana")
    pred = prediccion.iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Ingresos estimados", f"S/. {pred['ingreso_estimado']:,.0f}")
    with col2:
        st.metric("💻 Producto más vendido", pred["producto_mas_vendido"])
    with col3:
        st.metric("👥 Cliente activo", pred["tipo_cliente_activo"])

    st.divider()

    # ── TOP CLIENTES RECURRENTES ───────────────────────
    st.subheader("🏆 Clientes con mayor probabilidad de volver")
    top = recurrentes.sort_values(
        "prob_volver_a_comprar", ascending=False
    ).head(10)

    fig = px.bar(
        top,
        x="cliente_id",
        y="prob_volver_a_comprar",
        color="prob_volver_a_comprar",
        color_continuous_scale="Greens",
        labels={"prob_volver_a_comprar": "Probabilidad", "cliente_id": "Cliente"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Lista completa")
    st.dataframe(
        recurrentes[["cliente_id", "ciudad", "tipo_cliente", "total_gastado", "prob_volver_a_comprar"]]
        .sort_values("prob_volver_a_comprar", ascending=False)
        .head(20),
        use_container_width=True
    )