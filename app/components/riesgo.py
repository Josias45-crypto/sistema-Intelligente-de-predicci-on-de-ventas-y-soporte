import streamlit as st
import pandas as pd
import plotly.express as px

def mostrar_riesgo():
    st.title("⚠️ Clientes en Riesgo")
    st.caption("Clientes que llevan mucho tiempo sin comprar")
    st.divider()

    try:
        en_riesgo = pd.read_csv("data/outputs/clientes_en_riesgo.csv")
    except FileNotFoundError:
        st.error("⚠️ Ejecuta primero los scripts de src/")
        return

    st.metric("⚠️ Total clientes en riesgo", len(en_riesgo))
    st.divider()

    fig = px.scatter(
        en_riesgo,
        x="dias_desde_ultima",
        y="total_gastado",
        color="tipo_cliente",
        size="total_gastado",
        hover_data=["cliente_id", "ciudad"],
        labels={
            "dias_desde_ultima": "Días sin comprar",
            "total_gastado": "Total gastado (S/.)"
        }
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Lista de clientes en riesgo")
    st.dataframe(en_riesgo, use_container_width=True)