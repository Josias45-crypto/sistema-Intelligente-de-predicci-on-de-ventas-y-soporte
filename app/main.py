# =============================================================
# ARCHIVO: app/main.py
# DESCRIPCIÓN: Punto de entrada principal del dashboard
# =============================================================

import streamlit as st

# ── CONFIGURACIÓN DE LA PÁGINA ─────────────────────────────
st.set_page_config(
    page_title="Sistema Inteligente de Ventas",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ESTILOS PERSONALIZADOS ──────────────────────────────────
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stMetric { background-color: white; padding: 10px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Sistema Inteligente")
    st.caption("Análisis Comercial con IA")
    st.divider()

    pagina = st.selectbox("📂 Navegación", [
        "🏠 Inicio",
        "📊 Análisis de Ventas",
        "🤖 Predicciones",
        "⚠️  Clientes en Riesgo",
        "💡 Recomendaciones"
    ])

    st.divider()
    st.caption("Desarrollado con Python + IA")

# ── PÁGINAS ─────────────────────────────────────────────────
if pagina == "🏠 Inicio":
    from components.inicio import mostrar_inicio
    mostrar_inicio()

elif pagina == "📊 Análisis de Ventas":
    from app.components.ventas import mostrar_ventas
    mostrar_ventas()

elif pagina == "🤖 Predicciones":
    from app.components.predicciones import mostrar_predicciones
    mostrar_predicciones()

elif pagina == "⚠️  Clientes en Riesgo":
    from app.components.riesgo import mostrar_riesgo
    mostrar_riesgo()

elif pagina == "💡 Recomendaciones":
    from app.components.recomendaciones import mostrar_recomendaciones
    mostrar_recomendaciones()