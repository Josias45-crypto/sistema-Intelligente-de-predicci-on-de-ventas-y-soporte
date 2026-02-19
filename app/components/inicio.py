import streamlit as st
import pandas as pd
import plotly.express as px

def mostrar_inicio():
    st.title("🧠 Sistema Inteligente de Análisis Comercial")
    st.caption("Transforma tus datos de ventas en decisiones inteligentes")
    st.divider()

    # ── CARGAR DATOS ───────────────────────────────────
    try:
        ventas       = pd.read_csv("data/processed/ventas_con_encoding.csv", parse_dates=["fecha_venta"])
        clientes     = pd.read_csv("data/raw/clientes.csv")
        rentabilidad = pd.read_csv("data/outputs/productos_rentables.csv")
        recurrentes  = pd.read_csv("data/outputs/clientes_recurrentes.csv")
        en_riesgo    = pd.read_csv("data/outputs/clientes_en_riesgo.csv")
        prediccion   = pd.read_csv("data/outputs/prediccion_proxima_semana.csv")
    except FileNotFoundError:
        st.error("⚠️ Primero ejecuta todos los scripts de src/")
        st.code("python src/generar_datos.py\npython src/analisis_pandas.py\npython src/analisis_numpy.py\npython src/modelo_sklearn.py\npython src/modelo_pytorch.py")
        return

    # ── MÉTRICAS PRINCIPALES ───────────────────────────
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

    # ── GRÁFICA RÁPIDA DE VENTAS ───────────────────────
    st.subheader("📅 Tendencia de ventas")
    ventas["mes"] = ventas["fecha_venta"].dt.to_period("M").astype(str)
    tendencia = ventas.groupby("mes")["precio"].sum().reset_index()

    fig = px.area(
        tendencia,
        x="mes", y="precio",
        color_discrete_sequence=["#2ecc71"],
        labels={"mes": "Mes", "precio": "Ingresos (S/.)"}
    )
    fig.update_layout(height=300, xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── PRODUCTO ESTRELLA Y PREDICCIÓN ─────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Producto Estrella")
        producto_top = rentabilidad.iloc[0]
        st.success(f"**{producto_top['producto']}**")
        st.write(f"Ingresos totales: **S/. {producto_top['total_ingresos']:,.2f}**")
        st.write(f"Total vendido: **{int(producto_top['total_ventas'])} unidades**")
        st.write(f"Participación: **{producto_top['participacion_%']}%**")

    with col2:
        st.subheader("🔮 Próxima Semana")
        pred = prediccion.iloc[0]
        st.info(f"**Producto esperado:** {pred['producto_mas_vendido']}")
        st.info(f"**Cliente más activo:** {pred['tipo_cliente_activo']}")
        st.info(f"**Ingresos estimados:** S/. {pred['ingreso_estimado']:,.2f}")

    st.divider()

    # ── TABLA RESUMEN ──────────────────────────────────
    st.subheader("📋 Top productos")
    st.dataframe(
        rentabilidad[["producto", "total_ventas", "total_ingresos", "participacion_%"]],
        use_container_width=True
    )