# =============================================================
# ARCHIVO: app/components/cargar_datos.py
# DESCRIPCIÓN: Permite al cliente subir sus propios datos
# =============================================================

import streamlit as st
import pandas as pd
import os

def mostrar_carga():
    st.title("📂 Cargar Datos del Negocio")
    st.caption("Sube tus archivos de ventas y clientes para analizar tu negocio")
    st.divider()

    # ── INSTRUCCIONES ──────────────────────────────────
    with st.expander("📋 ¿Qué formato deben tener los archivos?"):
        st.markdown("### Archivo de Clientes")
        st.dataframe(pd.DataFrame({
            "cliente_id" : [1, 2, 3],
            "nombre"     : ["Juan Pérez", "María García", "Luis Torres"],
            "ciudad"     : ["Lima", "Arequipa", "Cusco"],
            "tipo_cliente": ["particular", "empresa", "estudiante"],
            "fecha_registro": ["2023-01-01", "2023-02-15", "2023-03-10"]
        }))

        st.markdown("### Archivo de Ventas")
        st.dataframe(pd.DataFrame({
            "venta_id"   : [1, 2, 3],
            "cliente_id" : [1, 2, 1],
            "producto"   : ["Laptop", "PC Gamer", "Servidor"],
            "marca"      : ["HP", "Asus", "Dell"],
            "precio"     : [2500.00, 4800.00, 12000.00],
            "fecha_venta": ["2023-01-15", "2023-01-20", "2023-02-01"]
        }))

    st.divider()

    # ── CARGA DE ARCHIVOS ──────────────────────────────
    st.subheader("📤 Sube tus archivos")

    col1, col2 = st.columns(2)

    with col1:
        archivo_clientes = st.file_uploader(
            "👤 Archivo de Clientes (CSV o Excel)",
            type=["csv", "xlsx"],
            key="clientes"
        )

    with col2:
        archivo_ventas = st.file_uploader(
            "💻 Archivo de Ventas (CSV o Excel)",
            type=["csv", "xlsx"],
            key="ventas"
        )

    # ── PROCESAR ARCHIVOS ──────────────────────────────
    if archivo_clientes and archivo_ventas:
        try:
            # Leer clientes
            if archivo_clientes.name.endswith(".csv"):
                clientes = pd.read_csv(archivo_clientes)
            else:
                clientes = pd.read_excel(archivo_clientes)

            # Leer ventas
            if archivo_ventas.name.endswith(".csv"):
                ventas = pd.read_csv(archivo_ventas)
            else:
                ventas = pd.read_excel(archivo_ventas)

            # ── VALIDAR COLUMNAS ───────────────────────
            cols_clientes = ["cliente_id", "nombre", "ciudad", "tipo_cliente"]
            cols_ventas   = ["venta_id", "cliente_id", "producto", "precio", "fecha_venta"]

            errores = []
            for col in cols_clientes:
                if col not in clientes.columns:
                    errores.append(f"Falta columna '{col}' en clientes")
            for col in cols_ventas:
                if col not in ventas.columns:
                    errores.append(f"Falta columna '{col}' en ventas")

            if errores:
                for error in errores:
                    st.error(f"❌ {error}")
                return

            # ── GUARDAR DATOS ──────────────────────────
            os.makedirs("data/raw", exist_ok=True)
            clientes.to_csv("data/raw/clientes.csv", index=False)
            ventas.to_csv("data/raw/ventas.csv",     index=False)

            st.success("✅ Archivos cargados correctamente")
            st.divider()

            # ── PREVIEW DE DATOS ───────────────────────
            st.subheader("👁️ Vista previa de tus datos")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👤 Clientes", f"{len(clientes):,}")
            with col2:
                st.metric("💻 Ventas", f"{len(ventas):,}")
            with col3:
                st.metric("💰 Ingresos Totales",
                          f"S/. {ventas['precio'].sum():,.0f}")

            tab1, tab2 = st.tabs(["👤 Clientes", "💻 Ventas"])
            with tab1:
                st.dataframe(clientes.head(10), use_container_width=True)
            with tab2:
                st.dataframe(ventas.head(10), use_container_width=True)

            # ── BOTÓN PARA ANALIZAR ────────────────────
            st.divider()
            st.info("✅ Datos listos. Ve a **Análisis de Ventas** para ver los resultados.")

        except Exception as e:
            st.error(f"❌ Error al procesar los archivos: {e}")

    else:
        st.warning("👆 Sube ambos archivos para continuar")

        # ── OPCIÓN DE USAR DATOS DE PRUEBA ────────────
        st.divider()
        st.subheader("🧪 ¿No tienes datos aún?")
        if st.button("Usar datos de prueba"):
            import subprocess
            with st.spinner("Generando datos de prueba..."):
                subprocess.run(["python", "src/generar_datos.py"])
                subprocess.run(["python", "src/analisis_pandas.py"])
                subprocess.run(["python", "src/analisis_numpy.py"])
                subprocess.run(["python", "src/modelo_sklearn.py"])
                subprocess.run(["python", "src/modelo_pytorch.py"])
            st.success("✅ Datos de prueba generados. ¡Ya puedes explorar el sistema!")
            st.rerun()