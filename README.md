# 🧠 Sistema Inteligente de Análisis Comercial

> Sistema de análisis y predicción de ventas basado en Machine Learning.
> Adaptable a cualquier negocio que maneje clientes y ventas.

---

##  ¿Qué hace este sistema?

Transforma los datos de ventas de un negocio en conocimiento útil para tomar
mejores decisiones comerciales. A través del análisis de datos y modelos de
inteligencia artificial, el sistema responde preguntas clave como:

- ¿Qué productos generan más ingresos?
- ¿Qué clientes van a volver a comprar?
- ¿Qué clientes están en riesgo de no volver?
- ¿Qué producto recomendar según el tipo de cliente?
- ¿Cuánto se espera facturar la próxima semana?

---

##  Tecnologías utilizadas

| Librería | Uso |
|---|---|
| **Pandas** | Carga, limpieza y análisis de datos |
| **NumPy** | Cálculos estadísticos vectorizados |
| **Scikit-learn** | Modelos de clasificación y predicción |
| **PyTorch** | Red neuronal LSTM para predicción temporal |
| **Matplotlib** | Generación de gráficas y reporte visual |

---

##  Estructura del proyecto
```
sistema-inteligente-ventas/
│
├── data/
│   ├── raw/               # Datos originales sin modificar
│   │   ├── clientes.csv
│   │   ├── ventas.csv
│   │   └── tickets.csv
│   ├── processed/         # Datos procesados y enriquecidos
│   └── outputs/           # Resultados, predicciones y gráficas
│       └── graficas/
│
├── src/
│   ├── generar_datos.py        # Generación de datos simulados
│   ├── analisis_pandas.py      # Análisis y merge de tablas
│   ├── analisis_numpy.py       # Estadísticas y encoding
│   ├── modelo_sklearn.py       # Modelos de ML (Random Forest)
│   ├── modelo_pytorch.py       # Red neuronal LSTM
│   └── reporte_final.py        # Gráficas y reporte visual
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/Josias45-crypto/sistema-Intelligente-de-predicci-on-de-ventas-y-soporte.git
cd sistema-Intelligente-de-predicci-on-de-ventas-y-soporte
```

### 2. Crear entorno virtual
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

##  Uso del sistema

Ejecuta los scripts en este orden:
```bash
# 1. Generar datos de prueba
python src/generar_datos.py

# 2. Análisis con Pandas
python src/analisis_pandas.py

# 3. Estadísticas con NumPy
python src/analisis_numpy.py

# 4. Modelos de predicción con Scikit-learn
python src/modelo_sklearn.py

# 5. Red neuronal con PyTorch
python src/modelo_pytorch.py

# 6. Generar reporte visual
python src/reporte_final.py
```

---

##  Resultados generados

Al ejecutar el sistema completo obtienes:

| Archivo | Contenido |
|---|---|
| `productos_rentables.csv` | Ranking de productos por ingresos |
| `clientes_recurrentes.csv` | Clientes con mayor probabilidad de volver |
| `clientes_en_riesgo.csv` | Clientes que podrían no volver |
| `recomendaciones_producto.csv` | Producto recomendado por tipo de cliente |
| `prediccion_proxima_semana.csv` | Predicción de ingresos y ventas |
| `graficas/` | 5 gráficas visuales del análisis |

---

##  Arquitectura del sistema
```
[Datos Brutos]
      ↓
[Pandas — Limpieza y Merge]
      ↓
[NumPy — Estadísticas y Encoding]
      ↓
      ├──→ [Scikit-learn — Clasificación y Recomendación]
      └──→ [PyTorch LSTM — Predicción Temporal]
                        ↓
              [Reporte Visual — Matplotlib]
```

---

##  Autor

Desarrollado por **Josias** durante prácticas profesionales.
Formación: SENATI — Ingeniería de Software con IA

---

##  Licencia

Este proyecto fue desarrollado con fines educativos y comerciales.