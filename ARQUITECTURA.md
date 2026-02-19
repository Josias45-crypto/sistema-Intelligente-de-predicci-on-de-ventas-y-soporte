#  Arquitectura del Sistema

## Visión General

El sistema está compuesto por 6 módulos independientes que se
ejecutan en secuencia, donde cada uno toma los resultados del
anterior y los complementa

## Flujo de datos
```
[DATA RAW]
clientes.csv
ventas.csv                  ←  Entrada del negocio
tickets.csv
      │
      ▼
[MÓDULO 1 — Pandas]
analisis_pandas.py
  • Merge de tablas
  • Window functions 48h      →  ventas_procesadas.csv
  • Agrupaciones de negocio
      │
      ▼
[MÓDULO 2 — NumPy]
analisis_numpy.py
  • Estadísticas avanzadas
  • Detección de outliers     →  ventas_con_encoding.csv
  • Target Encoding
      │
      ├─────────────────────────────────┐
      ▼                                 ▼
[MÓDULO 3 — Scikit-learn]     [MÓDULO 4 — PyTorch]
modelo_sklearn.py             modelo_pytorch.py
  • Productos rentables         • Red neuronal LSTM
  • Clientes recurrentes        • Predicción ingresos
  • Clientes en riesgo          • Producto más vendido
  • Recomendaciones             • Tipo cliente activo
      │                                 │
      └─────────────┬───────────────────┘
                    ▼
         [MÓDULO 5 — Reporte]
         reporte_final.py
           • 5 gráficas visuales
           • Resumen ejecutivo
```

## Descripción de cada módulo

###  Datos de entrada
Tres tablas principales que representan las operaciones del negocio.
Son reemplazables por datos reales del cliente.

| Archivo | Descripción |
|---|---|
| `clientes.csv` | Registro de clientes del negocio |
| `ventas.csv` | Historial de ventas |
| `tickets.csv` | Tickets de soporte técnico |

###  Módulo 1 — Pandas
Responsable de cargar, limpiar y unir las tablas. Aplica window
functions para calcular el gasto promedio de cada cliente en las
últimas 48 horas, un indicador clave de comportamiento reciente.

###  Módulo 2 — NumPy
Realiza cálculos estadísticos avanzados sobre los datos procesados.
Detecta precios atípicos usando el método IQR y aplica Target
Encoding vectorizado para convertir variables categóricas en
numéricas sin usar bucles.

###  Módulo 3 — Scikit-learn
Contiene 5 sub-módulos de Machine Learning:
- **Rentabilidad**: ranking de productos por ingresos
- **Perfil de cliente**: construcción de variables por cliente
- **Recurrencia**: Random Forest para predecir si un cliente volverá
- **Riesgo**: Gradient Boosting para detectar clientes inactivos
- **Recomendación**: producto sugerido según tipo de cliente

###  Módulo 4 — PyTorch
Red neuronal LSTM con 3 cabezas de predicción simultánea:
- Ingresos esperados la próxima semana
- Producto que más se venderá
- Tipo de cliente más activo

###  Módulo 5 — Reporte Visual
Genera 5 gráficas en PNG listas para presentar al cliente o dueño
del negocio, incluyendo un resumen ejecutivo con las métricas
más importantes.

## Tecnologías
```
Python 3.12
├── pandas       — Manipulación de datos
├── numpy        — Cálculo numérico
├── scikit-learn — Machine Learning clásico
├── torch        — Deep Learning
├── matplotlib   — Visualización
└── faker        — Generación de datos de prueba
```

## Adaptabilidad

El sistema está diseñado para funcionar con cualquier negocio.
Para adaptarlo solo se necesita:
1. Reemplazar los CSVs de `data/raw/` con datos reales
2. Ajustar los nombres de columnas en `generar_datos.py`
3. Ejecutar el pipeline completo