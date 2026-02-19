# src/modelo_pytorch.py
# PEA 2 — PyTorch
# Red neuronal para predicción de ventas, productos y clientes

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

os.makedirs("data/outputs", exist_ok=True)

print("=" * 60)
print("  SISTEMA DE PREDICCIÓN — PyTorch")
print(f"  Dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

# ── CARGAR DATOS ───────────────────────────────────
print("\n Cargando datos...")
ventas   = pd.read_csv("data/processed/ventas_con_encoding.csv", parse_dates=["fecha_venta"])
clientes = pd.read_csv("data/raw/clientes.csv")

# ── PREPARAR SERIE TEMPORAL POR DÍA ───────────────
print("\n Construyendo serie temporal diaria...")

ventas["fecha"] = ventas["fecha_venta"].dt.date

# Agregar tipo de cliente
ventas["tipo_cliente"] = ventas["cliente_id"].map(
    clientes.set_index("cliente_id")["tipo_cliente"]
)

# Producto más vendido del día
producto_del_dia = (
    ventas.groupby(["fecha", "producto"])
    .size()
    .reset_index(name="cantidad")
    .sort_values(["fecha", "cantidad"], ascending=[True, False])
    .groupby("fecha")
    .first()
    .reset_index()[["fecha", "producto"]]
)

# Tipo de cliente más activo del día
tipo_del_dia = (
    ventas.groupby(["fecha", "tipo_cliente"])
    .size()
    .reset_index(name="cantidad")
    .sort_values(["fecha", "cantidad"], ascending=[True, False])
    .groupby("fecha")
    .first()
    .reset_index()[["fecha", "tipo_cliente"]]
)

# Serie principal
serie = (
    ventas.groupby("fecha")
    .agg(
        ingresos      = ("precio", "sum"),
        num_ventas    = ("venta_id", "count"),
        precio_prom   = ("precio", "mean")
    )
    .reset_index()
)

serie = serie.merge(producto_del_dia, on="fecha", how="left")
serie = serie.merge(tipo_del_dia, on="fecha", how="left")
serie = serie.sort_values("fecha").reset_index(drop=True)

# Codificar categóricas
le_producto = LabelEncoder()
le_tipo     = LabelEncoder()
serie["producto_cod"]  = le_producto.fit_transform(serie["producto"].fillna("PC Básica"))
serie["tipo_cod"]      = le_tipo.fit_transform(serie["tipo_cliente"].fillna("particular"))

print(f" Serie temporal: {len(serie)} días de datos")
print(serie.head())

# ── CREAR SECUENCIAS PARA LA RED NEURONAL ──────────
print("\n Creando secuencias de entrenamiento...")

VENTANA = 7  # Usar 7 días para predecir el día siguiente

FEATURES_NUM = ["ingresos", "num_ventas", "precio_prom", "producto_cod", "tipo_cod"]

scaler = StandardScaler()
datos_scaled = scaler.fit_transform(serie[FEATURES_NUM])

def crear_secuencias(datos, ventana):
    X, y_ing, y_prod, y_tipo = [], [], [], []
    for i in range(len(datos) - ventana):
        X.append(datos[i:i+ventana])
        y_ing.append(datos[i+ventana][0])   # ingresos
        y_prod.append(serie["producto_cod"].iloc[i+ventana])  # producto
        y_tipo.append(serie["tipo_cod"].iloc[i+ventana])      # tipo cliente
    return (
        np.array(X),
        np.array(y_ing),
        np.array(y_prod),
        np.array(y_tipo)
    )

X, y_ing, y_prod, y_tipo = crear_secuencias(datos_scaled, VENTANA)

# Convertir a tensores
X_tensor      = torch.FloatTensor(X)
y_ing_tensor  = torch.FloatTensor(y_ing).unsqueeze(1)
y_prod_tensor = torch.LongTensor(y_prod)
y_tipo_tensor = torch.LongTensor(y_tipo)

# Split
split = int(len(X) * 0.8)
X_train, X_test         = X_tensor[:split], X_tensor[split:]
yi_train, yi_test       = y_ing_tensor[:split], y_ing_tensor[split:]
yp_train, yp_test       = y_prod_tensor[:split], y_prod_tensor[split:]
yt_train, yt_test       = y_tipo_tensor[:split], y_tipo_tensor[split:]

print(f"   Entrenamiento: {len(X_train)} secuencias")
print(f"   Prueba       : {len(X_test)} secuencias")

# ── DEFINIR RED NEURONAL ───────────────────────────
print("\n Definiendo red neuronal...")

NUM_PRODUCTOS = serie["producto_cod"].nunique()
NUM_TIPOS     = serie["tipo_cod"].nunique()

class RedPrediccion(nn.Module):
    def __init__(self, input_size, hidden_size, num_productos, num_tipos):
        super(RedPrediccion, self).__init__()

        # Capa compartida LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Cabeza 1: predicción de ingresos
        self.cabeza_ingresos = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Cabeza 2: predicción de producto más vendido
        self.cabeza_producto = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_productos)
        )

        # Cabeza 3: predicción de tipo de cliente
        self.cabeza_tipo = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_tipos)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        ultimo = lstm_out[:, -1, :]  # último paso temporal
        return (
            self.cabeza_ingresos(ultimo),
            self.cabeza_producto(ultimo),
            self.cabeza_tipo(ultimo)
        )

modelo = RedPrediccion(
    input_size  = len(FEATURES_NUM),
    hidden_size = 128,
    num_productos = NUM_PRODUCTOS,
    num_tipos     = NUM_TIPOS
)

print(f" Red neuronal creada")
print(f"   Parámetros totales: {sum(p.numel() for p in modelo.parameters()):,}")

# ── ENTRENAR ───────────────────────────────────────
print("\n  Entrenando red neuronal...")

optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)
criterio_reg  = nn.MSELoss()
criterio_clas = nn.CrossEntropyLoss()

EPOCHS = 50
losses = []

for epoch in range(EPOCHS):
    modelo.train()
    optimizer.zero_grad()

    pred_ing, pred_prod, pred_tipo = modelo(X_train)

    loss_ing  = criterio_reg(pred_ing, yi_train)
    loss_prod = criterio_clas(pred_prod, yp_train)
    loss_tipo = criterio_clas(pred_tipo, yt_train)

    # Loss total combinada
    loss_total = loss_ing + loss_prod + loss_tipo
    loss_total.backward()
    optimizer.step()

    losses.append(loss_total.item())

    if (epoch + 1) % 10 == 0:
        print(f"   Época {epoch+1:>3}/{EPOCHS} | Loss: {loss_total.item():.4f}")

print(" Entrenamiento completado")

# ── EVALUAR Y PREDECIR ─────────────────────────────
print("\n Evaluando modelo...")
modelo.eval()
with torch.no_grad():
    pred_ing, pred_prod, pred_tipo = modelo(X_test)

    # Ingresos
    ing_reales    = yi_test.numpy().flatten()
    ing_predichos = pred_ing.numpy().flatten()

    # Producto
    prod_reales    = yp_test.numpy()
    prod_predichos = pred_prod.argmax(dim=1).numpy()
    acc_prod = (prod_predichos == prod_reales).mean() * 100

    # Tipo cliente
    tipo_reales    = yt_test.numpy()
    tipo_predichos = pred_tipo.argmax(dim=1).numpy()
    acc_tipo = (tipo_predichos == tipo_reales).mean() * 100

print(f"   Accuracy producto más vendido : {acc_prod:.2f}%")
print(f"   Accuracy tipo de cliente      : {acc_tipo:.2f}%")

# ── PREDICCIÓN PRÓXIMA SEMANA ──────────────────────
print("\n" + "═" * 60)
print("  PREDICCIÓN — PRÓXIMA SEMANA")
print("═" * 60)

modelo.eval()
with torch.no_grad():
    ultima_secuencia = X_tensor[-1].unsqueeze(0)
    pred_ing, pred_prod, pred_tipo = modelo(ultima_secuencia)

    # Reconstruir ingreso real (des-escalar)
    dummy = np.zeros((1, len(FEATURES_NUM)))
    dummy[0][0] = pred_ing.item()
    ingreso_predicho = scaler.inverse_transform(dummy)[0][0]

    producto_predicho = le_producto.inverse_transform(
        [pred_prod.argmax().item()]
    )[0]

    tipo_predicho = le_tipo.inverse_transform(
        [pred_tipo.argmax().item()]
    )[0]

print(f"\n   Ingresos estimados     : S/. {ingreso_predicho:,.2f}")
print(f"   Producto más vendido   : {producto_predicho}")
print(f"   Tipo de cliente activo : {tipo_predicho}")

# ── GUARDAR ────────────────────────────────────────
prediccion = pd.DataFrame([{
    "ingreso_estimado"    : round(ingreso_predicho, 2),
    "producto_mas_vendido": producto_predicho,
    "tipo_cliente_activo" : tipo_predicho,
    "accuracy_producto_%" : round(acc_prod, 2),
    "accuracy_tipo_%"     : round(acc_tipo, 2)
}])

prediccion.to_csv("data/outputs/prediccion_proxima_semana.csv", index=False)

print("\n" + "=" * 60)
print("   MODELO PYTORCH COMPLETADO")
print("   data/outputs/prediccion_proxima_semana.csv")
print("=" * 60)