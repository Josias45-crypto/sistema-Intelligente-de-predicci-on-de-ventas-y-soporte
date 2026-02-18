import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker("es_ES")
np.random.seed(42)
random.seed(42)

os.makedirs("data/raw", exist_ok=True)

# ── TABLA 1: CLIENTES ──────────────────────────────
NUM_CLIENTES = 500

clientes = pd.DataFrame({
    "cliente_id": range(1, NUM_CLIENTES + 1),
    "nombre": [fake.name() for _ in range(NUM_CLIENTES)],
    "ciudad": [fake.city() for _ in range(NUM_CLIENTES)],
    "tipo_cliente": np.random.choice(
        ["particular", "empresa", "estudiante"],
        size=NUM_CLIENTES,
        p=[0.5, 0.3, 0.2]
    ),
    "fecha_registro": pd.date_range(
        start="2023-01-01", periods=NUM_CLIENTES, freq="12h"
    )
})

# ── TABLA 2: VENTAS ────────────────────────────────
NUM_VENTAS = 2000

ventas = pd.DataFrame({
    "venta_id": range(1, NUM_VENTAS + 1),
    "cliente_id": np.random.randint(1, NUM_CLIENTES + 1, size=NUM_VENTAS),
    "producto": np.random.choice(
        ["PC Gamer", "Laptop Oficina", "Servidor", "PC Básica", "Laptop Gamer"],
        size=NUM_VENTAS,
        p=[0.25, 0.30, 0.10, 0.20, 0.15]
    ),
    "marca": np.random.choice(
        ["HP", "Dell", "Lenovo", "Asus", "Acer"],
        size=NUM_VENTAS
    ),
    "precio": np.round(
        np.random.uniform(400, 8000, size=NUM_VENTAS), 2
    ),
    "fecha_venta": pd.date_range(
        start="2023-01-01", periods=NUM_VENTAS, freq="6h"
    )
})

# ── TABLA 3: TICKETS DE SOPORTE ────────────────────
NUM_TICKETS = 1500

tickets = pd.DataFrame({
    "ticket_id": range(1, NUM_TICKETS + 1),
    "cliente_id": np.random.randint(1, NUM_CLIENTES + 1, size=NUM_TICKETS),
    "tipo_problema": np.random.choice(
        ["Hardware", "Software", "Red", "Sistema Operativo", "Otro"],
        size=NUM_TICKETS,
        p=[0.30, 0.35, 0.15, 0.15, 0.05]
    ),
    "tecnico": np.random.choice(
        ["Carlos", "María", "Luis", "Ana", "Jorge"],
        size=NUM_TICKETS
    ),
    "horas_resolucion": np.round(
        np.random.exponential(scale=5, size=NUM_TICKETS).clip(0.5, 48), 1
    ),
    "resuelto": np.random.choice(
        [1, 0], size=NUM_TICKETS, p=[0.85, 0.15]
    ),
    "fecha_ticket": pd.date_range(
        start="2023-01-01", periods=NUM_TICKETS, freq="8h"
    )
})

# ── GUARDAR ────────────────────────────────────────
clientes.to_csv("data/raw/clientes.csv", index=False)
ventas.to_csv("data/raw/ventas.csv", index=False)
tickets.to_csv("data/raw/tickets.csv", index=False)

print("Datos generados:")
print(f"    Clientes  : {len(clientes):,}")
print(f"    Ventas    : {len(ventas):,}")
print(f"    Tickets   : {len(tickets):,}")
print("\n Guardados en data/raw/")