"""
Núcleo común del proyecto SISMO AI.
Constantes, categorías de intensidad, pipeline KNN y función de clasificación
con votos de vecinos. Lo usan entrenar_modelo.py, escenarios.py y app.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent
DIR_DATOS = RAIZ / "datos"
DIR_MODELO = RAIZ / "modelo"

FEATURES = ["magnitud", "profundidad_km", "distancia_km"]
CLASES = ["Leve (I-IV)", "Moderada (V-VI)", "Fuerte (VII-XII)"]
VALORES_K = [3, 5, 7, 9]

# Rangos físicamente razonables (se usan para limpiar datos y validar entradas)
LIMITES = {
    "magnitud": (1.0, 9.5),
    "profundidad_km": (0.0, 700.0),
    "distancia_km": (0.0, 2000.0),
}

# Tres escenarios hipotéticos para la demostración (actividad 7)
ESCENARIOS = {
    "Escenario 1 - Sismo moderado cercano": {
        "magnitud": 5.2, "profundidad_km": 35.0, "distancia_km": 25.0},
    "Escenario 2 - Sismo pequeño y lejano": {
        "magnitud": 3.8, "profundidad_km": 15.0, "distancia_km": 90.0},
    "Escenario 3 - Sismo grande, superficial y cercano": {
        "magnitud": 6.9, "profundidad_km": 20.0, "distancia_km": 15.0},
}


def categoria_mercalli(intensidad):
    """Agrupa la escala Mercalli modificada: I-IV, V-VI, VII-XII."""
    if intensidad <= 4:
        return CLASES[0]
    if intensidad <= 6:
        return CLASES[1]
    return CLASES[2]


def crear_pipeline(k):
    """Pipeline = StandardScaler + KNeighborsClassifier."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


def validar_entrada(entrada):
    """Devuelve una lista de errores (vacía si la entrada es válida)."""
    errores = []
    for campo, (minimo, maximo) in LIMITES.items():
        valor = entrada.get(campo)
        if valor is None or not np.isfinite(valor):
            errores.append(f"{campo}: falta el valor.")
        elif not (minimo <= valor <= maximo):
            errores.append(f"{campo}: debe estar entre {minimo} y {maximo}.")
    return errores


def advertencias_rango(entrada, train_df):
    """Avisa si la entrada cae fuera del rango visto en el entrenamiento."""
    avisos = []
    for campo in FEATURES:
        lo, hi = train_df[campo].min(), train_df[campo].max()
        if not (lo <= entrada[campo] <= hi):
            avisos.append(
                f"{campo} = {entrada[campo]} está fuera del rango de entrenamiento "
                f"({lo:.1f} - {hi:.1f}); el resultado es una extrapolación poco fiable.")
    return avisos


def clasificar(entrada, train_df, k):
    """Clasifica una entrada y devuelve predicción, probabilidades y votos.

    Ajusta el mismo pipeline (StandardScaler + KNN) sobre el conjunto de
    entrenamiento, por lo que se puede cambiar K sin reentrenar todo el proyecto.
    """
    pipe = crear_pipeline(k).fit(train_df[FEATURES], train_df["categoria"])
    X = pd.DataFrame([entrada])[FEATURES]

    prediccion = pipe.predict(X)[0]
    proba = dict(zip(pipe.classes_, pipe.predict_proba(X)[0]))
    probabilidades = {c: float(proba.get(c, 0.0)) for c in CLASES}

    # Vecinos en el espacio ya escalado
    escalador = pipe.named_steps["scaler"]
    knn = pipe.named_steps["knn"]
    dist, idx = knn.kneighbors(escalador.transform(X), n_neighbors=k)
    vecinos = train_df.iloc[idx[0]].copy()
    vecinos.insert(0, "distancia_escalada", np.round(dist[0], 3))
    conteo = vecinos["categoria"].value_counts()
    votos = {c: int(conteo.get(c, 0)) for c in CLASES}

    return {
        "prediccion": prediccion,
        "probabilidades": probabilidades,
        "votos": votos,
        "vecinos": vecinos,
    }
