"""
Entrenamiento y evaluación del clasificador KNN de intensidad sísmica.

Pasos (actividades 1 a 4 del reto):
  1. Cargar, documentar y limpiar el dataset con Pandas y NumPy.
  2. Separar 80 % entrenamiento / 20 % prueba AGRUPANDO por evento
     (un mismo evento no puede estar en ambos conjuntos: evita fuga de información).
  3. Pipeline(StandardScaler + KNeighborsClassifier). Comparar K = 3, 5, 7, 9
     con validación cruzada agrupada (GroupKFold) solo dentro del entrenamiento.
  4. Evaluar en prueba: exactitud, precisión, recall, F1 y matriz de confusión.

Uso:  python entrenar_modelo.py [ruta_csv]
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit

from nucleo import (CLASES, DIR_DATOS, DIR_MODELO, FEATURES, LIMITES, VALORES_K,
                    categoria_mercalli, crear_pipeline)

SEMILLA = 42


def limpiar(df):
    """Limpieza con Pandas/NumPy. Devuelve (df_limpio, informe)."""
    info = {"filas_iniciales": int(len(df))}
    df = df.copy()

    df["localidad"] = df["localidad"].astype(str).str.strip()
    df["departamento"] = df["departamento"].astype(str).str.strip()

    n = len(df)
    df = df.drop_duplicates()
    info["duplicados_eliminados"] = int(n - len(df))

    n = len(df)
    obligatorias = FEATURES + ["intensidad", "evento_id", "latitud", "longitud"]
    df = df.dropna(subset=obligatorias)
    info["filas_con_nulos_eliminadas"] = int(n - len(df))

    valido = pd.Series(True, index=df.index)
    for col, (lo, hi) in LIMITES.items():
        valido &= df[col].between(lo, hi)
    valido &= df["intensidad"].between(1, 12)
    info["filas_fuera_de_rango_eliminadas"] = int((~valido).sum())
    df = df[valido].copy()

    df["intensidad"] = df["intensidad"].astype(int)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["categoria"] = df["intensidad"].map(categoria_mercalli)
    df = df.reset_index(drop=True)

    info["filas_finales"] = int(len(df))
    info["eventos"] = int(df["evento_id"].nunique())
    return df, info


def main():
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else DIR_DATOS / "sismos_simulado.csv"
    if not ruta.exists():
        print(f"No existe {ruta}; se genera el dataset simulado...")
        from generar_dataset import generar_y_guardar
        ruta = generar_y_guardar()

    DIR_MODELO.mkdir(exist_ok=True)
    crudo = pd.read_csv(ruta)
    df, limpieza = limpiar(crudo)
    if len(df) < 500:
        print(f"ADVERTENCIA: solo hay {len(df)} registros (se piden al menos 500).")
    df.to_csv(DIR_DATOS / "sismos_limpio.csv", index=False, encoding="utf-8")

    print("=== Limpieza ===")
    for k, v in limpieza.items():
        print(f"  {k}: {v}")
    print("\nDistribución de categorías:")
    print(df["categoria"].value_counts().reindex(CLASES))

    # --- 2. Separación 80/20 agrupada por evento ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMILLA)
    i_train, i_test = next(gss.split(df, df["categoria"], groups=df["evento_id"]))
    train, test = df.iloc[i_train].copy(), df.iloc[i_test].copy()
    assert set(train["evento_id"]).isdisjoint(test["evento_id"]), "Fuga: eventos repetidos"
    train.to_csv(DIR_DATOS / "train.csv", index=False, encoding="utf-8")
    test.to_csv(DIR_DATOS / "test.csv", index=False, encoding="utf-8")
    print(f"\nEntrenamiento: {len(train)} registros / {train['evento_id'].nunique()} eventos")
    print(f"Prueba:        {len(test)} registros / {test['evento_id'].nunique()} eventos")

    X_train, y_train, g_train = train[FEATURES], train["categoria"], train["evento_id"]
    X_test, y_test = test[FEATURES], test["categoria"]

    # --- 3. Pipeline + comparación de K con validación cruzada agrupada ---
    busqueda = GridSearchCV(
        crear_pipeline(5),
        param_grid={"knn__n_neighbors": VALORES_K},
        cv=GroupKFold(n_splits=5),
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        refit="f1_macro",
    )
    busqueda.fit(X_train, y_train, groups=g_train)
    r = busqueda.cv_results_
    cv = pd.DataFrame({
        "K": [int(k) for k in r["param_knn__n_neighbors"]],
        "exactitud_media": r["mean_test_accuracy"],
        "exactitud_desv": r["std_test_accuracy"],
        "f1_macro_medio": r["mean_test_f1_macro"],
        "f1_macro_desv": r["std_test_f1_macro"],
    }).round(4)
    cv.to_csv(DIR_MODELO / "cv_resultados.csv", index=False)
    mejor_k = int(busqueda.best_params_["knn__n_neighbors"])
    print("\n=== Validación cruzada agrupada (solo entrenamiento) ===")
    print(cv.to_string(index=False))
    print(f"\nK seleccionado (mejor F1 macro): {mejor_k}")

    modelo = busqueda.best_estimator_
    joblib.dump(modelo, DIR_MODELO / "modelo_knn.joblib")

    # --- 4. Evaluación en prueba ---
    y_pred = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, y_pred)
    p, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=CLASES, average="macro", zero_division=0)
    reporte = classification_report(y_test, y_pred, labels=CLASES,
                                    output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred, labels=CLASES)
    baseline = float(y_test.value_counts(normalize=True).max())

    print("\n=== Evaluación en prueba ===")
    print(classification_report(y_test, y_pred, labels=CLASES, zero_division=0))
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(pd.DataFrame(matriz, index=CLASES, columns=CLASES))
    print(f"\nLínea base (siempre la clase mayoritaria): {baseline:.3f}")

    # Figuras
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=CLASES, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Matriz de confusión (prueba, K = {mejor_k})")
    plt.setp(ax.get_xticklabels(), rotation=15)
    plt.tight_layout()
    fig.savefig(DIR_MODELO / "matriz_confusion.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(cv["K"], cv["exactitud_media"], yerr=cv["exactitud_desv"],
                marker="o", capsize=4, label="Exactitud")
    ax.errorbar(cv["K"], cv["f1_macro_medio"], yerr=cv["f1_macro_desv"],
                marker="s", capsize=4, label="F1 macro")
    ax.axvline(mejor_k, color="red", linestyle="--", alpha=0.6, label=f"K elegido = {mejor_k}")
    ax.set_xticks(VALORES_K)
    ax.set_xlabel("K (número de vecinos)")
    ax.set_ylabel("Puntaje en validación cruzada agrupada")
    ax.set_title("Comparación de K")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(DIR_MODELO / "comparacion_k.png", dpi=130)
    plt.close(fig)

    metricas = {
        "k_seleccionado": mejor_k,
        "limpieza": limpieza,
        "distribucion_clases": {c: int((df["categoria"] == c).sum()) for c in CLASES},
        "n_train": int(len(train)), "n_test": int(len(test)),
        "eventos_train": int(train["evento_id"].nunique()),
        "eventos_test": int(test["evento_id"].nunique()),
        "cv": json.loads(cv.to_json(orient="records")),
        "test": {
            "exactitud": float(exactitud),
            "precision_macro": float(p), "recall_macro": float(rec), "f1_macro": float(f1),
            "linea_base_mayoritaria": baseline,
            "por_clase": {c: {m: float(reporte[c][m]) for m in
                              ("precision", "recall", "f1-score", "support")} for c in CLASES},
            "matriz_confusion": matriz.tolist(),
        },
        "fuente": "simulado" if (df["origen"] == "simulado").all() else "mixta/real",
    }
    (DIR_MODELO / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nModelo y métricas guardados en {DIR_MODELO}")


if __name__ == "__main__":
    main()
