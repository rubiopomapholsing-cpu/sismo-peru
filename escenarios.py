"""
Demostración de tres escenarios hipotéticos (actividad 7).
Imprime la categoría estimada, los votos de los vecinos y guarda los resultados en
modelo/escenarios.csv.   Uso:  python escenarios.py
"""

import json

import pandas as pd

from nucleo import DIR_DATOS, DIR_MODELO, ESCENARIOS, clasificar

metricas = json.loads((DIR_MODELO / "metricas.json").read_text(encoding="utf-8"))
k = metricas["k_seleccionado"]
train = pd.read_csv(DIR_DATOS / "train.csv")

filas = []
for nombre, entrada in ESCENARIOS.items():
    r = clasificar(entrada, train, k)
    print("=" * 70)
    print(nombre, entrada)
    print(f"Categoría estimada: {r['prediccion']}")
    print("Votos:", r["votos"])
    print("Probabilidades:", {c: round(p, 2) for c, p in r["probabilidades"].items()})
    print(r["vecinos"][["evento_id", "localidad", "magnitud", "profundidad_km",
                        "distancia_km", "intensidad", "categoria"]].to_string(index=False))
    filas.append({"escenario": nombre, **entrada, "k": k,
                  "categoria_estimada": r["prediccion"],
                  **{f"votos_{c}": n for c, n in r["votos"].items()},
                  **{f"prob_{c}": round(p, 3) for c, p in r["probabilidades"].items()}})

pd.DataFrame(filas).to_csv(DIR_MODELO / "escenarios.csv", index=False, encoding="utf-8")
print("\nGuardado en modelo/escenarios.csv")
