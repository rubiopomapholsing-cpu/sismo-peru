# SISMO AI · Informe del proyecto

**Clasificador KNN de categoría de intensidad sísmica (escala Mercalli modificada)**

> **Aviso.** Este proyecto no predice la fecha, ubicación ni magnitud de futuros terremotos. No constituye una alerta ni una herramienta oficial de seguridad. Los datos usados son **simulados** y las conclusiones se limitan a la práctica educativa.

---

## 1. El reto

Se construyó una aplicación web que, dadas las características de un evento sísmico hipotético o ya registrado (magnitud, profundidad focal y distancia epicentral a una localidad), estima una **categoría de intensidad percibida** mediante el algoritmo KNN:

| Categoría | Escala Mercalli |
|---|---|
| Leve | I–IV |
| Moderada | V–VI |
| Fuerte | VII–XII |

## 2. Datos y fuentes

- **Origen: simulado.** No se contó con un catálogo público que traiga la intensidad observada por localidad (un catálogo de magnitudes, por sí solo, no contiene esa etiqueta). Por eso se generó un dataset sintético con `generar_dataset.py` y se identificó explícitamente con la columna `origen = "simulado"`.
- **Cómo se simuló:** 230 eventos con epicentros cerca de 16 localidades del Perú (con énfasis en Huánuco), magnitudes con distribución exponencial (muchos sismos pequeños, pocos grandes), profundidades superficiales/intermedias/profundas, y 3 localidades afectadas por evento. La distancia epicentral se calcula con la fórmula de Haversine. La intensidad se obtiene con una relación de atenuación sencilla (aumenta con la magnitud y disminuye con la distancia hipocentral) más un efecto de sitio por localidad y ruido aleatorio.
- **Campos:** `evento_id, fecha, latitud, longitud, localidad, departamento, magnitud, profundidad_km, distancia_km, intensidad, origen`.
- **Fuentes para una versión con datos reales:** IGP, CENSIS y USGS. Basta con colocar un CSV con las mismas columnas y ejecutar `python entrenar_modelo.py ruta.csv`.

### Limpieza (Pandas y NumPy)

Al dataset se le introdujeron errores a propósito para practicar la limpieza:

| Problema | Filas |
|---|---|
| Duplicados exactos eliminados | 8 |
| Filas con valores nulos eliminadas | 5 |
| Valores fuera de rango (magnitud 15, distancia negativa) eliminados | 5 |
| Espacios sobrantes en nombres de localidad | corregidos con `strip()` |
| **Registros finales** | **680** (230 eventos) |

Distribución de categorías: Leve 478 (70 %), Moderada 136 (20 %), Fuerte 66 (10 %). Las clases están **desbalanceadas**.

## 3. Metodología

1. **Separación 80/20 agrupada por evento** (`GroupShuffleSplit`). Los 3 registros de un mismo evento comparten magnitud y profundidad; si quedaran repartidos entre entrenamiento y prueba, el modelo "vería" el evento al evaluarse (fuga de información). Se comprobó con una aserción que ningún evento aparece en ambos conjuntos. Resultado: 544 registros / 184 eventos para entrenar y 136 registros / 46 eventos para probar.
2. **Pipeline:** `StandardScaler` + `KNeighborsClassifier`. El escalado es indispensable porque KNN usa distancias y las variables tienen unidades muy distintas (magnitud ~3–8, distancia hasta cientos de km). El escalador se ajusta solo con los datos de entrenamiento.
3. **Selección de K** con validación cruzada agrupada (`GroupKFold`, 5 pliegues) únicamente dentro del entrenamiento, comparando K = 3, 5, 7 y 9 y usando F1 macro como criterio (por el desbalance de clases).

| K | Exactitud media | F1 macro medio |
|---|---|---|
| 3 | 0.844 | 0.763 |
| 5 | 0.855 | 0.776 |
| 7 | 0.858 | 0.775 |
| **9** | **0.870** | **0.781** |

Se seleccionó **K = 9**. Las diferencias entre valores de K son pequeñas frente a la desviación entre pliegues (≈ 0.03–0.07), así que la elección no es contundente.

## 4. Evaluación en el conjunto de prueba

| Métrica | Valor |
|---|---|
| Exactitud | **0.868** |
| Precisión (macro) | 0.802 |
| Recall (macro) | 0.849 |
| F1 (macro) | 0.822 |
| Línea base (siempre "Leve") | 0.713 |

| Clase | Precisión | Recall | F1 | Casos |
|---|---|---|---|---|
| Leve (I–IV) | 0.927 | 0.918 | 0.922 | 97 |
| Moderada (V–VI) | 0.680 | 0.630 | 0.654 | 27 |
| Fuerte (VII–XII) | 0.800 | 1.000 | 0.889 | 12 |

**Matriz de confusión** (filas = real, columnas = predicho):

| | Leve | Moderada | Fuerte |
|---|---|---|---|
| **Leve** | 89 | 8 | 0 |
| **Moderada** | 7 | 17 | 3 |
| **Fuerte** | 0 | 0 | 12 |

**Interpretación.**
- El modelo supera claramente la línea base (0.868 frente a 0.713), por lo que aprende una relación real entre las variables y la categoría.
- Todos los errores ocurren entre categorías **vecinas** (Leve↔Moderada, Moderada↔Fuerte); nunca confunde Leve con Fuerte. Es lo esperable, porque la intensidad es continua y las fronteras entre categorías son artificiales.
- La clase **Moderada** es la más difícil (F1 = 0.65): queda en medio y es donde caen los casos de frontera.
- El recall de 1.00 en **Fuerte** se basa en solo 12 casos, por lo que es una estimación poco estable.
- La exactitud global está inflada por la clase mayoritaria; por eso se reportan F1 macro y la matriz de confusión.

## 5. Aplicación web (Streamlit)

Ejecutar con `streamlit run app.py`. Tiene cuatro pestañas:

- **Dashboard:** indicadores, histogramas de magnitud y profundidad, distribución regional, categorías, dispersión magnitud–distancia e intensidad media según magnitud.
- **Mapa de epicentros (Folium):** círculos por evento, con color según la intensidad máxima observada y tamaño según la magnitud; filtros por categoría y magnitud; marcadores de las localidades.
- **Evaluación del modelo:** comparación de K, métricas, matriz de confusión.
- **Simulador:** formulario con validación de entradas (rangos físicos, valores vacíos y advertencia si la entrada está fuera del rango de entrenamiento), selección de K, categoría estimada, **votación de los vecinos**, probabilidades y la tabla de vecinos utilizados.

## 6. Tres escenarios hipotéticos (K = 9)

| Escenario | Magnitud | Prof. (km) | Dist. (km) | Categoría estimada | Votos (Leve / Moderada / Fuerte) |
|---|---|---|---|---|---|
| 1. Sismo moderado cercano | 5.2 | 35 | 25 | **Moderada (V–VI)** | 0 / 9 / 0 |
| 2. Sismo pequeño y lejano | 3.8 | 15 | 90 | **Leve (I–IV)** | 9 / 0 / 0 |
| 3. Sismo grande, superficial y cercano | 6.9 | 20 | 15 | **Fuerte (VII–XII)** | 0 / 1 / 8 |

**Interpretación.**
- **Escenario 1:** los 9 vecinos son eventos moderados parecidos; la estimación es unánime. Coincide con lo esperado: magnitud media y cercanía dan una sacudida claramente percibida.
- **Escenario 2:** una magnitud baja y la distancia hacen que la sacudida sea apenas perceptible; el voto es unánime por "Leve".
- **Escenario 3:** combina magnitud alta, poca profundidad y poca distancia, que es la situación en que la intensidad es mayor. Hay un voto discrepante ("Moderada"), lo que indica que el evento está cerca de la frontera entre categorías (probabilidad estimada de Fuerte = 0.89).

Una votación unánime indica un caso "típico" del entrenamiento, no que el resultado sea exacto: una votación dividida señala una zona de frontera donde conviene confiar menos en la estimación.

## 7. Limitaciones

1. **Datos simulados.** La relación entre variables y la intensidad la define la fórmula de simulación que yo mismo escribí; el modelo solo la redescubre. No demuestra nada sobre el comportamiento sísmico real, y las métricas obtenidas no se pueden extrapolar a datos reales.
2. **Pocas variables.** Solo se usan magnitud, profundidad y distancia. La intensidad real depende también del tipo de suelo, la geología local, la calidad de las construcciones, la directividad de la ruptura y la densidad de población reportante.
3. **Desbalance de clases.** Predominan los sismos leves (70 %); hay solo 66 registros "Fuertes" y 12 en el conjunto de prueba, por lo que las métricas de esa clase son inestables.
4. **Conjunto de prueba pequeño** (136 registros, 46 eventos): una sola partición 80/20 da una estimación ruidosa del desempeño.
5. **Categorías artificiales.** Agrupar una escala continua en tres categorías genera errores en las fronteras, sobre todo en la clase Moderada.
6. **KNN no extrapola.** Para entradas fuera del rango de entrenamiento (por ejemplo, magnitud 8.5), el modelo solo compara con los vecinos más cercanos disponibles; la app lo advierte, pero la estimación no es confiable.
7. **Sin dimensión temporal ni espacial.** No usa fecha ni coordenadas como variables: describe la relación evento–intensidad, no la ocurrencia de sismos.
8. **Uso.** No es una herramienta de alerta temprana ni de gestión del riesgo.

## 8. Cómo reproducir

```
pip install -r requirements.txt
python generar_dataset.py      # opcional: crea datos/sismos_simulado.csv
python entrenar_modelo.py      # limpia, entrena, evalúa y guarda el modelo
python escenarios.py           # opcional: los tres escenarios en consola
streamlit run app.py           # abre la aplicación web
```

## 9. Estructura del proyecto

```
sismo_ai/
├── app.py                 Interfaz Streamlit
├── nucleo.py              Constantes, pipeline y función de clasificación con votos
├── generar_dataset.py     Dataset simulado
├── entrenar_modelo.py     Limpieza, split agrupado, GridSearch, evaluación
├── escenarios.py          Tres escenarios hipotéticos
├── requirements.txt
├── INFORME.md
├── datos/                 sismos_simulado.csv, sismos_limpio.csv, train.csv, test.csv
└── modelo/                modelo_knn.joblib, metricas.json, cv_resultados.csv,
                           matriz_confusion.png, comparacion_k.png, escenarios.csv
```
