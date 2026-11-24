"""
Generador del dataset SIMULADO de eventos sísmicos (evento-localidad).

IMPORTANTE: estos datos son SIMULADOS con fines educativos. Los epicentros,
magnitudes y profundidades son sintéticos, y la intensidad se obtiene con una
relación de atenuación sencilla más ruido. No representan sismos reales ni
sirven para conclusiones científicas. Para un trabajo con datos reales,
reemplazar datos/sismos_simulado.csv por un CSV con las mismas columnas
(fuentes: IGP, CENSIS, USGS) y volver a ejecutar entrenar_modelo.py.
"""

import numpy as np
import pandas as pd

from nucleo import DIR_DATOS

SEMILLA = 2026
N_EVENTOS = 230

# (localidad, departamento, latitud, longitud)
LOCALIDADES = [
    ("Huánuco", "Huánuco", -9.9306, -76.2422),
    ("Tingo María", "Huánuco", -9.2955, -75.9981),
    ("Cerro de Pasco", "Pasco", -10.6864, -76.2623),
    ("Lima", "Lima", -12.0464, -77.0428),
    ("Huancayo", "Junín", -12.0651, -75.2049),
    ("Pucallpa", "Ucayali", -8.3791, -74.5539),
    ("Cusco", "Cusco", -13.5319, -71.9675),
    ("Arequipa", "Arequipa", -16.4090, -71.5375),
    ("Ica", "Ica", -14.0678, -75.7286),
    ("Ayacucho", "Ayacucho", -13.1588, -74.2239),
    ("Piura", "Piura", -5.1945, -80.6328),
    ("Tarapoto", "San Martín", -6.4889, -76.3650),
    ("Moyobamba", "San Martín", -6.0342, -76.9714),
    ("Tacna", "Tacna", -18.0066, -70.2463),
    ("Trujillo", "La Libertad", -8.1116, -79.0288),
    ("Huaraz", "Áncash", -9.5278, -77.5278),
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia epicentral (km) entre dos puntos."""
    radio = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * radio * np.arcsin(np.sqrt(a))


def generar(n_eventos=N_EVENTOS):
    rng = np.random.default_rng(SEMILLA)
    loc = pd.DataFrame(LOCALIDADES, columns=["localidad", "departamento", "latitud", "longitud"])
    efecto_sitio = {n: rng.normal(0, 0.35) for n in loc["localidad"]}

    # Más eventos cerca de Huánuco (zona de interés del reto)
    pesos = np.where(loc["departamento"] == "Huánuco", 2.0, 1.0)
    pesos = pesos / pesos.sum()

    inicio = np.datetime64("2000-01-01")
    dias = int((np.datetime64("2025-12-31") - inicio).astype(int))

    filas = []
    for e in range(1, n_eventos + 1):
        ancla = loc.iloc[rng.choice(len(loc), p=pesos)]
        lat = ancla["latitud"] + rng.normal(0, 0.6)
        lon = ancla["longitud"] + rng.normal(0, 0.6)
        magnitud = round(float(min(3.8 + rng.exponential(1.0), 7.8)), 1)

        u = rng.random()
        if u < 0.60:
            prof = rng.uniform(8, 60)       # superficiales
        elif u < 0.88:
            prof = rng.uniform(60, 150)     # intermedios
        else:
            prof = rng.uniform(150, 600)    # profundos
        prof = round(float(prof), 1)

        fecha = str(inicio + np.timedelta64(int(rng.integers(0, dias)), "D"))
        d_todas = haversine_km(lat, lon, loc["latitud"].values, loc["longitud"].values)
        cercanas = np.argsort(d_todas)[:6]

        for j in rng.choice(cercanas, size=3, replace=False):
            fila_loc = loc.iloc[j]
            d = float(d_todas[j])
            r_hipo = np.hypot(d, prof)
            # Atenuación sencilla: crece con la magnitud, decrece con la distancia
            i = (0.8 + 1.5 * magnitud - 1.7 * np.log10(r_hipo + 10)
                 + efecto_sitio[fila_loc["localidad"]] + rng.normal(0, 0.5))
            filas.append({
                "evento_id": f"EV{e:04d}",
                "fecha": fecha,
                "latitud": round(float(lat), 4),
                "longitud": round(float(lon), 4),
                "localidad": fila_loc["localidad"],
                "departamento": fila_loc["departamento"],
                "magnitud": magnitud,
                "profundidad_km": prof,
                "distancia_km": round(d, 1),
                "intensidad": int(np.clip(round(i), 1, 12)),
                "origen": "simulado",
            })

    df = pd.DataFrame(filas)

    # --- Se introducen errores a propósito para practicar la limpieza ---
    rs = np.random.default_rng(SEMILLA + 1)
    df.loc[rs.choice(len(df), 5, replace=False), "profundidad_km"] = np.nan
    df.loc[rs.choice(len(df), 3, replace=False), "magnitud"] = 15.0      # imposible
    df.loc[rs.choice(len(df), 2, replace=False), "distancia_km"] = -10.0  # imposible
    for i in rs.choice(len(df), 4, replace=False):
        df.loc[i, "localidad"] = " " + df.loc[i, "localidad"] + " "       # espacios sobrantes
    duplicados = df.sample(8, random_state=SEMILLA)
    df = pd.concat([df, duplicados], ignore_index=True)
    return df


def generar_y_guardar():
    DIR_DATOS.mkdir(exist_ok=True)
    df = generar()
    ruta = DIR_DATOS / "sismos_simulado.csv"
    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"Dataset simulado guardado en {ruta} ({len(df)} filas, {df['evento_id'].nunique()} eventos)")
    return ruta


if __name__ == "__main__":
    generar_y_guardar()
