"""
SISMO AI - Centro de análisis sísmico (interfaz web con Streamlit)

Ejecutar:  streamlit run app.py
Requiere haber ejecutado antes:  python entrenar_modelo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import folium
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from generar_dataset import LOCALIDADES
from nucleo import (CLASES, DIR_DATOS, DIR_MODELO, ESCENARIOS, LIMITES, VALORES_K,
                    advertencias_rango, categoria_mercalli, clasificar, validar_entrada)

COLORES = {CLASES[0]: "#2e9e5b", CLASES[1]: "#f0a30a", CLASES[2]: "#d62f2f"}

st.set_page_config(page_title="SISMO PERÚ", page_icon="🌎", layout="wide")

st.markdown("""
<style>
    :root {
        --panel: #183640;
        --panel-2: #203f49;
        --line: rgba(255,255,255,.10);
        --text: #edf5f3;
        --muted: #9fb1b4;
        --aqua: #79b7b3;
    }
    .stApp {
        background: #e9eef0;
        color: #23383e;
    }
    [data-testid="stAppViewContainer"] > .main {
        background: #eef3f4;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background: #17323d;
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * {
        color: #dce7e8;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 7px 10px;
        margin: 3px 0;
        border-radius: 7px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, #2f6470, #3d7079);
        color: white;
        box-shadow: inset 3px 0 #86bdbe;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,.07);
    }
    section.main .block-container {
        max-width: 1600px;
        padding: 0.5rem 1.2rem 1rem 1.2rem;
    }
    h1, h2, h3, p, div, span, label {
        font-family: "Segoe UI", Arial, sans-serif;
    }
    .app-header {
        height: 82px;
        margin: -8px -20px 12px -20px;
        padding: 12px 22px;
        background:
            linear-gradient(90deg, rgba(31,62,70,.98), rgba(74,106,112,.92)),
            url("https://images.unsplash.com/photo-1500530855697-b586d89ba3ee");
        background-size: cover;
        background-position: center 58%;
        color: white;
        border-radius: 0 0 8px 8px;
        box-shadow: 0 4px 16px rgba(36,62,68,.18);
    }
    .app-header h1 {
        font-size: 25px;
        margin: 4px 0 0 0;
        letter-spacing: .4px;
    }
    .app-header p {
        margin: 2px 0 0 0;
        color: #dbe6e6;
        font-size: 12px;
    }
    .metric-card {
        min-height: 82px;
        padding: 13px 16px;
        border-radius: 8px;
        color: white;
        background: linear-gradient(135deg, #355c64, #58777c);
        box-shadow: 0 5px 14px rgba(31,57,63,.12);
    }
    .metric-card .label {
        font-size: 11px;
        color: #c8d9d9;
    }
    .metric-card .value {
        font-size: 23px;
        font-weight: 700;
        margin-top: 4px;
    }
    .panel {
        background: #fff;
        border-radius: 8px;
        padding: 10px 12px;
        box-shadow: 0 4px 14px rgba(42,65,70,.09);
        border: 1px solid rgba(55,83,89,.08);
    }
    .panel-dark {
        background: linear-gradient(145deg, #1d3b45, #31545e);
        color: white;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 5px 16px rgba(31,57,63,.14);
    }
    .panel-title {
        font-size: 12px;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    .prediction-box {
        background: #eff6f5;
        border-radius: 7px;
        color: #263f45;
        text-align: center;
        padding: 12px 8px;
        margin: 6px 0;
    }
    .prediction-box strong {
        display: block;
        font-size: 21px;
        color: #f0f4f4;
    }
    .prediction-fuerte { background: #9d5c61; }
    .prediction-moderada { background: #b2a86b; }
    .prediction-leve { background: #4f9581; }
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: #f4f7f7 !important;
        border: 1px solid #d6e0e1 !important;
        color: #263e43 !important;
        border-radius: 5px !important;
    }
    .stButton button, .stFormSubmitButton button {
        width: 100%;
        border: 0;
        border-radius: 6px;
        background: #7faeb1;
        color: white;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #dce6e7;
        border-radius: 8px;
    }
    [data-testid="stMetricValue"] { color: #23383e; }
    [data-testid="stMetricLabel"] { color: #66777a; }
    [data-testid="stVerticalBlock"] {
        gap: .55rem;
    }
    .block-container {
        background: transparent;
    }
    .streamlit-expanderHeader,
    [data-testid="stForm"] {
        border: 0;
    }
    [data-testid="stForm"] {
        background: transparent;
    }
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #dde6e7;
    }
    .leaflet-tooltip.sismo-tooltip {
        background: #17343d;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 8px;
        color: #fff;
        box-shadow: 0 8px 22px rgba(20,44,51,.28);
        padding: 8px 10px;
        font-family: "Segoe UI", Arial, sans-serif;
    }
    .leaflet-tooltip.sismo-tooltip::before {
        border-top-color: #17343d;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def cargar():
    datos = pd.read_csv(DIR_DATOS / "sismos_limpio.csv", parse_dates=["fecha"])
    train = pd.read_csv(DIR_DATOS / "train.csv")
    metricas = json.loads((DIR_MODELO / "metricas.json").read_text(encoding="utf-8"))
    return datos, train, metricas


requeridos = [DIR_DATOS / "sismos_limpio.csv", DIR_DATOS / "train.csv",
              DIR_MODELO / "metricas.json"]
if not all(a.exists() for a in requeridos):
    st.error("Faltan los archivos del modelo. Ejecuta primero en la terminal: "
             "`python entrenar_modelo.py`")
    st.stop()

datos, train, metricas = cargar()
mejor_k = metricas["k_seleccionado"]

eventos = (datos.groupby("evento_id")
           .agg(fecha=("fecha", "first"), latitud=("latitud", "first"),
                longitud=("longitud", "first"), magnitud=("magnitud", "first"),
                profundidad_km=("profundidad_km", "first"),
                intensidad_max=("intensidad", "max"))
           .reset_index())
eventos["categoria_max"] = eventos["intensidad_max"].map(categoria_mercalli)

# ---------------------------------------------------------------- Encabezado
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:8px 0 12px'>"
                "<div style='font-size:28px'>🌎</div>"
                "<div style='font-size:18px;font-weight:700'>SISMOS</div>"
                "<div style='font-size:10px;color:#9fb7ba'>MONITOREO NACIONAL</div>"
                "</div>", unsafe_allow_html=True)
    pagina = st.radio(
        "Navegación",
        ["📊 Dashboard", "🗺️ Mapa de eventos", "🧮 Análisis KNN", "📈 Simulador de sismo",
         "🗃️ Dataset", "✅ Evaluación del modelo", "📄 Reportes", "❔ Ayuda"],
        index=0,
        label_visibility="collapsed",
    )

st.markdown("""
<div class="app-header">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
            <h1>SISMO PERÚ</h1>
            <p>Sistema Inteligente de Análisis y Estimación Sísmica con KNN</p>
        </div>
        <div style="text-align:right;font-size:11px;color:#dbe6e6">
            <b>Monitoreo sísmico</b><br>
            <span style="color:#8fabae">Datos simulados educativos</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if pagina == "🗺️ Mapa de eventos":
    st.markdown("### Mapa de eventos sísmicos")
    mapa = folium.Map(
        location=[-9.93, -76.24], zoom_start=6,
        tiles="OpenStreetMap")
    mapa.get_root().header.add_child(folium.Element("""
    <style>
        .leaflet-tooltip.sismo-tooltip {
            background:#17343d !important;
            border:1px solid rgba(255,255,255,.16) !important;
            border-radius:8px !important;
            color:#fff !important;
            box-shadow:0 8px 22px rgba(20,44,51,.28) !important;
            padding:8px 10px !important;
            font-family:"Segoe UI",Arial,sans-serif !important;
        }
        .leaflet-tooltip.sismo-tooltip::before {
            border-top-color:#17343d !important;
        }
    </style>
    """))
    for _, r in eventos.iterrows():
        folium.CircleMarker(
            location=[r["latitud"], r["longitud"]],
            radius=3 + max(0, r["magnitud"] - 3) * 1.2,
            color=COLORES[r["categoria_max"]], fill=True, fill_opacity=.65,
            tooltip=folium.Tooltip(
                f"<div style='font-size:11px;line-height:1.55'>"
                f"<b style='font-size:13px'>{r['evento_id']}</b><br>"
                f"MAGNITUD <b>{r['magnitud']:.1f} Mw</b><br>"
                f"Profundidad <b>{r['profundidad_km']} km</b><br>"
                f"Categoría <b>{r['categoria_max'].split(' ')[0]}</b></div>",
                sticky=True,
                direction="top",
                className="sismo-tooltip",
            ),
            popup=folium.Popup(
                f"<b>{r['evento_id']}</b><br>Magnitud: {r['magnitud']}<br>"
                f"Profundidad: {r['profundidad_km']} km",
                max_width=220),
        ).add_to(mapa)
    st.iframe(mapa.get_root().render(), height=650)

elif pagina in ("🧮 Análisis KNN", "✅ Evaluación del modelo"):
    st.markdown("### Evaluación del modelo")
    c1, c2, c3, c4 = st.columns(4)
    t = metricas["test"]
    c1.metric("Exactitud", f"{t['exactitud']:.3f}")
    c2.metric("Precisión macro", f"{t['precision_macro']:.3f}")
    c3.metric("Recall macro", f"{t['recall_macro']:.3f}")
    c4.metric("F1 macro", f"{t['f1_macro']:.3f}")

    st.markdown("#### Comparación de valores de K")
    cv = pd.DataFrame(metricas["cv"])
    st.dataframe(cv, use_container_width=True, hide_index=True)
    st.caption("El modelo seleccionado es aquel con el mejor F1 macro durante "
               "la validación cruzada agrupada por evento.")

    g1, g2 = st.columns(2)
    g1.image(str(DIR_MODELO / "matriz_confusion.png"), caption="Matriz de confusión")
    g2.image(str(DIR_MODELO / "comparacion_k.png"), caption="Comparación de K")

elif pagina == "📄 Reportes":
    st.markdown("### Reportes")

    st.markdown("#### Resumen ejecutivo")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Eventos", datos["evento_id"].nunique())
    r2.metric("Registros", len(datos))
    r3.metric("Localidades", datos["localidad"].nunique())
    r4.metric("K seleccionado", mejor_k)

    st.markdown("#### Distribución por categoría")
    reporte_categoria = (
        datos["categoria"]
        .value_counts()
        .rename_axis("Categoría")
        .reset_index(name="Registros")
    )
    reporte_categoria["Porcentaje"] = (
        reporte_categoria["Registros"] / len(datos) * 100
    ).round(2)
    st.dataframe(reporte_categoria, use_container_width=True, hide_index=True)

    st.markdown("#### Exportar datos")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Descargar dataset limpio",
        data=datos.to_csv(index=False).encode("utf-8"),
        file_name="sismos_limpio.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Descargar métricas del modelo",
        data=json.dumps(metricas, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="metricas_modelo.json",
        mime="application/json",
    )

elif pagina == "🗃️ Dataset":
    st.markdown("### Dataset histórico")
    st.dataframe(datos, use_container_width=True, hide_index=True)

elif pagina == "❔ Ayuda":
    st.markdown("### Ayuda")
    st.info("Este sistema clasifica intensidad sísmica con KNN. "
            "No predice futuros terremotos ni reemplaza alertas oficiales.")

else:
    # ---------------------------------------------------------------- KPIs
    k1, k2, k3, k4 = st.columns(4)
    valores = [
        ("Total de eventos", f"{eventos['evento_id'].nunique():,}", "Registros históricos"),
        ("Magnitud promedio", f"{eventos['magnitud'].mean():.1f} Mw", "Promedio nacional"),
        ("Profundidad promedio", f"{eventos['profundidad_km'].mean():.0f} km", "Profundidad focal"),
        ("Regiones analizadas", f"{datos['departamento'].nunique():,}", "Departamentos"),
    ]
    for col, (label, value, caption) in zip((k1, k2, k3, k4), valores):
        col.markdown(
            f"<div class='metric-card'><div class='label'>{label}</div>"
            f"<div class='value'>{value}</div>"
            f"<div style='font-size:10px;color:#b4c5c6'>{caption}</div></div>",
            unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    mapa_col, simulador_col, resultado_col = st.columns([1.55, 1.12, .72])

    with mapa_col:
        st.markdown("<div class='panel'><div class='panel-title'>Mapa de eventos sísmicos del Perú</div>",
                    unsafe_allow_html=True)
        mapa = folium.Map(
            location=[-9.93, -76.24], zoom_start=5,
            tiles="OpenStreetMap")
        mapa.get_root().header.add_child(folium.Element("""
        <style>
            .leaflet-tooltip.sismo-tooltip {
                background:#17343d !important;
                border:1px solid rgba(255,255,255,.16) !important;
                border-radius:8px !important;
                color:#fff !important;
                box-shadow:0 8px 22px rgba(20,44,51,.28) !important;
                padding:8px 10px !important;
                font-family:"Segoe UI",Arial,sans-serif !important;
            }
            .leaflet-tooltip.sismo-tooltip::before {
                border-top-color:#17343d !important;
            }
        </style>
        """))
        for _, r in eventos.iterrows():
            folium.CircleMarker(
                location=[r["latitud"], r["longitud"]],
                radius=2 + max(0, r["magnitud"] - 3) * .9,
                color=COLORES[r["categoria_max"]], fill=True, fill_opacity=.58,
                tooltip=folium.Tooltip(
                    f"<div style='font-size:11px;line-height:1.55'>"
                    f"<b style='font-size:13px'>{r['evento_id']}</b><br>"
                    f"MAGNITUD <b>{r['magnitud']:.1f} Mw</b><br>"
                    f"Profundidad <b>{r['profundidad_km']} km</b><br>"
                    f"Categoría <b>{r['categoria_max'].split(' ')[0]}</b></div>",
                    sticky=True,
                    direction="top",
                    className="sismo-tooltip",
                ),
                popup=folium.Popup(
                    f"<b>{r['evento_id']}</b><br>Magnitud: {r['magnitud']}<br>"
                    f"Profundidad: {r['profundidad_km']} km",
                    max_width=220),
            ).add_to(mapa)
        st.iframe(mapa.get_root().render(), height=410)
        st.markdown("</div>", unsafe_allow_html=True)

    with simulador_col:
        st.markdown("<div class='panel-dark'><div class='panel-title'>"
                    "Simulador de intensidad sísmica (KNN)</div>", unsafe_allow_html=True)
        opciones = ["Personalizado"] + list(ESCENARIOS)
        preset = st.selectbox("Escenario", opciones)
        base = ESCENARIOS.get(preset, {"magnitud": 6.5, "profundidad_km": 20.0,
                                       "distancia_km": 20.0})
        clave = opciones.index(preset)
        with st.form("formulario_prediccion", clear_on_submit=False):
            magnitud = st.number_input(
                "Magnitud (Mw)", min_value=LIMITES["magnitud"][0],
                max_value=LIMITES["magnitud"][1], value=base["magnitud"],
                step=0.1, format="%.1f", key=f"mag_visual_{clave}")
            profundidad = st.number_input(
                "Profundidad (km)", min_value=LIMITES["profundidad_km"][0],
                max_value=LIMITES["profundidad_km"][1], value=base["profundidad_km"],
                step=1.0, format="%.1f", key=f"prof_visual_{clave}")
            distancia = st.number_input(
                "Distancia epicentral (km)", min_value=LIMITES["distancia_km"][0],
                max_value=LIMITES["distancia_km"][1], value=base["distancia_km"],
                step=1.0, format="%.1f", key=f"dist_visual_{clave}")
            k = st.selectbox("K (vecinos)", VALORES_K,
                             index=VALORES_K.index(mejor_k))
            predecir = st.form_submit_button("Estimar intensidad")
        st.markdown("</div>", unsafe_allow_html=True)

    with resultado_col:
        resultado = None
        entrada = {"magnitud": float(magnitud), "profundidad_km": float(profundidad),
                   "distancia_km": float(distancia)}
        if predecir:
            errores = validar_entrada(entrada)
            if errores:
                for error in errores:
                    st.error(error)
            else:
                resultado = clasificar(entrada, train, k)
                st.session_state["resultado_sismo"] = resultado
        elif "resultado_sismo" in st.session_state:
            resultado = st.session_state["resultado_sismo"]

        if resultado is None:
            resultado = clasificar(base, train, mejor_k)

        prediccion = resultado["prediccion"]
        clase = "leve" if prediccion.startswith("Leve") else \
            "moderada" if prediccion.startswith("Moderada") else "fuerte"
        nombre = prediccion.split(" ")[0]
        st.markdown("<div class='panel-dark'><div class='panel-title'>"
                    "Resultado del modelo</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-box prediction-{clase}'>"
                    f"<span style='color:#dfe8e8;font-size:11px'>Intensidad estimada</span>"
                    f"<strong>{nombre}</strong></div>",
                    unsafe_allow_html=True)
        st.markdown("<div style='font-size:11px;margin-top:8px'>"
                    "Probabilidad por clase</div>", unsafe_allow_html=True)
        for categoria, prob in resultado["probabilidades"].items():
            nombre_corta = categoria.split(" ")[0]
            porcentaje = prob * 100
            color = COLORES[categoria]
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-size:10px;"
                f"margin-top:6px'><span>{nombre_corta}</span><b>{porcentaje:.1f}%</b></div>"
                f"<div style='height:7px;background:#e1e7e7;border-radius:5px'>"
                f"<div style='width:{porcentaje:.1f}%;height:7px;background:{color};"
                f"border-radius:5px'></div></div>",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    bottom_left, bottom_right = st.columns([1.55, 1.84])
    with bottom_left:
        st.markdown("<div class='panel'><div class='panel-title'>"
                    "Distribución de eventos por departamento</div>", unsafe_allow_html=True)
        por_depto = datos.groupby("departamento").size().nlargest(5).sort_values()
        fig, ax = plt.subplots(figsize=(5.5, 2.7))
        ax.barh(por_depto.index, por_depto.values, color=["#9cb1a0", "#8da99e",
                                                          "#789d96", "#648f88", "#4f7772"])
        ax.set_facecolor("white")
        fig.patch.set_alpha(0)
        ax.tick_params(colors="#4e6266", labelsize=8)
        ax.spines[:].set_visible(False)
        ax.grid(axis="x", alpha=.15)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bottom_right:
        st.markdown("<div class='panel'><div class='panel-title'>"
                    "Rendimiento del modelo</div>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.metric("K seleccionado", mejor_k)
        r2.metric("Exactitud", f"{metricas['test']['exactitud']:.1%}")
        r3.metric("F1 macro", f"{metricas['test']['f1_macro']:.2f}")
        st.caption("Modelo KNN con estandarización. Datos simulados para uso educativo.")
        st.markdown("</div>", unsafe_allow_html=True)
