# SISMO AI

Clasificador KNN de categoría de intensidad sísmica (Mercalli: I-IV, V-VI, VII-XII)
con aplicación web en Streamlit. **Datos simulados, uso educativo.**
No predice terremotos ni es una herramienta oficial de alerta.

## Ejecución rápida (Windows / PowerShell)

1.

       cd RUTA\sismo_ai

2. Instala las librerías:

       pip install -r requirements.txt

3. (Opcional) Reentrena el modelo; ya viene entrenado:

       python entrenar_modelo.py

4. Abre la aplicación:

       streamlit run app.py

   Si el comando `streamlit` no se reconoce:

       python -m streamlit run app.py

Se abrirá el navegador en http://localhost:8501.

Lee `INFORME.md` para la metodología, los resultados, los tres escenarios y las limitaciones.
