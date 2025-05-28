import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.oauth2.service_account import Credentials
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr


# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Plataforma de IA para Asignaciones",
    page_icon="🧠",
    layout="wide"
)

# --- ESTILOS Y ANIMACIÓN ---
st.markdown("""
<style>
/* Estilos para el efecto de vidrio, botones, etc. */
div.st-emotion-cache-1r6slb0, div.st-emotion-cache-1r6slb0 div.stBlock,
div.st-emotion-cache-16txtl3, div.st-emotion-cache-16txtl3 div.stBlock {
    background: rgba(40, 40, 40, 0.5); backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px); border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1); padding: 20px;
    margin-bottom: 15px; color: white;
}
.st-emotion-cache-16txtl3, h1, h2, h3, h4, h5, h6, .st-emotion-cache-1avcm0n { color: white; }
div.stButton > button {
    border-radius: 20px; border: none; padding: 10px 20px;
    font-size: 16px; transition: all 0.3s;
}
div.stButton > button:hover { transform: scale(1.05); }
</style>
""", unsafe_allow_html=True)
particles_html = """
<div id="particles-js" style="position: fixed; width: 100%; height: 100%; top: 0; left: 0; z-index: -1;"></div>
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
particlesJS("particles-js", {
  "particles": {"number": {"value": 150}, "color": {"value": "#cb5d2a"}, "shape": {"type": "circle"}, "opacity": {"value": 0.5}, "size": {"value": 3}, "line_linked": {"enable": true, "distance": 150, "color": "#7b2cbf", "opacity": 0.4}, "move": {"enable": true, "speed": 2}},
  "interactivity": {"events": {"onhover": {"enable": true, "mode": "repulse"}, "onclick": {"enable": true, "mode": "push"}}}, "retina_detect": true
});
</script>
"""
st.components.v1.html(particles_html, height=20)


# --- NAVEGACIÓN EN EL SIDEBAR ---
with st.sidebar:
    selected = option_menu(
        "Plataforma de IA",
        ["Análisis de datos", "Entrenamiento", "Predicción Masiva"],
        icons=['bar-chart-line-fill', 'tools', 'cpu-fill'],
        menu_icon="robot",
        default_index=0,
        styles={
            "container": {"background-color": "#333"},
            "icon": {"color": "#cb5d2a", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#555"},
            "nav-link-selected": {"background-color": "#cb5d2a"},
        }
    )

# ==============================================================================
# --- PÁGINA 1: ANÁLISIS DE DATOS ---
# ==============================================================================
if selected == "Análisis de datos":
    st.title("🔬 Análisis Estadístico Avanzado de Datos")
    st.write("Carga, procesa y analiza tus datos con pruebas de correlación, VIF y significancia estadística.")

    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        client = gspread.authorize(creds)
    except Exception as e:
        st.error(f"🔴 **Error de autenticación con Google**: Revisa tu archivo '.streamlit/secrets.toml'. Error: {e}")
        st.stop()

    source_sheet_url_analisis = st.text_input("URL de la hoja de Origen para Análisis", "https://docs.google.com/spreadsheets/d/1ZLwfPqKG2LP2eqp7hDkFeUCSw2jsgGhkmek3xL2KzGc/edit")
    source_worksheet_name_analisis = st.text_input("Nombre de la hoja de Origen para Análisis", "reservas_clientes V1.2")

    @st.cache_data(ttl=600)
    def load_and_process_data_analisis(url, worksheet):
        spreadsheet = client.open_by_url(url); sheet = spreadsheet.worksheet(worksheet)
        df = get_as_dataframe(sheet, evaluate_formulas=True).dropna(how="all").dropna(axis=1, how="all")
        df_processed = df.copy()
        zones = ["OTROS", "RESTO DE SANTIAGO", "SANTIAGO ALTA DEMANDA 1", "SANTIAGO ALTA DEMANDA 2", "SANTIAGO ALTA DEMANDA 3", "SANTIAGO SUR 1", "SANTIAGO SUR 2"]
        if "pickup_geofence_name" in df_processed.columns:
            df_processed["pickup_geofence_name"] = df_processed["pickup_geofence_name"].apply(lambda x: x if x in zones else "OTROS")
            df_processed = pd.get_dummies(df_processed, columns=["pickup_geofence_name"], prefix="zone")
        if "dow" in df_processed.columns:
            diasem_dummies = pd.get_dummies(df_processed["dow"], prefix="dow")
            df_processed = pd.concat([df_processed, diasem_dummies], axis=1).drop(columns=["dow"])
        def categorize_hour(hour):
            h = int(hour); return [1,0,0] if 0 <= h < 8 else ([0,1,0] if 8 <= h < 17 else [0,0,1])
        if "hour" in df_processed.columns:
            time_encoded = df_processed["hour"].apply(lambda x: pd.Series(categorize_hour(x), index=["hora_madrugada", "hora_dia", "hora_noche"]))
            df_processed = pd.concat([df_processed, time_encoded], axis=1).drop(columns=["hour"])
        def categorize_venta(value):
            v = int(value)
            if v <= 5000: return [1,0,0,0,0,0]
            elif v <= 10000: return [0,1,0,0,0,0]; 
            elif v <= 15000: return [0,0,1,0,0,0]
            elif v <= 20000: return [0,0,0,1,0,0]; 
            elif v <= 30000: return [0,0,0,0,1,0]
            else: return [0,0,0,0,0,1]
        if "payment" in df_processed.columns:
            venta_encoded = df_processed["payment"].apply(lambda x: pd.Series(categorize_venta(x), index=["venta_1", "venta_2", "venta_3", "venta_4", "venta_5", "venta_6"]))
            df_processed = pd.concat([df_processed, venta_encoded], axis=1).drop(columns=["payment"])
        for col in df_processed.columns: df_processed[col] = df_processed[col].astype(int)
        return df_processed

    def perform_full_analysis(df):
        results = {}
        corr_matrix = df.corr()
        results['corr_matrix'] = corr_matrix
        corr_filtered = corr_matrix[corr_matrix > 0.1].unstack().sort_values(ascending=False).dropna()
        corr_filtered = corr_filtered[corr_filtered < 1].drop_duplicates()
        results['corr_filtered'] = corr_filtered.reset_index().rename(columns={'level_0': 'Variable 1', 'level_1': 'Variable 2', 0: 'Correlación'})
        fig, ax = plt.subplots(figsize=(14, 8)); sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Mapa de Calor de Correlaciones (Pearson)"); results['heatmap_fig'] = fig
        X = df.dropna(); vif_data = pd.DataFrame(); vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        results['vif_data'] = vif_data
        significance_results = []; alpha = 0.05
        for index, row in results['corr_filtered'].iterrows():
            var1, var2, corr_value = row['Variable 1'], row['Variable 2'], row['Correlación']
            pearson_corr, p_value = pearsonr(df[var1], df[var2])
            significant = "Sí" if p_value < alpha else "No"
            significance_results.append((var1, var2, corr_value, p_value, significant))
        df_significance = pd.DataFrame(significance_results, columns=["Variable 1", "Variable 2", "Corr. Pearson", "p-Valor", "Significativa (α=0.05)?"])
        results['df_significance'] = df_significance.sort_values(by="p-Valor")
        return results

    if st.button("🚀 Realizar Análisis Completo", key="analisis_button"):
        if source_sheet_url_analisis and source_worksheet_name_analisis:
            with st.spinner("Cargando, procesando y analizando los datos..."):
                try:
                    df_processed_analisis = load_and_process_data_analisis(source_sheet_url_analisis, source_worksheet_name_analisis)
                    st.session_state['df_processed_analisis'] = df_processed_analisis
                    analysis_results = perform_full_analysis(df_processed_analisis)
                    st.session_state['analysis_results'] = analysis_results
                    st.success("¡Análisis completado! Revisa los resultados en las pestañas de abajo.")
                except Exception as e: st.error(f"🔴 **Error**: {e}")
        else: st.warning("Por favor, ingresa la URL y el nombre de la hoja de origen.")

    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        df_processed_analisis = st.session_state['df_processed_analisis']
        st.header("🔬 Resultados del Análisis Estadístico")
        tab_corr, tab_heatmap, tab_vif, tab_sig, tab_data = st.tabs(["Correlaciones Relevantes", "Mapa de Calor", "Análisis VIF", "Pruebas de Significancia", "Datos Procesados"])
        with tab_corr: st.dataframe(results['corr_filtered'])
        with tab_heatmap: st.pyplot(results['heatmap_fig'])
        with tab_vif:
            st.markdown("El **Factor de Inflación de la Varianza (VIF)** mide la multicolinealidad. VIF > 5 o 10 puede ser problemático.")
            st.dataframe(results['vif_data'].sort_values('VIF', ascending=False))
        with tab_sig:
            st.markdown("Un **p-valor < 0.05** sugiere que la correlación es estadísticamente significativa.")
            st.dataframe(results['df_significance'])
        with tab_data: st.dataframe(df_processed_analisis)

# ==============================================================================
# --- PÁGINA 2: ENTRENAMIENTO ---
# ==============================================================================
if selected == "Entrenamiento":
    st.title("🏋️ Entrenador de Modelos de Red Neuronal")
    st.write("Usa esta herramienta para entrenar el modelo de predicción usando datos desde una Google Sheet.")
    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        client = gspread.authorize(creds)
    except Exception as e: st.error(f"🔴 **Error de autenticación con Google**: Revisa tu archivo 'secrets.toml'. Error: {e}"); st.stop()
    st.subheader("1. Especifica la fuente de datos para entrenar")
    source_sheet_url = st.text_input("URL de la hoja de Google", "https://docs.google.com/spreadsheets/d/1ZLwfPqKG2LP2eqp7hDkFeUCSw2jsgGhkmek3xL2KzGc/edit")
    source_worksheet_name = st.text_input("Nombre de la hoja de Origen", "reservas_clientes V1.2")
    st.subheader("2. Inicia el Proceso")
    st.warning("**Atención**: Este proceso puede tardar y sobreescribirá cualquier modelo en la carpeta 'modelo_entrenado'.")
    if st.button("🚀 Iniciar Entrenamiento", key="train_button"):
        try:
            with st.spinner("Paso 1/7: Cargando datos..."):
                spreadsheet = client.open_by_url(source_sheet_url); sheet = spreadsheet.worksheet(source_worksheet_name)
                df = get_as_dataframe(sheet, evaluate_formulas=True).dropna(how="all").dropna(axis=1, how="all")
            st.success("✔ Datos cargados.")
            with st.spinner("Paso 2/7: Preprocesando características..."):
                zones = ["OTROS", "RESTO DE SANTIAGO", "SANTIAGO ALTA DEMANDA 1", "SANTIAGO ALTA DEMANDA 2", "SANTIAGO ALTA DEMANDA 3", "SANTIAGO SUR 1", "SANTIAGO SUR 2"]
                df["pickup_geofence_name"] = df["pickup_geofence_name"].apply(lambda x: x if x in zones else "OTROS"); df = pd.get_dummies(df, columns=["pickup_geofence_name"], prefix="zone").astype(int)
                diasem_dummies = pd.get_dummies(df["dow"], prefix="dow").astype(int); df = pd.concat([df, diasem_dummies], axis=1).drop(columns=["dow"])
                def categorize_hour(hour):
                    if 0 <= hour < 8: return [1,0,0]
                    elif 8 <= hour < 17: return [0,1,0]
                    else: return [0,0,1]
                time_encoded = df["hour"].apply(lambda x: pd.Series(categorize_hour(x), index=["hora_madrugada", "hora_dia", "hora_noche"])); df = pd.concat([df, time_encoded], axis=1).drop(columns=["hour"]).astype(int)
                def categorize_venta(value):
                    if pd.isnull(value): return [0,0,0,0,0,0]
                    if value <= 5000: return [1,0,0,0,0,0]; 
                    elif value <= 10000: return [0,1,0,0,0,0]
                    elif value <= 15000: return [0,0,1,0,0,0]; 
                    elif value <= 20000: return [0,0,0,1,0,0]
                    elif value <= 30000: return [0,0,0,0,1,0]; 
                    else: return [0,0,0,0,0,1]
                venta_encoded = df["payment"].apply(lambda x: pd.Series(categorize_venta(x), index=["venta_1", "venta_2", "venta_3", "venta_4", "venta_5", "venta_6"])); df = pd.concat([df, venta_encoded], axis=1).drop(columns=["payment"]).astype(int)
            st.success("✔ Preprocesamiento completado.")
            with st.spinner("Paso 3/7: Balanceando clases con SMOTE..."):
                Y = df["is_assigned_automatic"]; X = df.drop(columns=["is_assigned_automatic"])
                X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=42)
                smote = SMOTE(random_state=42); X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
            st.success("✔ Clases balanceadas.")
            with st.spinner("Paso 4/7: Escalando datos..."):
                scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train_resampled)
            st.success("✔ Datos escalados.")
            with st.spinner("Paso 5/7: Construyendo la red neuronal..."):
                model = Sequential([Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)), Dropout(0.3), Dense(32, activation="relu"), Dropout(0.3), Dense(16, activation="relu"), Dense(1, activation="sigmoid")])
                model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
            st.success("✔ Modelo construido.")
            with st.spinner("Paso 6/7: Entrenando el modelo..."):
                model.fit(X_train_scaled, Y_train_resampled, epochs=5, batch_size=32, verbose=0)
            st.success("✔ Modelo entrenado.")
            with st.spinner("Paso 7/7: Guardando artefactos..."):
                os.makedirs("modelo_entrenado", exist_ok=True); model.save("modelo_entrenado/modelo_mlp.h5")
                joblib.dump(scaler, "modelo_entrenado/scaler.pkl"); joblib.dump(X.columns.tolist(), "modelo_entrenado/columnas_entrenamiento.pkl")
            st.success("✔ Artefactos guardados.")
            st.balloons(); st.header("🎉 ¡Entrenamiento Finalizado! 🎉")
        except Exception as e: st.error(f"🔴 **Ocurrió un error durante el entrenamiento**: {e}")

# ==============================================================================
# --- PÁGINA 3: PREDICCIÓN MASIVA ---
# ==============================================================================
if selected == "Predicción Masiva":
    st.title("🚀 Predicción Masiva de Asignaciones")
    st.write("Carga un lote de datos desde una Google Sheet y predice la probabilidad de asignación para cada uno.")
    @st.cache_resource
    def load_model_artifacts_and_client():
        try:
            model = tf.keras.models.load_model("modelo_entrenado/modelo_mlp.h5"); scaler = joblib.load("modelo_entrenado/scaler.pkl")
            columns = joblib.load("modelo_entrenado/columnas_entrenamiento.pkl"); scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes); client = gspread.authorize(creds)
            return model, scaler, columns, client
        except FileNotFoundError: st.error("🔴 **Modelo no encontrado!** Ve a la página de **Entrenamiento** para crear un modelo primero."); return None, None, None, None
        except Exception as e: st.error(f"🔴 Error de autenticación o carga: {e}"); return None, None, None, None
    model, scaler, columnas_entrenamiento, client = load_model_artifacts_and_client()
    if model:
        sheet_name_pred = "reservas_programadas"; st.info(f"**Fuente de Datos para Predicción:** Pestaña `{sheet_name_pred}`.")
        if st.button("⚡ Iniciar Proceso de Predicción", key="predict_button"):
            try:
                with st.spinner("Paso 1/4: Cargando datos para predicción..."):
                    sheet_id = "1ZLwfPqKG2LP2eqp7hDkFeUCSw2jsgGhkmek3xL2KzGc"
                    url_csv = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_pred}"
                    df_original = pd.read_csv(url_csv); df_nuevo = df_original.copy()
                st.success("✔ Datos cargados.")
                with st.spinner("Paso 2/4: Preprocesando datos..."):
                    zones = ["OTROS", "RESTO DE SANTIAGO", "SANTIAGO ALTA DEMANDA 1", "SANTIAGO ALTA DEMANDA 2", "SANTIAGO ALTA DEMANDA 3", "SANTIAGO SUR 1", "SANTIAGO SUR 2"]; df_nuevo["pickup_geofence_name"] = df_nuevo["pickup_geofence_name"].apply(lambda x: x if x in zones else "OTROS"); df_nuevo = pd.get_dummies(df_nuevo, columns=["pickup_geofence_name"], prefix="zone")
                    if "dow" in df_nuevo.columns: dow_dummies = pd.get_dummies(df_nuevo["dow"], prefix="dow"); df_nuevo = pd.concat([df_nuevo, dow_dummies], axis=1).drop(columns=["dow"])
                    def categorize_hour(hour):
                        if 0 <= hour < 8: return [1,0,0]
                        elif 8 <= hour < 17: return [0,1,0]
                        else: return [0,0,1]
                    if "hour" in df_nuevo.columns: time_encoded = df_nuevo["hour"].apply(lambda x: pd.Series(categorize_hour(x), index=["hora_madrugada", "hora_dia", "hora_noche"])); df_nuevo = pd.concat([df_nuevo, time_encoded], axis=1).drop(columns=["hour"])
                    def categorize_venta(value):
                        if pd.isnull(value): return [0,0,0,0,0,0];
                        if value <= 5000: return [1,0,0,0,0,0]; 
                        elif value <= 10000: return [0,1,0,0,0,0];
                        elif value <= 15000: return [0,0,1,0,0,0]; 
                        elif value <= 20000: return [0,0,0,1,0,0];
                        elif value <= 30000: return [0,0,0,0,1,0]; 
                        else: return [0,0,0,0,0,1]
                    if "payment" in df_nuevo.columns: venta_encoded = df_nuevo["payment"].apply(lambda x: pd.Series(categorize_venta(x), index=["venta_1", "venta_2", "venta_3", "venta_4","venta_5","venta_6"])); df_nuevo = pd.concat([df_nuevo, venta_encoded], axis=1).drop(columns=["payment"])
                    for col in columnas_entrenamiento:
                        if col not in df_nuevo.columns: df_nuevo[col] = 0
                    df_nuevo = df_nuevo[columnas_entrenamiento]
                st.success("✔ Datos preprocesados.")
                with st.spinner("Paso 3/4: Realizando predicciones..."):
                    X_nuevo_scaled = scaler.transform(df_nuevo); Y_pred_proba_nuevo = model.predict(X_nuevo_scaled).flatten()
                st.success("✔ Predicciones realizadas.")
                with st.spinner("Paso 4/4: Generando tabla de resultados..."):
                    threshold = 0.30; df_original["Probabilidad_asignacion"] = Y_pred_proba_nuevo; df_original["Prediccion_asignacion"] = (Y_pred_proba_nuevo >= threshold).astype(int)
                    st.session_state['df_results'] = df_original
                st.success("✅ ¡Proceso de predicción completado! Revisa los resultados abajo.")
            except Exception as e: st.error(f"🔴 Ocurrió un error durante la predicción masiva: {e}")
        if 'df_results' in st.session_state:
            st.header("Resultados de la Predicción")
            results_df = st.session_state['df_results']
            st.dataframe(results_df)
            st.subheader("Opciones de Exportación")
            col1, col2 = st.columns(2)
            with col1:
                @st.cache_data
                def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')
                csv = convert_df_to_csv(results_df)
                st.download_button(label="📥 Descargar como CSV", data=csv, file_name="predicciones_masivas.csv", mime="text/csv")
            with col2:
                if st.button("📤 Exportar a Google Sheets", key="export_button"):
                    with st.spinner("Guardando resultados en Google Sheets..."):
                        try:
                            target_gsheet_url = "https://docs.google.com/spreadsheets/d/1ZLwfPqKG2LP2eqp7hDkFeUCSw2jsgGhkmek3xL2KzGc/edit"
                            target_worksheet_name = "Reservas_Manuales"; spreadsheet = client.open_by_url(target_gsheet_url)
                            try: worksheet = spreadsheet.worksheet(target_worksheet_name)
                            except gspread.exceptions.WorksheetNotFound: worksheet = spreadsheet.add_worksheet(title=target_worksheet_name, rows=1, cols=1)
                            worksheet.clear(); set_with_dataframe(worksheet, results_df)
                            st.success(f"✔ ¡Éxito! Resultados guardados en la hoja '{target_worksheet_name}'."); st.balloons()
                        except Exception as e: st.error(f"🔴 Error al guardar en Google Sheets: {e}")
