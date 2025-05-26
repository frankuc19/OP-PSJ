# Preroute/pages/1_🚀_Pre-Route.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, time
from PIL import Image
import os
import io
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe

# 1. st.set_page_config() es el PRIMER comando de Streamlit
st.set_page_config(
    page_title="PreRoute | Transvip",
    page_icon="🚀",
    layout="wide"
)

# 2. Guardia de seguridad para verificar el login
if not st.session_state.get('authenticated', False):
    st.error("Debes iniciar sesión para ver esta página.")
    st.stop()

# --- SI EL USUARIO ESTÁ AUTENTICADO, EL CÓDIGO CONTINÚA ---

# --- INICIALIZACIÓN DE SESSION STATE PARA RESULTADOS (MEMORIA PERSISTENTE) ---
if 'df_rutas_resultado' not in st.session_state:
    st.session_state.df_rutas_resultado = pd.DataFrame()
if 'df_no_asignadas_resultado' not in st.session_state:
    st.session_state.df_no_asignadas_resultado = pd.DataFrame()

# --- Estado de Sesión para Opciones de Filtro Dinámicas ---
if 'df_pred_preview_options' not in st.session_state:
    st.session_state.df_pred_preview_options = {
        "categories_viaje": [],
        "convenios": [],
        "categories_pred": [],
        "file_name": None
    }

# --- Funciones para crear plantillas ---
@st.cache_data
def create_template_csv(columns):
    """Crea un DataFrame de plantilla y lo convierte a CSV en memoria."""
    df_template = pd.DataFrame(columns=columns)
    buffer = io.StringIO()
    df_template.to_csv(buffer, index=False, encoding='utf-8-sig')
    return buffer.getvalue().encode('utf-8-sig')

# --- Constantes ---
LOGO_PATH = "Preroute/transvip.png"
RADIO_TIERRA_KM = 6371
PRECISION_SIMULATE_H3 = 1
INTERVALO_CAMBIO_INTERREGIONAL = 270
INTERVALO_URBANO_NOCTURNO = 70
INTERVALO_URBANO_DIURNO = 80
INTERVALO_GENERAL = 80
INTERVALO_MIN_DEFAULT_FACTOR = 1.5
MAX_INTERREGIONALES_POR_MOVIL = 2
MAX_OTRAS_DIVISIONES_POR_MOVIL = 2
CONVENIOS_OBLIGATORIOS = [
    "CODELCO CHILE", "M. PELAMBRES V REGIÓN", "MINERA LOS PELAMBRES", "CODELCO PREMIUM"
]
REQUIRED_HIST_COLS = [
    'latrecogida', 'lonrecogida', 'latdestino', 'londestino', 'tiempoestimada'
]
REQUIRED_PRED_COLS_ORIGINAL = [
    'pickup_datetime', 'job_id', 'Categoria', 'estimated_payment',
    'Categoria_viaje', 'latrecogida', 'lonrecogida',
    'latdestino', 'londestino', 'Convenio', 'Tipo_servicio', 'ZonaOrigen', 'Zonadestino'
]
RENAME_MAP_PRED = {
    'pickup_datetime': 'HoraFecha', 'job_id': 'reserva',
}
REQUIRED_PRED_COLS_RENAMED = list(RENAME_MAP_PRED.values()) + [
    col for col in REQUIRED_PRED_COLS_ORIGINAL if col not in RENAME_MAP_PRED.keys()
]

hist_template_csv = create_template_csv(REQUIRED_HIST_COLS)
pred_template_csv = create_template_csv(REQUIRED_PRED_COLS_ORIGINAL)

# --- INICIO DE LA INTERFAZ DE USUARIO ---
col_title, col_logo = st.columns([10, 1])
with col_title:
    st.title("Asignación de rutas")
    st.markdown(
        """
        Esta aplicación asigna reservas a móviles disponibles según los parámetros y reglas de negocio.
        Siga los 3 pasos: **1.** Suba el archivo de Históricos, **2.** Suba el de Predicciones y **3.** Ejecute la asignación.
        """
    )
with col_logo:
    if os.path.exists(LOGO_PATH):
        try:
            st.image(Image.open(LOGO_PATH), width=90)
        except Exception as e_img:
            st.warning(f"No se pudo cargar el logo desde '{LOGO_PATH}': {e_img}")
    else:
        st.warning(f"Logo no encontrado en: {LOGO_PATH}")

st.header("PreRoute 2.0", divider='blue')

# --- Parámetros Configurables por el Usuario (Sidebar) ---
st.sidebar.header("Parámetros de Asignación")
st.sidebar.success("Ajusta los parámetros necesarios para realizar el ruteo.")
prioritize_supervip_param = st.sidebar.toggle(
    "✨ Priorizar Categoria SUPERVIP",
    value=True,
    help="Si se activa, las reservas 'SUPERVIP' se procesarán con mayor prioridad, después de los convenios obligatorios."
)
max_moviles_param = st.sidebar.slider('Máximo de Móviles:', 0, 500, 100, 5, key="pre_max_moviles_slider")
max_monto_param = st.sidebar.slider('Monto Máximo por Móvil ($):', 100000, 1000000, 500000, 50000, format="$%d", key="pre_max_monto_slider")
max_reservas_param = st.sidebar.slider('Máximo de Reservas por Móvil:', 1, 20, 5, key="pre_max_reservas_slider")
max_horas_param = st.sidebar.slider('Máximo de Horas por Ruta:', 0, 24, 10, key="pre_max_horas_slider")

# --- Layout de carga de archivos en columnas ---
col_upload_1, col_upload_2 = st.columns(2)

with col_upload_1:
    st.subheader("1. Cargar Archivo de Históricos")
    uploaded_file_hist = st.file_uploader("Arrastra o selecciona un archivo CSV", type="csv", key="pre_hist_uploader", help="Archivo con datos históricos de viajes.")
    with st.expander("Ver formato y plantilla para Históricos"):
        st.info("El archivo debe ser un CSV y contener las siguientes columnas exactas:")
        st.code(f"{', '.join(REQUIRED_HIST_COLS)}", language='text')
        st.download_button("📥 Descargar Plantilla de Históricos", hist_template_csv, "plantilla_historicos.csv", "text/csv", key="pre_download_hist_template")
    if uploaded_file_hist:
        try:
            st.success(f"✅ Archivo '{uploaded_file_hist.name}' cargado.")
            df_hist_preview = pd.read_csv(uploaded_file_hist)
            uploaded_file_hist.seek(0)
            with st.expander(f"Ver previsualización de datos ({len(df_hist_preview)} filas)"):
                st.dataframe(df_hist_preview.head())
        except Exception as e:
            st.error(f"Error al leer el archivo de Históricos: {e}")
            uploaded_file_hist = None

with col_upload_2:
    st.subheader("2. Cargar Archivo de Predicciones")
    uploaded_file_pred = st.file_uploader("Arrastra o selecciona un archivo CSV", type="csv", key="pre_pred_uploader", help="Archivo con las reservas a asignar.")
    with st.expander("Ver formato y plantilla para Predicciones"):
        st.info("El archivo debe ser un CSV y contener las siguientes columnas exactas:")
        st.code(f"{', '.join(REQUIRED_PRED_COLS_ORIGINAL)}", language='text')
        st.download_button("📥 Descargar Plantilla de Predicciones", pred_template_csv, "plantilla_predicciones.csv", "text/csv", key="pre_download_pred_template")
    if uploaded_file_pred:
        try:
            st.success(f"✅ Archivo '{uploaded_file_pred.name}' cargado.")
            df_pred_preview = pd.read_csv(uploaded_file_pred)
            uploaded_file_pred.seek(0)
            with st.expander(f"Ver previsualización de datos ({len(df_pred_preview)} filas)"):
                st.dataframe(df_pred_preview.head())
        except Exception as e:
            st.error(f"Error al leer el archivo de Predicciones: {e}")
            uploaded_file_pred = None

# --- Lógica para Pre-cargar Opciones de Filtro (Sidebar) ---
current_pred_filename_state = st.session_state.df_pred_preview_options.get("file_name")
if uploaded_file_pred is not None:
    if uploaded_file_pred.name != current_pred_filename_state:
        try:
            temp_df_for_options = pd.read_csv(io.BytesIO(uploaded_file_pred.getvalue()))
            st.session_state.df_pred_preview_options = {
                "categories_viaje": sorted(temp_df_for_options['Categoria_viaje'].dropna().unique()) if 'Categoria_viaje' in temp_df_for_options else [],
                "convenios": sorted(temp_df_for_options['Convenio'].dropna().unique()) if 'Convenio' in temp_df_for_options else [],
                "categories_pred": sorted(temp_df_for_options['Categoria'].dropna().unique()) if 'Categoria' in temp_df_for_options else [],
                "file_name": uploaded_file_pred.name
            }
        except Exception as e_filter_load:
            st.sidebar.warning(f"No se pudieron pre-cargar filtros: {e_filter_load}")
elif current_pred_filename_state is not None:
    st.session_state.df_pred_preview_options = {"categories_viaje": [], "convenios": [], "categories_pred": [], "file_name": None}

# --- Filtros Adicionales en Sidebar ---
st.sidebar.header("Filtros Adicionales (Predicciones)")
if not st.session_state.df_pred_preview_options.get("file_name"):
    st.sidebar.info("Cargue el archivo de Predicciones para ver los filtros.")
selected_categories_viaje_user = st.sidebar.multiselect('1. Filtrar por Categoria_viaje:', options=st.session_state.df_pred_preview_options.get("categories_viaje", []), key="pre_filter_cat_viaje")
selected_convenios_user = st.sidebar.multiselect('2. Filtrar por Convenio:', options=st.session_state.df_pred_preview_options.get("convenios", []), key="pre_filter_convenio")
selected_categoria_pred_user = st.sidebar.multiselect("3. Filtrar por Categoria:", options=st.session_state.df_pred_preview_options.get("categories_pred", []), key="pre_filter_cat_pred")
st.sidebar.markdown("---")
st.sidebar.markdown("**Filtrar por Horario de Recogida (HH:MM):**")
selected_start_time_user = st.sidebar.time_input('Desde la hora:', value=time(0, 0), key="pre_filter_start_time")
selected_end_time_user = st.sidebar.time_input('Hasta la hora:', value=time(23, 59, 59), key="pre_filter_end_time")


# --- Definición de Funciones de Lógica de Negocio ---
# (Se han omitido para brevedad, pero deben estar aquí en tu código real)
def check_columns(df, required_columns, filename):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Error Crítico: Faltan columnas en '{filename}': {', '.join(missing_cols)}.")
        st.info(f"Columnas encontradas: {list(df.columns)}")
        st.info(f"Columnas requeridas: {required_columns}")
        st.stop()

def haversine_vectorized(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)); return RADIO_TIERRA_KM * c

def simulate_h3_vectorized(lats, lons, precision=PRECISION_SIMULATE_H3):
    lats = pd.to_numeric(lats, errors='coerce'); lons = pd.to_numeric(lons, errors='coerce')
    return lats.round(precision).astype(str) + "_" + lons.round(precision).astype(str)

def calcular_intervalo(ultima_reserva, nueva_reserva):
    cat_nueva = nueva_reserva.get("Categoria_viaje", "Desconocida")
    hora_nueva = nueva_reserva.get("HoraFecha")
    if pd.isna(hora_nueva): return "Error Hora", 99999
    if cat_nueva in ["Interregional", "Divisiones", "Division"]: return "Cambio/Especial", INTERVALO_CAMBIO_INTERREGIONAL
    if cat_nueva == "Urbano": return ("Urbano nocturno", INTERVALO_URBANO_NOCTURNO) if 0 <= hora_nueva.hour < 6 else ("Urbano diurno", INTERVALO_URBANO_DIURNO)
    return "General", INTERVALO_GENERAL

def monto_total_movil(movil_reservas):
    monto = 0
    for r in movil_reservas:
        pago = r.get("estimated_payment", 0)
        if pd.notnull(pago): monto += pago
    return monto

def ruta_cumple_convenio_obligatorio(lista_reservas_movil, convenios_obligatorios_list):
    if not lista_reservas_movil: return False
    for reserva in lista_reservas_movil:
        if reserva.get('Convenio') in convenios_obligatorios_list: return True
    return False

def puede_agregarse_a_movil(movil_reservas, nueva_reserva, max_reservas_param_func, max_monto_param_func, max_horas_param_func, convenios_obligatorios_func, max_interregionales_func, max_otras_divisiones_func):
    intervalos_minimos = {('urbano', 'urbano'): 80, ('urbano', 'interregional'): 200, ('interregional', 'urbano'): 200, ('interregional', 'interregional'): 200}
    categorias_viaje_validas = {"interregional", "urbano", "division", "divisiones", "desconocida", "supervip"}

    if len(movil_reservas) >= max_reservas_param_func: return False, None, None, f"Máx. {max_reservas_param_func} reservas"
    ultima_reserva = movil_reservas[-1]
    nueva_hora_recogida = nueva_reserva.get("HoraFecha")
    nueva_monto = nueva_reserva.get("estimated_payment", 0)
    nueva_cat_viaje_original = nueva_reserva.get("Categoria_viaje", "Desconocida")
    nueva_cat_viaje = nueva_cat_viaje_original.lower()
    nueva_tiempo_viaje_estimado = nueva_reserva.get("avg_travel_time")
    ultima_hora_llegada_estimada = ultima_reserva.get("estimated_arrival")

    if pd.isna(nueva_hora_recogida): return False, None, None, "Datos inválidos (hora) en nueva reserva"
    if pd.isna(ultima_hora_llegada_estimada): return False, None, None, "Hora llegada inválida en últ. reserva móvil"
    
    if nueva_cat_viaje not in categorias_viaje_validas:
        return False, None, None, f"Categoria_viaje nueva inválida: {nueva_cat_viaje_original}"
    
    tipo_int_base, intervalo_base = calcular_intervalo(ultima_reserva, nueva_reserva)
    ultima_cat_viaje_original = ultima_reserva.get("Categoria_viaje", "Desconocida")
    ultima_cat_viaje = ultima_cat_viaje_original.lower()
    if ultima_cat_viaje not in categorias_viaje_validas: return False, None, None, f"Categoria_viaje últ. reserva inválida: {ultima_cat_viaje_original}"

    intervalo_min_requerido = intervalo_base
    key_intervalo_cats_norm = (ultima_cat_viaje, nueva_cat_viaje) 
    if key_intervalo_cats_norm in intervalos_minimos:
        intervalo_min_requerido = max(intervalo_base, intervalos_minimos[key_intervalo_cats_norm])
    elif pd.notnull(nueva_tiempo_viaje_estimado) and nueva_tiempo_viaje_estimado > 0:
        intervalo_min_requerido = max(intervalo_base, int(nueva_tiempo_viaje_estimado * INTERVALO_MIN_DEFAULT_FACTOR))

    hora_minima_recogida = ultima_hora_llegada_estimada + timedelta(minutes=intervalo_min_requerido)
    if nueva_hora_recogida < hora_minima_recogida:
        return False, tipo_int_base, intervalo_min_requerido, (f"Intervalo ({nueva_cat_viaje_original} post {ultima_cat_viaje_original}) < {intervalo_min_requerido} min. "
                                                            f"Rec: {nueva_hora_recogida.strftime('%H:%M')}, Lleg: {ultima_hora_llegada_estimada.strftime('%H:%M')}, MinRec: {hora_minima_recogida.strftime('%H:%M')}")

    monto_actual = monto_total_movil(movil_reservas)
    if monto_actual + (nueva_monto if pd.notnull(nueva_monto) else 0) > max_monto_param_func: return False, None, None, f"Excede monto máx. (${max_monto_param_func:,.0f})"
    
    primera_hora_recogida = movil_reservas[0].get("HoraFecha")
    if pd.isna(primera_hora_recogida): return False, None, None, "Datos inválidos en 1ra reserva móvil"
    
    duracion_total_horas = (nueva_hora_recogida - primera_hora_recogida).total_seconds() / 3600
    if duracion_total_horas > max_horas_param_func: return False, None, None, f"Excede {max_horas_param_func}h de ruta"

    categorias_viaje_actuales_ruta = [r.get("Categoria_viaje", "").lower() for r in movil_reservas]
    num_interregional_actual = categorias_viaje_actuales_ruta.count("interregional")
    otras_divisiones_unicas_actuales = set(cat for cat in categorias_viaje_actuales_ruta if cat not in ["interregional", "urbano", "desconocida"])
    num_otras_divisiones_unicas_actual = len(otras_divisiones_unicas_actuales)
    es_nueva_interregional = (nueva_cat_viaje == "interregional")
    es_nueva_otra_division = (nueva_cat_viaje not in ["interregional", "urbano", "desconocida"])

    if es_nueva_interregional and num_interregional_actual >= max_interregionales_func: return False, None, None, f"Máx. {max_interregionales_func} Interregionales"
    if es_nueva_otra_division and nueva_cat_viaje not in otras_divisiones_unicas_actuales and num_otras_divisiones_unicas_actual >= max_otras_divisiones_func:
        return False, None, None, f"Máx. {max_otras_divisiones_func} divisiones distintas"

    ruta_propuesta = movil_reservas + [nueva_reserva]
    if len(ruta_propuesta) == max_reservas_param_func:
        if not ruta_cumple_convenio_obligatorio(ruta_propuesta, convenios_obligatorios_func):
            return False, None, None, f"Completaría ruta ({max_reservas_param_func} serv.) sin convenio obligatorio."
    
    return True, tipo_int_base, intervalo_min_requerido, None


# --- BLOQUE DE CÁLCULO PRINCIPAL ---
st.header("3. Ejecutar Asignación", divider='blue')
files_ready = uploaded_file_hist is not None and uploaded_file_pred is not None

# El cálculo principal solo se ejecuta al presionar este botón
if st.button(
    "🚀 Ejecutar Asignación",
    disabled=not files_ready,
    help="Debe cargar ambos archivos (Históricos y Predicciones) para poder ejecutar.",
    type="primary",
    use_container_width=True
):
    # Inicialización de variables locales para el cálculo
    moviles = []; rutas_asignadas_list = []; reservas_no_asignadas_list = []
    
    # --- Fase 1: Lectura y Validación Inicial ---
    with st.expander("👁️ FASE 1: Lectura y Validación de Archivos", expanded=True):
        with st.spinner('Leyendo y validando archivos...'):
            df_hist = pd.read_csv(uploaded_file_hist)
            st.write(f"✔️ Histórico '{uploaded_file_hist.name}' leído ({len(df_hist)} filas).")
            check_columns(df_hist, REQUIRED_HIST_COLS, uploaded_file_hist.name)

            df_pred_original_temp = pd.read_csv(io.BytesIO(uploaded_file_pred.getvalue()))
            df_pred = df_pred_original_temp.copy()
            st.write(f"✔️ Predicciones '{uploaded_file_pred.name}' leído ({len(df_pred)} filas).")
            check_columns(df_pred, REQUIRED_PRED_COLS_ORIGINAL, uploaded_file_pred.name)
            
            df_pred.rename(columns=RENAME_MAP_PRED, inplace=True)
            check_columns(df_pred, REQUIRED_PRED_COLS_RENAMED, f"{uploaded_file_pred.name} (renombrado)")
            
            df_pred["HoraFecha"] = pd.to_datetime(df_pred["HoraFecha"], errors='coerce')
            if df_pred["HoraFecha"].isnull().sum() > 0:
                st.warning(f"⚠️ {df_pred['HoraFecha'].isnull().sum()} fechas inválidas en Predicciones, filas eliminadas.")
                df_pred.dropna(subset=["HoraFecha"], inplace=True)
            df_hist['tiempoestimada'] = pd.to_numeric(df_hist['tiempoestimada'], errors='coerce')
            df_pred['estimated_payment'] = pd.to_numeric(df_pred['estimated_payment'], errors='coerce').fillna(0)
            if df_pred.empty: 
                st.error("No quedaron predicciones válidas tras limpieza inicial.")
                st.stop()

            if not df_pred.empty:
                st.write("---"); st.write("🔎 Aplicando filtros de barra lateral (si se seleccionaron):")
                if selected_categories_viaje_user:
                    df_pred = df_pred[df_pred['Categoria_viaje'].isin(selected_categories_viaje_user)]
                if selected_convenios_user:
                    if not df_pred.empty: df_pred = df_pred[df_pred['Convenio'].isin(selected_convenios_user)]
                if selected_categoria_pred_user:
                    if not df_pred.empty and 'Categoria' in df_pred.columns: df_pred = df_pred[df_pred['Categoria'].isin(selected_categoria_pred_user)]
                
                is_time_filter_active = not (selected_start_time_user == time(0, 0) and selected_end_time_user == time(23, 59, 59))
                if is_time_filter_active:
                    if not df_pred.empty and selected_start_time_user <= selected_end_time_user:
                        pickup_times = df_pred['HoraFecha'].dt.time
                        df_pred = df_pred[(pickup_times >= selected_start_time_user) & (pickup_times <= selected_end_time_user)]
                st.write(f"✔️ Filtros aplicados. Filas restantes en Predicciones: {len(df_pred)}.")

        st.success("Fase 1 completada.")

    # --- Fase 2: Procesamiento Histórico ---
    with st.expander("⚙️ FASE 2: Procesamiento Histórico", expanded=True):
        with st.spinner('Calculando rutas y promedios históricos...'):
            df_hist['h3_origin'] = simulate_h3_vectorized(df_hist['latrecogida'], df_hist['lonrecogida'])
            df_hist['h3_destino'] = simulate_h3_vectorized(df_hist['latdestino'], df_hist['londestino'])
            summary_df = df_hist.dropna(subset=['tiempoestimada', 'h3_origin', 'h3_destino']) \
                                .groupby(['h3_origin', 'h3_destino'], as_index=False)['tiempoestimada'] \
                                .mean().rename(columns={'tiempoestimada': 'avg_travel_time'})
            if summary_df.empty: st.warning("⚠️ No se calcularon rutas promedio desde históricos.")
        st.success("Fase 2 completada.")

    # --- Fase 3: Enriquecimiento de Predicciones ---
    with st.expander("📈 FASE 3: Enriquecimiento de Predicciones", expanded=True):
        if df_pred.empty: 
            st.warning("No hay datos de predicción para enriquecer. Saltando Fase 3.")
            df_resultado_sorted = pd.DataFrame() 
        else:
            with st.spinner('Enriqueciendo predicciones y aplicando orden de prioridad...'):
                df_pred_copy = df_pred.copy()
                df_pred_copy['h3_origin'] = simulate_h3_vectorized(df_pred_copy['latrecogida'], df_pred_copy['lonrecogida'])
                df_pred_copy['h3_destino'] = simulate_h3_vectorized(df_pred_copy['latdestino'], df_pred_copy['londestino'])
                
                if 'summary_df' in locals() and not summary_df.empty:
                    df_resultado = pd.merge(df_pred_copy, summary_df, on=['h3_origin', 'h3_destino'], how='left')
                else:
                    df_resultado = df_pred_copy.copy(); df_resultado['avg_travel_time'] = np.nan
                
                time_delta_hist = pd.to_timedelta(df_resultado['avg_travel_time'], unit='m', errors='coerce')
                df_resultado['estimated_arrival'] = df_resultado['HoraFecha'] + time_delta_hist
                mask_na_arrival = df_resultado['estimated_arrival'].isna()
                
                if mask_na_arrival.any():
                    DEFAULT_TIME_SPECIAL_CONDITIONS = 200; DEFAULT_TIME_REGULAR = 70
                    cat_viaje_lower = df_resultado.loc[mask_na_arrival, 'Categoria_viaje'].str.lower()
                    cond_interregional = cat_viaje_lower == 'interregional'
                    cond_division = cat_viaje_lower.isin(['division', 'divisiones'])
                    
                    df_resultado.loc[mask_na_arrival, 'default_time_min'] = np.where(cond_interregional | cond_division, DEFAULT_TIME_SPECIAL_CONDITIONS, DEFAULT_TIME_REGULAR)
                    default_timedelta = pd.to_timedelta(df_resultado.loc[mask_na_arrival, 'default_time_min'], unit='m')
                    df_resultado.loc[mask_na_arrival, 'estimated_arrival'] = df_resultado.loc[mask_na_arrival, 'HoraFecha'] + default_timedelta
                    df_resultado.loc[mask_na_arrival, 'tiempo_usado'] = 'Default (' + df_resultado.loc[mask_na_arrival, 'default_time_min'].astype(str) + 'min)'
                    df_resultado.drop(columns=['default_time_min'], inplace=True, errors='ignore')
                
                if 'tiempo_usado' in df_resultado.columns:
                    # Si existe, llena los valores nulos como lo tenías planeado
                    df_resultado.loc[df_resultado['tiempo_usado'].isnull(), 'tiempo_usado'] = 'Historico'
                else:
                    # Opcional: Si la columna no existe, puedes crearla
                    # df_resultado['tiempo_usado'] = 'Historico' 
                    # O puedes mostrar un error o simplemente ignorarlo
                    st.warning("La columna 'tiempo_usado' no se encontró en los datos de origen y no se pudo procesar.")

                df_resultado['is_obligatorio_convenio'] = df_resultado['Convenio'].isin(CONVENIOS_OBLIGATORIOS)
                df_resultado['is_supervip'] = (df_resultado['Categoria'] == 'SUPERVIP') if 'Categoria' in df_resultado.columns else False
                
                sort_by_cols = ['is_obligatorio_convenio']
                sort_ascending_flags = [False]
                if prioritize_supervip_param:
                    sort_by_cols.append('is_supervip')
                    sort_ascending_flags.append(False)
                sort_by_cols.extend(['HoraFecha', 'estimated_payment'])
                sort_ascending_flags.extend([True, False])
                
                df_resultado_sorted = df_resultado.sort_values(by=sort_by_cols, ascending=sort_ascending_flags, na_position='last').reset_index(drop=True)
                df_resultado_sorted.dropna(subset=['HoraFecha', 'estimated_arrival'], inplace=True) 
                if df_resultado_sorted.empty: st.warning("No hay predicciones válidas para asignar.")
            st.success("Fase 3 completada.")

    # --- Fase 4: Asignación de Reservas ---
    with st.expander("🚚 FASE 4: Asignación de Reservas", expanded=True):
        if 'df_resultado_sorted' not in locals() or df_resultado_sorted.empty:
            st.warning("No hay reservas para asignar (df_resultado_sorted vacía o no definida). Saltando Fase 4.")
        else:
            with st.spinner('Asignando reservas a móviles...'):
                reservas_a_procesar = df_resultado_sorted.to_dict('records')
                num_total_reservas = len(reservas_a_procesar)
                st.write(f"Iniciando asignación para {num_total_reservas} reservas válidas...")
                progress_bar = st.progress(0); status_text = st.empty()

                for i, reserva_actual in enumerate(reservas_a_procesar):
                    progress_bar.progress((i + 1) / num_total_reservas)
                    status_text.text(f"Procesando reserva {i+1}/{num_total_reservas}...")
                    asignado = False; mejor_motivo_no_asignado = "No se encontró móvil compatible o límite de móviles."

                    for idx, movil_actual in enumerate(moviles):
                        puede_agregar, tipo_rel, int_aplicado, motivo_rechazo = puede_agregarse_a_movil(movil_actual, reserva_actual, max_reservas_param, max_monto_param, max_horas_param, CONVENIOS_OBLIGATORIOS, MAX_INTERREGIONALES_POR_MOVIL, MAX_OTRAS_DIVISIONES_POR_MOVIL)
                        if puede_agregar:
                            movil_actual.append(reserva_actual)
                            rutas_asignadas_list.append({"movil_id": idx + 1, **reserva_actual, "tipo_relacion": tipo_rel, "min_intervalo_aplicado": int_aplicado})
                            asignado = True; break
                        else: mejor_motivo_no_asignado = motivo_rechazo
                    
                    if not asignado and len(moviles) < max_moviles_param:
                        if not (max_reservas_param == 1 and not ruta_cumple_convenio_obligatorio([reserva_actual], CONVENIOS_OBLIGATORIOS)):
                            moviles.append([reserva_actual])
                            rutas_asignadas_list.append({"movil_id": len(moviles), **reserva_actual, "tipo_relacion": "Inicio Ruta", "min_intervalo_aplicado": 0})
                            asignado = True
                        else: mejor_motivo_no_asignado = f"No puede iniciar ruta ({max_reservas_param} serv.) sin convenio oblig."
                    
                    if not asignado:
                        reserva_actual["motivo_no_asignado"] = mejor_motivo_no_asignado
                        reservas_no_asignadas_list.append(reserva_actual)
                status_text.text("Asignación completada."); progress_bar.empty()
            st.success("Fase 4 completada.")
    
    # --- GUARDADO EN MEMORIA ---
    # Al final del cálculo, guardamos los resultados en la memoria de la sesión.
    st.session_state.df_rutas_resultado = pd.DataFrame(rutas_asignadas_list)
    st.session_state.df_no_asignadas_resultado = pd.DataFrame(reservas_no_asignadas_list)

    # --- DEPURACIÓN (puedes descomentar estas líneas para verificar) ---
    # st.success("✅ RESULTADOS GUARDADOS EN MEMORIA")
    # st.write("Contenido de 'df_rutas_resultado' al momento de guardar:")
    # st.dataframe(st.session_state.df_rutas_resultado)
    # ---------------------------------------------------------------------

st.write("---")

# --- BLOQUE DE VISUALIZACIÓN DE RESULTADOS ---
# Esta sección siempre se ejecuta para mostrar los resultados guardados en memoria.
with st.expander("🏁 FASE 5: Resultados Finales", expanded=True):
    try:
        # --- DEPURACIÓN (puedes descomentar estas líneas para verificar) ---
        # st.info("🔎 LEYENDO RESULTADOS DESDE MEMORIA...")
        # st.write("Contenido de la memoria (`st.session_state`):")
        # st.write(st.session_state)
        # ---------------------------------------------------------------------

        # LECTURA DESDE MEMORIA: Cargamos los DataFrames desde el estado de la sesión.
        df_rutas = st.session_state.df_rutas_resultado
        df_no_asignadas = st.session_state.df_no_asignadas_resultado

        if df_rutas.empty and df_no_asignadas.empty:
            st.info("Aún no se han generado rutas. Carga los archivos y presiona 'Ejecutar Asignación'.")
        else:
            num_asignadas = len(df_rutas)
            num_no_asignadas = len(df_no_asignadas)
            total_reservas_intentadas = num_asignadas + num_no_asignadas
            num_moviles_usados = df_rutas['movil_id'].nunique() if not df_rutas.empty else 0
            monto_total_asignado = df_rutas['estimated_payment'].sum() if not df_rutas.empty else 0
            perc_asignadas = (num_asignadas / total_reservas_intentadas * 100) if total_reservas_intentadas > 0 else 0

            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            col_res1.metric("Reservas Procesadas", f"{total_reservas_intentadas}")
            col_res2.metric("Reservas Asignadas", f"{num_asignadas} ({perc_asignadas:.1f}%)")
            col_res3.metric("Reservas No Asignadas", f"{num_no_asignadas}")
            col_res4.metric("Móviles Utilizados", f"{num_moviles_usados} / {max_moviles_param}")
            st.metric("Monto Total Asignado", f"${monto_total_asignado:,.0f}")

            st.subheader("📋 Reservas Asignadas por Móvil")
            if not df_rutas.empty:
                cols_rutas_exist = [c for c in ['movil_id', 'reserva', 'HoraFecha', 'estimated_arrival', 'is_supervip', 'estimated_payment', 'Categoria_viaje', 'Categoria', 'Convenio', 'tipo_relacion', 'tiempo_usado', 'Tipo_servicio', 'ZonaOrigen', 'Zonadestino'] if c in df_rutas.columns]
                st.dataframe(df_rutas[cols_rutas_exist])

                # --- NUEVO: Crear dos columnas para alinear los botones ---
                col1, col2 = st.columns(2)

                # Botón de descarga en la primera columna
                with col1:
                    st.download_button(
                        label="📥 Descargar rutas_asignadas.csv",
                        data=df_rutas[cols_rutas_exist].to_csv(index=False, encoding='utf-8-sig'),
                        file_name="rutas_asignadas.csv",
                        mime="text/csv",
                        key="download_rutas",
                        use_container_width=True # Para que ocupe todo el ancho de la columna
                    )

                # Botón de envío a Google Sheets en la segunda columna
                with col2:
                    if st.button(
                        "📤 Enviar Rutas a Google Sheet",
                        key="send_to_gsheet",
                        use_container_width=True # Para que ocupe todo el ancho de la columna
                    ):
                        with st.spinner("Enviando datos a Google Sheets..."):
                            try:
                                df_para_enviar = df_rutas[cols_rutas_exist].copy()
                                if 'HoraFecha' in df_para_enviar.columns:
                                    df_para_enviar['HoraFecha'] = pd.to_datetime(df_para_enviar['HoraFecha']).dt.strftime('%Y-%m-%d %H:%M:%S')
                                if 'estimated_arrival' in df_para_enviar.columns:
                                    df_para_enviar['estimated_arrival'] = pd.to_datetime(df_para_enviar['estimated_arrival']).dt.strftime('%Y-%m-%d %H:%M:%S')

                                scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
                                creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
                                gc = gspread.authorize(creds)

                                nombre_spreadsheet = "RoutePlanning"
                                nombre_worksheet = "RutaPreroute"
                                sh = gc.open(nombre_spreadsheet).worksheet(nombre_worksheet)
                                
                                sh.clear()
                                set_with_dataframe(sh, df_para_enviar)
                                st.success(f"✅ ¡ÉXITO! Datos enviados a la pestaña '{nombre_worksheet}'!")

                            except gspread.exceptions.SpreadsheetNotFound:
                                st.error(f"❌ ERROR: Hoja de cálculo no encontrada. Revisa el nombre '{nombre_spreadsheet}' y los permisos.")
                            except gspread.exceptions.WorksheetNotFound:
                                st.error(f"❌ ERROR: Pestaña no encontrada. Revisa el nombre '{nombre_worksheet}'.")
                            except gspread.exceptions.APIError as api_error:
                                st.error(f"❌ ERROR DE API DE GOOGLE: {api_error}.")
                            except Exception as e_gsheets:
                                st.error(f"❌ Ocurrió un error inesperado: {e_gsheets}")
            
            st.subheader("🚨 Reservas No Asignadas")
            if not df_no_asignadas.empty:
                cols_no_asignadas_exist = [c for c in ['reserva', 'HoraFecha', 'is_supervip', 'estimated_payment', 'Categoria_viaje', 'Categoria', 'Convenio', 'motivo_no_asignado', 'Tipo_servicio', 'ZonaOrigen', 'Zonadestino'] if c in df_no_asignadas.columns]
                st.dataframe(df_no_asignadas[cols_no_asignadas_exist])
                st.download_button("📥 Descargar reservas_no_asignadas.csv", df_no_asignadas[cols_no_asignadas_exist].to_csv(index=False, encoding='utf-8-sig'), "reservas_no_asignadas.csv", "text/csv", key="download_no_asignadas")

    except Exception as e:
        st.error(f"Error en Fase 5: {e}")

# Botón de logout en la barra lateral
if st.sidebar.button("Cerrar Sesión", key="logout_preroute_page_final_main"):
    for key_session in list(st.session_state.keys()):
        # No borrar las claves de autenticación si existen
        if key_session not in ['authenticated', 'username', 'role']:
            del st.session_state[key_session]
    # Redirigir o limpiar la página de una manera más controlada si es necesario
    st.session_state.authenticated = False
    st.rerun()
