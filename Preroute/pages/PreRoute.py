# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, time # Import time for default time values
import traceback
from PIL import Image
import os

# --- GUARDIA DE SEGURIDAD ---
# Verifica si el usuario está autenticado antes de mostrar la página
if not st.session_state.get('authenticated', False):
    st.error("Debes iniciar sesión para ver esta página.")
    st.stop() # Detiene la ejecución del script

# --- CONTENIDO DE LA PÁGINA ---
st.set_page_config(page_title="Dashboard General", layout="wide")

st.title("📊 Dashboard General")
st.write("Esta página es visible para administradores y usuarios comunes.")

# Aquí puedes añadir tus gráficos, tablas y demás componentes...
st.write("Aquí iría el contenido del dashboard general.")



# --- Configuración de Página (solo una vez) ---
st.set_page_config(
    page_title="PreRoute | Transvip",
    page_icon="🚀",
    layout="wide"
)

# --- Estado de Sesión para Opciones de Filtro Dinámicas ---
if 'df_pred_preview_options' not in st.session_state:
    st.session_state.df_pred_preview_options = {
        "categories_viaje": [],
        "convenios": [],
        "categories_pred": [], # Para la nueva 'Categoria'
        "file_name": None
    }

# --- Título y Navegación Lateral ---
st.title("Bienvenido a PreRoute de Transvip")
st.sidebar.success("Ajusta los parámetros necesarios para realizar el ruteo.")

st.markdown(
    """
    Esta aplicación asigna reservas de transporte a móviles disponibles
    según parámetros configurables y reglas de negocio.
    Sube los archivos CSV requeridos y haz clic en 'Ejecutar Asignación'.
    """
)

# --- Logo y Título ---
LOGO_PATH = "transvip.png"
LOGO_WIDTH = 90
COLUMN_RATIO = [12, 1]

try:
    col_title, col_logo = st.columns(COLUMN_RATIO)
    with col_title:
        st.title("PreRoute 2.0")
    with col_logo:
        if os.path.exists(LOGO_PATH):
            try:
                logo_image = Image.open(LOGO_PATH)
                st.image(logo_image, width=LOGO_WIDTH)
            except Exception as e:
                st.warning(f"⚠️ No se pudo cargar la imagen del logo '{LOGO_PATH}': {e}")
        else:
            st.warning(f"⚠️ Logo no encontrado en '{LOGO_PATH}'.")
except Exception as e:
    st.warning(f"No se pudo crear el layout para el título y logo: {e}")
    st.title("PreRoute 2.0")

# --- Constantes ---
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
    'latdestino', 'londestino', 'Convenio'
]
RENAME_MAP_PRED = {
    'pickup_datetime': 'HoraFecha', 'job_id': 'reserva',
}
REQUIRED_PRED_COLS_RENAMED = list(RENAME_MAP_PRED.values()) + [
    col for col in REQUIRED_PRED_COLS_ORIGINAL if col not in RENAME_MAP_PRED.keys()
]

# --- Parámetros Configurables por el Usuario (Sidebar) ---
st.sidebar.header("Parámetros de Asignación")
prioritize_supervip_param = st.sidebar.toggle(
    "✨ Priorizar Categoria SUPERVIP",
    value=True,
    help="Si se activa, las reservas con 'Categoria' == 'SUPERVIP' se procesarán con mayor prioridad, después de los convenios obligatorios. El orden secundario es por HoraFecha y Monto Estimado."
)
max_moviles_param = st.sidebar.slider('Máximo de Móviles:', min_value=0, max_value=500, value=100, step=5)
max_monto_param = st.sidebar.slider('Monto Máximo por Móvil ($):', min_value=100000, max_value=1000000, value=500000, step=50000)
max_reservas_param = st.sidebar.slider('Máximo de Reservas por Móvil:', min_value=1, max_value=20, value=5)
max_horas_param = st.sidebar.slider('Máximo de Horas por Ruta (desde 1ra recogida):', min_value=0, max_value=24, value=10)


# --- File Uploaders ---
uploaded_file_hist = st.file_uploader("1. Subir archivo Históricos", type="csv", key="hist_uploader")
uploaded_file_pred = st.file_uploader("2. Subir archivo Predicciones", type="csv", key="pred_uploader")

# --- Lógica para Pre-cargar Opciones de Filtro (Sidebar) ---
if uploaded_file_pred is not None:
    if uploaded_file_pred.name != st.session_state.df_pred_preview_options["file_name"]:
        try:
            original_pos = uploaded_file_pred.tell()
            temp_df_for_options = pd.read_csv(uploaded_file_pred) # Leer con nombres originales
            uploaded_file_pred.seek(original_pos)

            preview_categories_viaje = []
            if 'Categoria_viaje' in temp_df_for_options.columns:
                preview_categories_viaje = sorted(temp_df_for_options['Categoria_viaje'].dropna().unique())
            
            preview_convenios = []
            if 'Convenio' in temp_df_for_options.columns:
                preview_convenios = sorted(temp_df_for_options['Convenio'].dropna().unique())

            preview_categories_pred = []
            if 'Categoria' in temp_df_for_options.columns: # Usar el nombre original 'Categoria'
                preview_categories_pred = sorted(temp_df_for_options['Categoria'].dropna().unique())
            
            st.session_state.df_pred_preview_options = {
                "categories_viaje": preview_categories_viaje,
                "convenios": preview_convenios,
                "categories_pred": preview_categories_pred,
                "file_name": uploaded_file_pred.name
            }
        except Exception as e:
            st.sidebar.warning(f"No se pudieron pre-cargar opciones de filtro: {e}")
            st.session_state.df_pred_preview_options = {"categories_viaje": [], "convenios": [], "categories_pred": [], "file_name": None}
elif st.session_state.df_pred_preview_options["file_name"] is not None:
    st.session_state.df_pred_preview_options = {"categories_viaje": [], "convenios": [], "categories_pred": [], "file_name": None}

# --- Filtros Adicionales en Sidebar ---
st.sidebar.header("Filtros Adicionales (Predicciones)")

if st.session_state.df_pred_preview_options["categories_viaje"]:
    selected_categories_viaje_user = st.sidebar.multiselect(
        '1. Filtrar por Categoria_viaje:', options=st.session_state.df_pred_preview_options["categories_viaje"], default=[]
    )
else:
    st.sidebar.info("Cargue Predicciones para ver filtros de Categoria_viaje.")
    selected_categories_viaje_user = []

if st.session_state.df_pred_preview_options["convenios"]:
    selected_convenios_user = st.sidebar.multiselect(
        '2. Filtrar por Convenio:', options=st.session_state.df_pred_preview_options["convenios"], default=[]
    )
else:
    st.sidebar.info("Cargue Predicciones para ver filtros de Convenio.")
    selected_convenios_user = []

if st.session_state.df_pred_preview_options["categories_pred"]:
    selected_categoria_pred_user = st.sidebar.multiselect(
        "3. Filtrar por Categoria:", options=st.session_state.df_pred_preview_options["categories_pred"], default=[]
    )
else:
    st.sidebar.info("Cargue Predicciones para ver filtros de Categoria (campo nuevo).")
    selected_categoria_pred_user = []

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtrar por Horario de Recogida (HH:MM):**")
default_start_time = time(0, 0)
default_end_time = time(23, 59, 59)

selected_start_time_user = st.sidebar.time_input('Desde la hora:', value=default_start_time, key='start_time_filter')
selected_end_time_user = st.sidebar.time_input('Hasta la hora:', value=default_end_time, key='end_time_filter')


# --- Funciones Auxiliares ---
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

def puede_agregarse_a_movil(movil_reservas, nueva_reserva):
    intervalos_minimos = {('urbano', 'urbano'): 80, ('urbano', 'interregional'): 200, ('interregional', 'urbano'): 200, ('interregional', 'interregional'): 200}
    categorias_viaje_validas = {"interregional", "urbano", "division", "divisiones", "desconocida", "supervip"} # SUPERVIP en Categoria_viaje

    if len(movil_reservas) >= max_reservas_param: return False, None, None, f"Máx. {max_reservas_param} reservas"
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
    if monto_actual + (nueva_monto if pd.notnull(nueva_monto) else 0) > max_monto_param: return False, None, None, f"Excede monto máx. (${max_monto_param:,.0f})"
    
    primera_hora_recogida = movil_reservas[0].get("HoraFecha")
    if pd.isna(primera_hora_recogida): return False, None, None, "Datos inválidos en 1ra reserva móvil"
    
    duracion_total_horas = (nueva_hora_recogida - primera_hora_recogida).total_seconds() / 3600
    if duracion_total_horas > max_horas_param: return False, None, None, f"Excede {max_horas_param}h de ruta"

    categorias_viaje_actuales_ruta = [r.get("Categoria_viaje", "").lower() for r in movil_reservas]
    num_interregional_actual = categorias_viaje_actuales_ruta.count("interregional")
    otras_divisiones_unicas_actuales = set(cat for cat in categorias_viaje_actuales_ruta if cat not in ["interregional", "urbano", "desconocida"])
    num_otras_divisiones_unicas_actual = len(otras_divisiones_unicas_actuales)
    es_nueva_interregional = (nueva_cat_viaje == "interregional")
    es_nueva_otra_division = (nueva_cat_viaje not in ["interregional", "urbano", "desconocida"])

    if es_nueva_interregional and num_interregional_actual >= MAX_INTERREGIONALES_POR_MOVIL: return False, None, None, f"Máx. {MAX_INTERREGIONALES_POR_MOVIL} Interregionales"
    if es_nueva_otra_division and nueva_cat_viaje not in otras_divisiones_unicas_actuales and num_otras_divisiones_unicas_actual >= MAX_OTRAS_DIVISIONES_POR_MOVIL:
        return False, None, None, f"Máx. {MAX_OTRAS_DIVISIONES_POR_MOVIL} divisiones distintas"

    ruta_propuesta = movil_reservas + [nueva_reserva]
    if len(ruta_propuesta) == max_reservas_param:
        if not ruta_cumple_convenio_obligatorio(ruta_propuesta, CONVENIOS_OBLIGATORIOS):
            return False, None, None, f"Completaría ruta ({max_reservas_param} serv.) sin convenio obligatorio."
    
    return True, tipo_int_base, intervalo_min_requerido, None

# --- Lógica Principal de la Aplicación ---
if uploaded_file_hist is not None and uploaded_file_pred is not None:
    boton_ejecutar = st.button("🚀 Ejecutar Asignación")

    if boton_ejecutar:
        df_hist = None; df_pred = None; summary_df = None; df_resultado = None
        moviles = []; rutas_asignadas_list = []; reservas_no_asignadas_list = []
        st.write("---")

        # --- Fase 1: Lectura y Validación Inicial ---
        with st.expander("👁️ FASE 1: Lectura y Validación de Archivos", expanded=True):
            with st.spinner('Leyendo y validando archivos...'):
                try:
                    df_hist = pd.read_csv(uploaded_file_hist)
                    st.write(f"✔️ Histórico '{uploaded_file_hist.name}' leído ({len(df_hist)} filas).")
                    check_columns(df_hist, REQUIRED_HIST_COLS, uploaded_file_hist.name)
                except Exception as e: st.error(f"Error crítico Histórico: {e}\n{traceback.format_exc()}"); st.stop()

                try:
                    df_pred_original_temp = pd.read_csv(uploaded_file_pred)
                    df_pred = df_pred_original_temp.copy()
                    st.write(f"✔️ Predicciones '{uploaded_file_pred.name}' leído ({len(df_pred)} filas).")
                    check_columns(df_pred, REQUIRED_PRED_COLS_ORIGINAL, uploaded_file_pred.name)
                    
                    df_pred.rename(columns=RENAME_MAP_PRED, inplace=True)
                    check_columns(df_pred, REQUIRED_PRED_COLS_RENAMED, f"{uploaded_file_pred.name} (renombrado)")
                except Exception as e: st.error(f"Error crítico Predicciones: {e}\n{traceback.format_exc()}"); st.stop()
                
                try:
                    df_pred["HoraFecha"] = pd.to_datetime(df_pred["HoraFecha"], errors='coerce')
                    if df_pred["HoraFecha"].isnull().sum() > 0:
                        st.warning(f"⚠️ {df_pred['HoraFecha'].isnull().sum()} fechas inválidas en Predicciones, filas eliminadas.")
                        df_pred.dropna(subset=["HoraFecha"], inplace=True)
                    df_hist['tiempoestimada'] = pd.to_numeric(df_hist['tiempoestimada'], errors='coerce')
                    df_pred['estimated_payment'] = pd.to_numeric(df_pred['estimated_payment'], errors='coerce').fillna(0)
                    if df_pred.empty: st.error("No quedaron predicciones válidas tras limpieza inicial."); st.stop()
                except Exception as e: st.error(f"Error en conversión de tipos: {e}\n{traceback.format_exc()}"); st.stop()

                if not df_pred.empty:
                    st.write("---"); st.write("🔎 Aplicando filtros de barra lateral (si se seleccionaron):")
                    initial_rows_bf_sidebar_filters = len(df_pred)
                    any_filter_applied_sidebar = False

                    if selected_categories_viaje_user: 
                        any_filter_applied_sidebar = True; df_pred = df_pred[df_pred['Categoria_viaje'].isin(selected_categories_viaje_user)]
                        st.write(f"✔️ Filtro 'Categoria_viaje' aplicado. Filas restantes: {len(df_pred)}.")
                        if df_pred.empty and initial_rows_bf_sidebar_filters > 0: st.error("No hay predicciones tras filtrar Categoria_viaje. Detenido."); st.stop()
                    
                    if selected_convenios_user:
                        if not df_pred.empty:
                            any_filter_applied_sidebar = True; df_pred = df_pred[df_pred['Convenio'].isin(selected_convenios_user)]
                            st.write(f"✔️ Filtro 'Convenio' aplicado. Filas restantes: {len(df_pred)}.")
                            if df_pred.empty: st.error("No hay predicciones tras filtrar Convenio. Detenido."); st.stop()
                        elif initial_rows_bf_sidebar_filters > 0 and selected_categories_viaje_user: st.warning("Filtro 'Convenio' no aplicado, datos ya filtrados a cero.")

                    if selected_categoria_pred_user:
                        if not df_pred.empty:
                            any_filter_applied_sidebar = True
                            if 'Categoria' in df_pred.columns:
                                df_pred = df_pred[df_pred['Categoria'].isin(selected_categoria_pred_user)]
                                st.write(f"✔️ Filtro 'Categoria (nuevo campo)' aplicado. Filas restantes: {len(df_pred)}.")
                                if df_pred.empty: st.error("No hay predicciones tras filtrar Categoria (nuevo campo). Detenido."); st.stop()
                            else:
                                st.warning("⚠️ Columna 'Categoria' no encontrada para filtrar. Filtro no aplicado.")
                        elif initial_rows_bf_sidebar_filters > 0 and (selected_categories_viaje_user or selected_convenios_user):
                            st.warning("Filtro 'Categoria (nuevo campo)' no aplicado, datos ya filtrados a cero.")
                    
                    is_time_filter_active = not (selected_start_time_user == default_start_time and selected_end_time_user == default_end_time)
                    if is_time_filter_active:
                        if not df_pred.empty:
                            any_filter_applied_sidebar = True
                            if selected_start_time_user > selected_end_time_user:
                                st.warning(f"Hora inicio ({selected_start_time_user.strftime('%H:%M')}) > Hora fin ({selected_end_time_user.strftime('%H:%M')}). Filtro horario no aplicado.")
                            else:
                                initial_rows_before_this_filter = len(df_pred)
                                pickup_times = df_pred['HoraFecha'].dt.time
                                df_pred = df_pred[(pickup_times >= selected_start_time_user) & (pickup_times <= selected_end_time_user)]
                                st.write(f"✔️ Filtro 'Horario Recogida' ({selected_start_time_user.strftime('%H:%M')} - {selected_end_time_user.strftime('%H:%M')}) aplicado. Filas restantes: {len(df_pred)}.")
                                if df_pred.empty and initial_rows_before_this_filter > 0: st.error("No hay predicciones tras filtrar Horario. Detenido."); st.stop()
                        elif initial_rows_bf_sidebar_filters > 0 and (selected_categories_viaje_user or selected_convenios_user or selected_categoria_pred_user): st.warning("Filtro 'Horario' no aplicado, datos ya filtrados a cero.")

                    if not any_filter_applied_sidebar: st.write("ℹ️ No se activaron filtros (o se dejaron valores por defecto que cubren todo).")
                    if df_pred.empty and initial_rows_bf_sidebar_filters > 0 and any_filter_applied_sidebar: st.error("Predicciones filtradas a cero. Detenido."); st.stop()
            st.success("Fase 1 completada.")

        # --- Fase 2: Procesamiento Histórico ---
        with st.expander("⚙️ FASE 2: Procesamiento Histórico", expanded=False):
            with st.spinner('Calculando rutas y promedios históricos...'):
                try:
                    df_hist['h3_origin'] = simulate_h3_vectorized(df_hist['latrecogida'], df_hist['lonrecogida'])
                    df_hist['h3_destino'] = simulate_h3_vectorized(df_hist['latdestino'], df_hist['londestino'])
                    summary_df = df_hist.dropna(subset=['tiempoestimada', 'h3_origin', 'h3_destino']) \
                                        .groupby(['h3_origin', 'h3_destino'], as_index=False)['tiempoestimada'] \
                                        .mean().rename(columns={'tiempoestimada': 'avg_travel_time'})
                    if summary_df.empty: st.warning("⚠️ No se calcularon rutas promedio desde históricos.")
                except Exception as e: st.error(f"Error en Fase 2: {e}\n{traceback.format_exc()}"); st.stop()
            st.success("Fase 2 completada.")


        # --- Fase 3: Enriquecimiento de Predicciones ---
        with st.expander("📈 FASE 3: Enriquecimiento de Predicciones", expanded=False):
            if df_pred.empty: 
                st.warning("No hay datos de predicción para enriquecer. Saltando Fase 3.")
                df_resultado_sorted = pd.DataFrame() 
            else:
                with st.spinner('Enriqueciendo predicciones y aplicando orden de prioridad...'):
                    try:
                        df_pred['h3_origin'] = simulate_h3_vectorized(df_pred['latrecogida'], df_pred['lonrecogida'])
                        df_pred['h3_destino'] = simulate_h3_vectorized(df_pred['latdestino'], df_pred['londestino'])
                        if summary_df is not None and not summary_df.empty:
                            df_resultado = pd.merge(df_pred, summary_df, on=['h3_origin', 'h3_destino'], how='left')
                        else:
                            df_resultado = df_pred.copy(); df_resultado['avg_travel_time'] = np.nan
                        
                        # --- INICIO DE LA MODIFICACIÓN ---
                        # 1. Calcular llegada estimada inicial usando datos históricos.
                        time_delta_hist = pd.to_timedelta(df_resultado['avg_travel_time'], unit='m', errors='coerce')
                        df_resultado['estimated_arrival'] = df_resultado['HoraFecha'] + time_delta_hist
                        
                        # 2. Identificar filas que necesitan un tiempo por defecto (sin histórico).
                        mask_na_arrival = df_resultado['estimated_arrival'].isna()
                        
                        if mask_na_arrival.any():
                            # 3. Definir tiempos por defecto.
                            DEFAULT_TIME_SPECIAL_CONDITIONS = 200 # Para Interregional, Division, etc.
                            DEFAULT_TIME_REGULAR = 70            # Para Urbano y otros.
                            
                            # 4. Crear condiciones para aplicar el tiempo especial.
                            # Se asume que las categorías mencionadas están en la columna 'Categoria_viaje'.
                            # Se usa str.lower() para ser insensible a mayúsculas/minúsculas.
                            cat_viaje_lower = df_resultado['Categoria_viaje'].str.lower()
                            cond_interregional = cat_viaje_lower == 'interregional'
                            cond_division = cat_viaje_lower.isin(['division', 'divisiones'])
                            
                            # 5. Usar np.where para asignar el tiempo por defecto correspondiente a cada fila.
                            df_resultado['default_time_min'] = np.where(
                                cond_interregional | cond_division,
                                DEFAULT_TIME_SPECIAL_CONDITIONS,
                                DEFAULT_TIME_REGULAR
                            )
                            
                            # 6. Calcular el timedelta para el tiempo por defecto y aplicarlo a las filas necesarias.
                            default_timedelta = pd.to_timedelta(df_resultado['default_time_min'], unit='m')
                            df_resultado.loc[mask_na_arrival, 'estimated_arrival'] = \
                                df_resultado.loc[mask_na_arrival, 'HoraFecha'] + default_timedelta[mask_na_arrival]
                            
                            # 7. Registrar qué tipo de tiempo se usó (Histórico o el Default específico).
                            df_resultado['tiempo_usado'] = np.where(
                                mask_na_arrival,
                                'Default (' + df_resultado['default_time_min'].astype(str) + 'min)',
                                'Historico'
                            )
                            # 8. Eliminar columna temporal.
                            df_resultado.drop(columns=['default_time_min'], inplace=True)
                        else:
                            # Si ninguna fila necesita default, todas usan Histórico.
                            df_resultado['tiempo_usado'] = 'Historico'
                        # --- FIN DE LA MODIFICACIÓN ---

                        df_resultado['is_obligatorio_convenio'] = df_resultado['Convenio'].isin(CONVENIOS_OBLIGATORIOS)

                        if prioritize_supervip_param:
                            if 'Categoria' not in df_resultado.columns:
                                st.warning("⚠️ Columna 'Categoria' no encontrada para priorización SUPERVIP. Se priorizarán convenios obligatorios, luego por fecha/monto.")
                                df_resultado_sorted = df_resultado.sort_values(
                                    by=['is_obligatorio_convenio', 'HoraFecha', 'estimated_payment'],
                                    ascending=[False, True, False],
                                    na_position='last'
                                ).reset_index(drop=True)
                            else:
                                st.write("ℹ️ Priorizando convenios obligatorios, luego 'Categoria' == 'SUPERVIP', y finalmente por fecha/monto en la asignación.")
                                df_resultado['is_supervip'] = (df_resultado['Categoria'] == 'SUPERVIP')
                                df_resultado_sorted = df_resultado.sort_values(
                                    by=['is_obligatorio_convenio', 'is_supervip', 'HoraFecha', 'estimated_payment'],
                                    ascending=[False, False, True, False], # is_obligatorio_convenio (True first), then is_supervip (True first)
                                    na_position='last'
                                ).reset_index(drop=True)
                        else: 
                            st.write("ℹ️ Priorizando convenios obligatorios, luego por fecha/monto (Priorización SUPERVIP no activa).")
                            df_resultado_sorted = df_resultado.sort_values(
                                by=['is_obligatorio_convenio', "HoraFecha", "estimated_payment"],
                                ascending=[False, True, False], # is_obligatorio_convenio (True first)
                                na_position='last'
                            ).reset_index(drop=True)
                            
                        df_resultado_sorted.dropna(subset=['HoraFecha', 'estimated_arrival'], inplace=True) 
                        if df_resultado_sorted.empty : st.warning("No hay predicciones válidas para asignar tras enriquecimiento y ordenamiento.");
                    except Exception as e: st.error(f"Error en Fase 3: {e}\n{traceback.format_exc()}"); st.stop()
                st.success("Fase 3 completada.")

        # --- Fase 4: Asignación de Reservas ---
        with st.expander("🚚 FASE 4: Asignación de Reservas", expanded=True):
            if df_resultado_sorted.empty:
                 st.warning("No hay reservas para asignar. Saltando Fase 4.")
            else:
                with st.spinner('Asignando reservas a móviles...'):
                    try:
                        reservas_a_procesar = df_resultado_sorted.to_dict('records')
                        num_total_reservas = len(reservas_a_procesar)
                        st.write(f"Iniciando asignación para {num_total_reservas} reservas válidas (ordenadas por prioridad)...")
                        
                        progress_bar = st.progress(0); status_text = st.empty()

                        for i, reserva_actual in enumerate(reservas_a_procesar):
                            progress_bar.progress((i + 1) / num_total_reservas)
                            cat_viaje_display = reserva_actual.get('Categoria_viaje', 'N/A')
                            cat_pred_display = reserva_actual.get('Categoria', 'N/A')
                            convenio_display = reserva_actual.get('Convenio', 'N/A')
                            status_text.text(f"Procesando reserva {i+1}/{num_total_reservas} (ID: {reserva_actual.get('reserva', 'N/A')}, Convenio: {convenio_display}, Cat.Viaje: {cat_viaje_display}, Cat.Pred: {cat_pred_display})...")
                            asignado = False
                            mejor_motivo_no_asignado = "No se encontró móvil compatible o se alcanzó límite de móviles"

                            for idx, movil_actual in enumerate(moviles):
                                puede_agregar, tipo_rel, int_aplicado, motivo_rechazo = puede_agregarse_a_movil(movil_actual, reserva_actual)
                                if puede_agregar:
                                    movil_actual.append(reserva_actual)
                                    rutas_asignadas_list.append({"movil_id": idx + 1, **reserva_actual, "tipo_relacion": tipo_rel, "min_intervalo_aplicado": int_aplicado})
                                    asignado = True; break
                                else: mejor_motivo_no_asignado = motivo_rechazo
                            
                            if not asignado and len(moviles) < max_moviles_param:
                                puede_iniciar_nueva_ruta = True; motivo_no_inicio_ruta = ""
                                if max_reservas_param == 1 and not ruta_cumple_convenio_obligatorio([reserva_actual], CONVENIOS_OBLIGATORIOS):
                                    puede_iniciar_nueva_ruta = False
                                    motivo_no_inicio_ruta = f"No puede iniciar ruta ({max_reservas_param} serv.) sin convenio oblig."
                                
                                if puede_iniciar_nueva_ruta:
                                    moviles.append([reserva_actual])
                                    tipo_rel_inicio = "Inicio Ruta"
                                    if max_reservas_param == 1:
                                        if ruta_cumple_convenio_obligatorio([reserva_actual], CONVENIOS_OBLIGATORIOS):
                                            tipo_rel_inicio = "Inicio Ruta (Única, Cumple Convenio Oblig.)"
                                    rutas_asignadas_list.append({"movil_id": len(moviles), **reserva_actual, "tipo_relacion": tipo_rel_inicio, "min_intervalo_aplicado": 0})
                                    asignado = True
                                else:
                                    if (mejor_motivo_no_asignado == "No se encontró móvil compatible o se alcanzó límite de móviles" or mejor_motivo_no_asignado == ""):
                                        mejor_motivo_no_asignado = motivo_no_inicio_ruta
                            if not asignado:
                                reserva_actual["motivo_no_asignado"] = mejor_motivo_no_asignado
                                reservas_no_asignadas_list.append(reserva_actual)
                        status_text.text(f"Asignación completada. {len(rutas_asignadas_list)} asignadas, {len(reservas_no_asignadas_list)} no asignadas.")
                        progress_bar.empty()
                    except Exception as e: st.error(f"Error en Fase 4: {e}\n{traceback.format_exc()}"); st.stop()
                st.success("Fase 4 completada.")


        # --- Fase 5: Resultados Finales ---
        st.subheader("🏁 Fase 5: Resultados Finales")
        try:
            df_rutas = pd.DataFrame(rutas_asignadas_list) if rutas_asignadas_list else pd.DataFrame()
            df_no_asignadas = pd.DataFrame(reservas_no_asignadas_list) if reservas_no_asignadas_list else pd.DataFrame()
            num_asignadas = len(df_rutas); num_no_asignadas = len(df_no_asignadas)
            
            total_reservas_intentadas = 0
            if 'df_resultado_sorted' in locals() and df_resultado_sorted is not None and not df_resultado_sorted.empty : 
                 total_reservas_intentadas = len(df_resultado_sorted) 
            elif 'df_pred' in locals() and df_pred is not None and not df_pred.empty and ('df_resultado_sorted' not in locals() or (df_resultado_sorted is not None and df_resultado_sorted.empty)):
                 total_reservas_intentadas = len(df_pred)
            
            num_moviles_usados = len(moviles)
            monto_total_asignado = df_rutas['estimated_payment'].sum() if not df_rutas.empty else 0
            perc_asignadas = (num_asignadas / total_reservas_intentadas * 100) if total_reservas_intentadas > 0 else 0
            perc_no_asignadas = (num_no_asignadas / total_reservas_intentadas * 100) if total_reservas_intentadas > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Reservas Procesadas (Fase 4)", f"{total_reservas_intentadas}")
            col2.metric("Reservas Asignadas", f"{num_asignadas} ({perc_asignadas:.1f}%)")
            col3.metric("Reservas No Asignadas", f"{num_no_asignadas} ({perc_no_asignadas:.1f}%)")
            col1b, col2b, _ = st.columns(3)
            col1b.metric("Móviles Utilizados", f"{num_moviles_usados} / {max_moviles_param}")
            col2b.metric("Monto Total Asignado", f"${monto_total_asignado:,.0f}")

            st.subheader("📋 Reservas Asignadas por Móvil")
            cols_display_base = ['movil_id', 'reserva', 'HoraFecha', 'estimated_arrival', 'estimated_payment', 
                                 'Categoria_viaje', 'Convenio', 'Tipo_servicio', 'ZonaOrigen', 'Zonadestino', 
                                 'tipo_relacion', 'min_intervalo_aplicado', 'avg_travel_time', 'tiempo_usado']
            
            cols_rutas = cols_display_base.copy()
            if 'Categoria' in df_rutas.columns:
                idx_cat_viaje = cols_rutas.index('Categoria_viaje') if 'Categoria_viaje' in cols_rutas else 0
                cols_rutas.insert(idx_cat_viaje + 1, 'Categoria')
            if 'is_supervip' in df_rutas.columns:
                idx_to_insert = 0
                if 'Categoria' in cols_rutas: idx_to_insert = cols_rutas.index('Categoria') + 1
                elif 'Categoria_viaje' in cols_rutas: idx_to_insert = cols_rutas.index('Categoria_viaje') + 1
                cols_rutas.insert(idx_to_insert, 'is_supervip')


            if not df_rutas.empty:
                df_rutas_display = df_rutas.copy()
                for col_date in ['HoraFecha', 'estimated_arrival']:
                    if col_date in df_rutas_display.columns: df_rutas_display[col_date] = pd.to_datetime(df_rutas_display[col_date]).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                final_cols_rutas = [c for c in cols_rutas if c in df_rutas_display.columns]
                
                st.dataframe(df_rutas_display[final_cols_rutas])
                st.download_button("📥 Descargar rutas_asignadas.csv", df_rutas_display[final_cols_rutas].to_csv(index=False, encoding='utf-8-sig'), "rutas_asignadas.csv", "text/csv")
            else: st.info("No se asignaron rutas.")

            st.subheader("🚨 Reservas No Asignadas")
            cols_no_asignadas_base = ['reserva', 'HoraFecha', 'estimated_payment', 
                                      'Categoria_viaje', 'Convenio', 'Tipo_servicio', 'ZonaOrigen', 'Zonadestino', 
                                      'avg_travel_time', 'motivo_no_asignado']
            
            cols_no_asignadas = cols_no_asignadas_base.copy()
            if 'Categoria' in df_no_asignadas.columns:
                idx_cat_viaje = cols_no_asignadas.index('Categoria_viaje') if 'Categoria_viaje' in cols_no_asignadas else 0
                cols_no_asignadas.insert(idx_cat_viaje + 1, 'Categoria')
            if 'is_supervip' in df_no_asignadas.columns:
                idx_to_insert = 0
                if 'Categoria' in cols_no_asignadas: idx_to_insert = cols_no_asignadas.index('Categoria') + 1
                elif 'Categoria_viaje' in cols_no_asignadas: idx_to_insert = cols_no_asignadas.index('Categoria_viaje') + 1
                cols_no_asignadas.insert(idx_to_insert, 'is_supervip')

            if not df_no_asignadas.empty:
                df_no_asignadas_display = df_no_asignadas.copy()
                if 'HoraFecha' in df_no_asignadas_display.columns: df_no_asignadas_display['HoraFecha'] = pd.to_datetime(df_no_asignadas_display['HoraFecha']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                final_cols_no_asignadas = [c for c in cols_no_asignadas if c in df_no_asignadas_display.columns]
                st.dataframe(df_no_asignadas_display[final_cols_no_asignadas])
                st.download_button("📥 Descargar reservas_no_asignadas.csv", df_no_asignadas_display[final_cols_no_asignadas].to_csv(index=False, encoding='utf-8-sig'), "reservas_no_asignadas.csv", "text/csv")
            else: st.info("🎉 Todas las reservas válidas fueron asignadas o no hubo para procesar.")
        except Exception as e: st.error(f"Error en Fase 5: {e}\n{traceback.format_exc()}")

else:
    st.info("Por favor, carga ambos archivos CSV (Históricos y Predicciones) para habilitar la ejecución.")

if st.sidebar.button("Cerrar Sesión"):
    st.session_state['authenticated'] = False
    st.session_state['user_role'] = None
    st.switch_page("Home.py") # Redirige a la página de login