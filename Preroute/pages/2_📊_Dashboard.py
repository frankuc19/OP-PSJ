# pages/2_📊_Dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import time
import streamlit.components.v1 as components # NUEVO: Importación para el auto-refresco

# 1. st.set_page_config() es el PRIMER comando de Streamlit
st.set_page_config(
    page_title="Dashboard de Logística",
    page_icon="📊",
    layout="wide"
)

# 2. Guardia de seguridad para verificar el login
if not st.session_state.get('authenticated', False):
    st.error("Debes iniciar sesión para ver esta página.")
    st.stop()

# --- SI EL USUARIO ESTÁ AUTENTICADO, EL CÓDIGO CONTINÚA DESDE AQUÍ ---

# MODIFICADO: El TTL se reduce a 60 segundos para una actualización más frecuente.
@st.cache_data(ttl=60)
def load_data_from_gsheet(sheet_url):
    """
    Carga datos desde Google Sheets. La caché expira cada 60 segundos.
    """
    try:
        csv_url = sheet_url.replace("/edit?usp=sharing", "/export?format=csv")
        df_original_case = pd.read_csv(csv_url)
        
        df = df_original_case.copy()
        df.columns = df.columns.str.lower().str.strip()

        if 'pickup_datetime' in df.columns:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
        else:
            st.warning("Advertencia en load_data: La columna 'pickup_datetime' no se encontró.")
            return None
        
        df['job_id'] = df['job_id'].astype(str)
        
        numeric_cols = ['estimated_payment', 'latrecogida', 'lonrecogida', 'latdestino', 'londestino', 'tiempoestimada', 'distancia']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                st.warning(f"Advertencia en load_data: La columna numérica '{col}' no se encontró después de convertir a minúsculas.")

        existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
        df[existing_numeric_cols] = df[existing_numeric_cols].fillna(0)
        return df
    except Exception as e:
        st.error(f"Error al cargar o procesar los datos del Google Sheet: {e}")
        st.warning("Verifica la URL del Google Sheet y que el formato de los datos sea el esperado.")
        return None

st.title("📊 Dashboard de Operaciones")
st.write("Análisis interactivo de los datos de viajes cargados. La información se actualiza automáticamente cada 60 segundos.")

# NUEVO: Componente que fuerza el refresco de la página cada 60 segundos.
components.html("<meta http-equiv='refresh' content='60'>", height=0)

GSHEET_URL = "https://docs.google.com/spreadsheets/d/1ZLwfPqKG2LP2eqp7hDkFeUCSw2jsgGhkmek3xL2KzGc/edit?usp=sharing"
df = load_data_from_gsheet(GSHEET_URL)

if df is not None:
    st.sidebar.header("Filtros Interactivos")
    
    # --- Filtros existentes ---
    job_id_input = st.sidebar.text_input("Buscar por Job ID:", key="dashboard_job_id_filter")
    selected_convenios = st.sidebar.multiselect('Convenio:', options=sorted(df['convenio'].unique()) if 'convenio' in df.columns else [], default=[], key="dashboard_convenio_filter")
    selected_tipos_servicio = st.sidebar.multiselect('Tipo de Servicio:', options=sorted(df['tipo_servicio'].unique()) if 'tipo_servicio' in df.columns else [], default=[], key="dashboard_tipo_servicio_filter")
    selected_zonas_origen = st.sidebar.multiselect('Zona de Origen:', options=sorted(df['zonaorigen'].unique()) if 'zonaorigen' in df.columns else [], default=[], key="dashboard_zona_origen_filter")
    selected_zonas_destino = st.sidebar.multiselect('Zona de Destino:', options=sorted(df['zonadestino'].unique()) if 'zonadestino' in df.columns else [], default=[], key="dashboard_zona_destino_filter")
    selected_Categorias = st.sidebar.multiselect('Categoria de Viaje:', options=sorted(df['categoria_viaje'].unique()) if 'categoria_viaje' in df.columns else [], default=[], key="dashboard_Categoria_viaje_filter")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtro por Fecha y Hora de Recogida")

    if not df['pickup_datetime'].isnull().all():
        min_date_data = df['pickup_datetime'].min().date()
        max_date_data = df['pickup_datetime'].max().date()
    else:
        min_date_data = pd.Timestamp.now().date()
        max_date_data = pd.Timestamp.now().date()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Fecha de inicio", value=min_date_data, min_value=min_date_data, max_value=max_date_data, key="dashboard_start_date")
    with col2:
        end_date = st.date_input("Fecha de fin", value=max_date_data, min_value=min_date_data, max_value=max_date_data, key="dashboard_end_date")

    col3, col4 = st.sidebar.columns(2)
    with col3:
        start_time = st.time_input("Hora de inicio", value=time(0, 0), key="dashboard_start_time")
    with col4:
        end_time = st.time_input("Hora de fin", value=time(23, 59, 59), key="dashboard_end_time")

    # Aplicación de filtros
    df_filtered = df.copy()
    
    if job_id_input and 'job_id' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['job_id'] == job_id_input]
    if selected_convenios and 'convenio' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['convenio'].isin(selected_convenios)]
    if selected_tipos_servicio and 'tipo_servicio' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['tipo_servicio'].isin(selected_tipos_servicio)]
    if selected_zonas_origen and 'zonaorigen' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['zonaorigen'].isin(selected_zonas_origen)]
    if selected_zonas_destino and 'zonadestino' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['zonadestino'].isin(selected_zonas_destino)]
    if selected_Categorias and 'categoria_viaje' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['categoria_viaje'].isin(selected_Categorias)]

    if start_date > end_date:
        st.sidebar.error("Error: La fecha de inicio no puede ser posterior a la fecha de fin.")
    else:
        df_filtered = df_filtered[
            (df_filtered['pickup_datetime'].dt.date >= start_date) &
            (df_filtered['pickup_datetime'].dt.date <= end_date)
        ]
        df_filtered = df_filtered[
            (df_filtered['pickup_datetime'].dt.time >= start_time) &
            (df_filtered['pickup_datetime'].dt.time <= end_time)
        ]

    st.header("Vista de Datos")
    st.dataframe(df_filtered)

    # --- Descarga de CSV ---
    desired_csv_columns_ordered = [
        'pickup_datetime', 'job_id', 'Categoria', 'estimated_payment', 
        'Categoria_viaje', 'latrecogida', 'lonrecogida', 
        'latdestino', 'londestino', 'Convenio', 'Tipo_servicio', 'ZonaOrigen', 'Zonadestino'
    ]
    rename_map_for_csv = {
        'pickup_datetime': 'pickup_datetime',
        'job_id': 'job_id',
        'categoria': 'Categoria',
        'estimated_payment': 'estimated_payment',
        'categoria_viaje': 'Categoria_viaje',
        'latrecogida': 'latrecogida',
        'lonrecogida': 'lonrecogida',
        'latdestino': 'latdestino',
        'londestino': 'londestino',
        'convenio': 'Convenio',
        'tipo_servicio' : 'Tipo_servicio',
        'zonaorigen' : 'ZonaOrigen',    
        'zonadestino' : 'Zonadestino'
    }
    columns_to_select_for_csv = [lc_col for lc_col in rename_map_for_csv.keys() if lc_col in df_filtered.columns]
    
    if columns_to_select_for_csv:
        df_for_download = df_filtered[columns_to_select_for_csv].copy()
        current_rename_map = {lc_col: desired_name for lc_col, desired_name in rename_map_for_csv.items() if lc_col in columns_to_select_for_csv}
        df_for_download.rename(columns=current_rename_map, inplace=True)
        final_ordered_columns_for_csv = [col for col in desired_csv_columns_ordered if col in df_for_download.columns]
        df_for_download = df_for_download[final_ordered_columns_for_csv]
        if 'pickup_datetime' in df_for_download.columns:
            df_for_download['pickup_datetime'] = df_for_download['pickup_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        csv_data = df_for_download.to_csv(index=False).encode('utf-8')
    else:
        st.warning("No hay columnas seleccionables para la descarga según la configuración.")
        csv_data = "".encode('utf-8')

    st.download_button(
        label="📥 Descargar datos como CSV",
        data=csv_data,
        file_name='datos_filtrados.csv',
        mime='text/csv',
    )
    
    st.markdown("---")
    st.header("Métricas Clave")
    
    if not df_filtered.empty:
        total_viajes = len(df_filtered)
        monto_total = df_filtered['estimated_payment'].sum() if 'estimated_payment' in df_filtered.columns else 0
        distancia_promedio = df_filtered['distancia'].mean() if 'distancia' in df_filtered.columns and df_filtered['distancia'].notna().any() else 0
        tiempo_promedio = df_filtered['tiempoestimada'].mean() if 'tiempoestimada' in df_filtered.columns and df_filtered['tiempoestimada'].notna().any() else 0
    else:
        total_viajes, monto_total, distancia_promedio, tiempo_promedio = 0, 0, 0, 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Viajes", f"{total_viajes:,.0f}")
    col2.metric("Monto Total", f"CLP ${monto_total:,.0f}")
    col3.metric("Distancia Promedio (km)", f"{distancia_promedio:.2f} km")
    col4.metric("Tiempo Promedio (min)", f"{tiempo_promedio:.2f} min")
    
    st.markdown("---")
    st.header("Análisis Gráfico")

    if not df_filtered.empty:
        col_map, col_dist = st.columns(2)
        with col_map:
            st.subheader("📍 Mapa de Origen de Viajes")
            if 'latrecogida' in df_filtered.columns and 'lonrecogida' in df_filtered.columns:
                map_data = df_filtered[['latrecogida', 'lonrecogida']].copy()
                map_data.rename(columns={'latrecogida': 'lat', 'lonrecogida': 'lon'}, inplace=True)
                map_data.dropna(subset=['lat', 'lon'], inplace=True)
                map_data = map_data[(map_data['lat'] != 0) & (map_data['lon'] != 0)]
                if not map_data.empty:
                    st.map(map_data)
                else:
                    st.info("No hay datos de geolocalización válidos para mostrar en el mapa.")
            else:
                st.warning("Columnas 'latrecogida' o 'lonrecogida' no encontradas.")
        
        with col_dist:
            st.subheader("📊 Distribución de Viajes por Monto")
            if 'estimated_payment' in df_filtered.columns:
                bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, np.inf]
                labels = [
                    '$0-$4.9k', '$5k-$9.9k', '$10k-$14.9k',
                    '$15k-$19.9k', '$20k-$24.9k', '$25k-$29.9k',
                    '$30k-$34.9k', '$35k-$39.9k', '$40k-$44.9k',
                    '+$45k'
                ]
                df_filtered['rango_monto'] = pd.cut(x=df_filtered['estimated_payment'], bins=bins, labels=labels, right=False)
                monto_counts = df_filtered['rango_monto'].value_counts().sort_index()
                st.bar_chart(monto_counts)
            else:
                st.warning("Columna 'estimated_payment' no encontrada.")
            
        st.markdown("---")

        col_convenio, col_Categoria_viaje_chart = st.columns(2)
        
        with col_convenio:
            st.subheader("📊 Viajes por Convenio")
            if 'convenio' in df_filtered.columns:
                df_convenio_chart = df_filtered[df_filtered['convenio'] != 'PERSONAL']
                convenio_counts = df_convenio_chart['convenio'].value_counts()
                st.bar_chart(convenio_counts)
            else:
                st.warning("Columna 'convenio' no encontrada.")
        
        with col_Categoria_viaje_chart:
            st.subheader("💰 Monto y Cantidad por Categoría de Viaje")
            if 'categoria_viaje' in df_filtered.columns and 'job_id' in df_filtered.columns and 'estimated_payment' in df_filtered.columns:
                analisis_Categoria = df_filtered.groupby('categoria_viaje').agg(
                    Monto_Total=('estimated_payment', 'sum'),
                    Cantidad_de_Viajes=('job_id', 'count')
                ).sort_values(by="Monto_Total", ascending=False)
                st.bar_chart(analisis_Categoria)
            else:
                st.warning("Columnas ('categoria_viaje', 'job_id', 'estimated_payment') no encontradas.")

    else:
        st.info("No hay datos para mostrar con los filtros seleccionados.")
else:
    st.warning("No se pudieron cargar los datos. Revisa la URL y los permisos de tu Google Sheet.")

if st.sidebar.button("Cerrar Sesión", key="logout_dashboard_page"):
    for key_session in st.session_state.keys():
        if key_session not in ['authenticated', 'username', 'role']:
            del st.session_state[key_session]
    st.session_state.authenticated = False
    
