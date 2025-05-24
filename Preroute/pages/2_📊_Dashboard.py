# pages/2_📊_Dashboard.py (o el nombre correcto de tu archivo)

import streamlit as st
import pandas as pd
import numpy as np

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
# Eliminamos el st.success de diagnóstico

@st.cache_data(ttl=600)
def load_data_from_gsheet(sheet_url):
    try:
        csv_url = sheet_url.replace("/edit?usp=sharing", "/export?format=csv")
        df = pd.read_csv(csv_url)
        df.columns = df.columns.str.lower().str.strip()
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
        df['job_id'] = df['job_id'].astype(str)
        numeric_cols = ['estimated_payment', 'latrecogida', 'lonrecogida', 'latdestino', 'londestino', 'tiempoestimada', 'distancia']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df
    except Exception as e:
        st.error(f"Error al cargar o procesar los datos del Google Sheet: {e}")
        st.warning("Verifica la URL del Google Sheet y que el formato de los datos sea el esperado.")
        return None

st.title("📊 Dashboard de Operaciones")
st.write("Análisis interactivo de los datos de viajes cargados")

GSHEET_URL = "https://docs.google.com/spreadsheets/d/1ZLwfPqKG2LP2eqp7hDkFeUCSw2jsgGhkmek3xL2KzGc/edit?usp=sharing"
df = load_data_from_gsheet(GSHEET_URL)

if df is not None:
    st.sidebar.header("Filtros Interactivos")
    job_id_input = st.sidebar.text_input("Buscar por Job ID:", key="dashboard_job_id_filter")
    selected_convenios = st.sidebar.multiselect('Convenio:', options=sorted(df['convenio'].unique()), default=[], key="dashboard_convenio_filter")
    selected_tipos_servicio = st.sidebar.multiselect('Tipo de Servicio:', options=sorted(df['tipo_servicio'].unique()), default=[], key="dashboard_tipo_servicio_filter")
    selected_zonas_origen = st.sidebar.multiselect('Zona de Origen:', options=sorted(df['zonaorigen'].unique()), default=[], key="dashboard_zona_origen_filter")
    selected_zonas_destino = st.sidebar.multiselect('Zona de Destino:', options=sorted(df['zonadestino'].unique()), default=[], key="dashboard_zona_destino_filter")
    selected_Categorias = st.sidebar.multiselect('Categoria de Viaje:', options=sorted(df['categoria_viaje'].unique()), default=[], key="dashboard_Categoria_viaje_filter")

    df_filtered = df.copy()
    if job_id_input:
        df_filtered = df_filtered[df_filtered['job_id'] == job_id_input]
    if selected_convenios:
        df_filtered = df_filtered[df_filtered['convenio'].isin(selected_convenios)]
    if selected_tipos_servicio:
        df_filtered = df_filtered[df_filtered['tipo_servicio'].isin(selected_tipos_servicio)]
    if selected_zonas_origen:
        df_filtered = df_filtered[df_filtered['zonaorigen'].isin(selected_zonas_origen)]
    if selected_zonas_destino:
        df_filtered = df_filtered[df_filtered['zonadestino'].isin(selected_zonas_destino)]
    if selected_Categorias:
        df_filtered = df_filtered[df_filtered['Categoria_viaje'].isin(selected_Categorias)]

    st.header("Vista de Datos")
    st.dataframe(df_filtered)
    st.download_button(
        label="📥 Descargar datos como CSV",
        data=df_filtered.to_csv(index=False).encode('utf-8'),
        file_name='datos_filtrados.csv',
        mime='text/csv',
    )
    
    st.markdown("---")
    st.header("Métricas Clave")
    
    if not df_filtered.empty:
        total_viajes = len(df_filtered)
        monto_total = df_filtered['estimated_payment'].sum()
        distancia_promedio = df_filtered['distancia'].mean()
        tiempo_promedio = df_filtered['tiempoestimada'].mean()
    else:
        total_viajes, monto_total, distancia_promedio, tiempo_promedio = 0, 0, 0, 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Viajes", f"{total_viajes:,.0f}")
    col2.metric("Monto Total", f"${monto_total:,.0f}")
    col3.metric("Distancia Promedio (km)", f"{distancia_promedio:.2f} km")
    col4.metric("Tiempo Promedio (min)", f"{tiempo_promedio:.2f} min")
    
    st.markdown("---")
    st.header("Análisis Gráfico")

    if not df_filtered.empty:
        col_map, col_dist = st.columns(2)
        with col_map:
            st.subheader("📍 Mapa de Origen de Viajes")
            map_data = df_filtered[['latrecogida', 'lonrecogida']].copy()
            map_data.rename(columns={'latrecogida': 'lat', 'lonrecogida': 'lon'}, inplace=True)
            map_data = map_data[(map_data['lat'] != 0) & (map_data['lon'] != 0)]
            if not map_data.empty:
                st.map(map_data)
            else:
                st.info("No hay datos de geolocalización para mostrar en el mapa con los filtros actuales.")
        
        with col_dist:
            st.subheader("📊 Distribución de Viajes por Monto")
            bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, np.inf]
            labels = [
                '$0 - $4,999', '$5,000 - $9,999', '$10,000 - $14,999',
                '$15,000 - $19,999', '$20,000 - $24,999', '$25,000 - $29,999',
                '$30,000 - $34,999', '$35,000 - $39,999', '$40,000 - $44,999',
                'Más de $45,000'
            ]
            if 'estimated_payment' in df_filtered.columns:
                df_filtered['rango_monto'] = pd.cut(
                    x=df_filtered['estimated_payment'], 
                    bins=bins, 
                    labels=labels, 
                    right=False
                )
                monto_counts = df_filtered['rango_monto'].value_counts().sort_index()
                st.bar_chart(monto_counts)
            else:
                st.warning("Columna 'estimated_payment' no encontrada para el gráfico de distribución.")
            
        st.markdown("---")

        col_convenio, col_Categoria_viaje_chart = st.columns(2)
        with col_convenio:
            st.subheader("📊 Viajes por Convenio")
            if 'convenio' in df_filtered.columns:
                convenio_counts = df_filtered['convenio'].value_counts()
                st.bar_chart(convenio_counts)
            else:
                st.warning("Columna 'convenio' no encontrada.")
        with col_Categoria_viaje_chart:
            st.subheader("💰 Monto y Cantidad por Categoría de Viaje")
            if 'Categoria_viaje' in df_filtered.columns and 'job_id' in df_filtered.columns and 'estimated_payment' in df_filtered.columns:
                analisis_Categoria = df_filtered.groupby('Categoria_viaje').agg(
                    Monto_Total=('estimated_payment', 'sum'),
                    Cantidad_de_Viajes=('job_id', 'count')
                )
                st.bar_chart(analisis_Categoria)
            else:
                st.warning("Columnas necesarias para el análisis por categoría de viaje no encontradas.")
    else:
        st.info("No hay datos para mostrar con los filtros seleccionados.")
else:
    st.warning("No se pudieron cargar los datos. Revisa la URL y los permisos de tu Google Sheet.")

if st.sidebar.button("Cerrar Sesión", key="logout_dashboard_page"): # Key única
    for key_session in st.session_state.keys():
        del st.session_state[key_session]
    st.switch_page("Home.py")