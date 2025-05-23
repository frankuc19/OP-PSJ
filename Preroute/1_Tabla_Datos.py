# pages/1_Tabla_MySQL.py

import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# Configuración de la Página de Streamlit
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Logística",
    page_icon="🗺️",
    layout="wide"
)

# -------------------------------------------------------------------
# Función para cargar y procesar datos con caché
# -------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_data_from_gsheet(sheet_url):
    """
    Carga los datos desde una URL pública de Google Sheets y realiza una
    limpieza básica para asegurar que los tipos de datos son correctos.
    """
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
        st.error(f"Error al cargar o procesar los datos. Revisa que todas las columnas necesarias existan en tu Google Sheet. Error: {e}")
        return None

# -------------------------------------------------------------------
# Cuerpo de la Aplicación
# -------------------------------------------------------------------

st.title("🗺️ Dashboard de Operaciones")
st.write("Análisis interactivo de los datos de viajes cargados")

GSHEET_URL = "https://docs.google.com/spreadsheets/d/1ZLwfPqKG2LP2eqp7hDkFeUCSw2jsgGhkmek3xL2KzGc/edit?usp=sharing"

df = load_data_from_gsheet(GSHEET_URL)

if df is not None:
    st.sidebar.header("Filtros Interactivos")
    
    job_id_input = st.sidebar.text_input("Buscar por Job ID:")
    selected_convenios = st.sidebar.multiselect('Convenio:', options=sorted(df['convenio'].unique()), default=[])
    selected_tipos_servicio = st.sidebar.multiselect('Tipo de Servicio:', options=sorted(df['tipo_servicio'].unique()), default=[])
    selected_zonas_origen = st.sidebar.multiselect('Zona de Origen:', options=sorted(df['zonaorigen'].unique()), default=[])
    selected_zonas_destino = st.sidebar.multiselect('Zona de Destino:', options=sorted(df['zonadestino'].unique()), default=[])
    selected_categorias = st.sidebar.multiselect('Categoria de Viaje:', options=sorted(df['categoria_viaje'].unique()), default=[])

    df_filtered = df
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
    if selected_categorias:
        df_filtered = df_filtered[df_filtered['categoria_viaje'].isin(selected_categorias)]

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
            st.map(map_data)
        
        with col_dist:
            # --- INICIO DE LA MODIFICACIÓN ---
            st.subheader("📊 Distribución de Viajes por Monto")
            
            # 1. Definir los límites de los intervalos (bins) manualmente.
            #    np.inf representa "infinito", para capturar todos los valores superiores.
            bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, np.inf]
            
            # 2. Crear las etiquetas para cada intervalo.
            labels = [
                '$0 - $4,999', '$5,000 - $9,999', '$10,000 - $14,999',
                '$15,000 - $19,999', '$20,000 - $24,999', '$25,000 - $29,999',
                '$30,000 - $34,999', '$35,000 - $39,999', '$40,000 - $44,999',
                'Más de $45,000'
            ]
            
            # Usar pd.cut para agrupar los viajes en los intervalos definidos
            df_filtered['rango_monto'] = pd.cut(
                x=df_filtered['estimated_payment'], 
                bins=bins, 
                labels=labels, 
                right=False # Intervalo [inicio, fin)
            )
            
            # Contar cuántos viajes hay en cada rango y ordenar por el rango
            monto_counts = df_filtered['rango_monto'].value_counts().sort_index()
            
            st.bar_chart(monto_counts)
            # --- FIN DE LA MODIFICACIÓN ---
            
        st.markdown("---")

        col_convenio, col_categoria = st.columns(2)
        with col_convenio:
            st.subheader("📊 Viajes por Convenio")
            convenio_counts = df_filtered['convenio'].value_counts()
            st.bar_chart(convenio_counts)
        with col_categoria:
            st.subheader("💰 Monto por Categoría de Viaje")
            monto_por_categoria = df_filtered.groupby('categoria_viaje')['estimated_payment'].sum()
            st.bar_chart(monto_por_categoria)
    else:
        st.info("No hay datos para mostrar con los filtros seleccionados.")
else:
    st.warning("No se pudieron cargar los datos. Revisa la URL y los permisos de tu Google Sheet.")