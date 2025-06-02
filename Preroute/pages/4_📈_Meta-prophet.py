import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import io

# --- 0. Configuración de Nombres de Columnas (igual que tu script) ---
FECHA_COLUMN_NAME = 'Fecha'
DEMANDA_COLUMN_NAME_PRINCIPAL = 'TOTAL_PAX_SERVICIO'
DEMANDA_COLUMN_NAME_FALLBACK = 'Demanda'
SERVICE_TYPE_COLUMN_NAME = 'CODIGO_UNICO_SERVICIO'
FERIADO_FECHA_COLUMN_NAME = 'Fecha_Evento'
FERIADO_NOMBRE_COLUMN_NAME = 'Evento'

# --- Funciones Auxiliares de Streamlit ---
def display_df_info(df, name):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    with st.expander(f"Información de {name}"):
        st.text(s)

@st.cache_data # Cache para la carga de datos
def load_data(uploaded_file, encoding_options=('utf-8', 'latin1', 'cp1252')):
    if uploaded_file is not None:
        original_pos = uploaded_file.tell()
        uploaded_file.seek(0)
        
        df = None
        successful_encoding = None
        for encoding in encoding_options:
            try:
                df_temp = pd.read_csv(uploaded_file, encoding=encoding)
                df = df_temp 
                successful_encoding = encoding
                uploaded_file.seek(0) 
                break 
            except UnicodeDecodeError:
                uploaded_file.seek(0) 
                continue
            except Exception as e:
                uploaded_file.seek(0) 
                return None 
        
        uploaded_file.seek(original_pos) 

        if df is not None:
            print(f"Info cache: Archivo '{uploaded_file.name}' cargado con '{successful_encoding}'.") 
            return df
        else:
            return None 
    return None

@st.cache_data # Cache para el preprocesamiento de feriados
def preprocess_feriados(_df_feriados_eventos_base): 
    if _df_feriados_eventos_base is None:
        return pd.DataFrame(columns=['ds', 'holiday', 'lower_window', 'upper_window']) 

    df = _df_feriados_eventos_base.copy()
    try:
        df = df.rename(columns={FERIADO_NOMBRE_COLUMN_NAME: 'holiday', FERIADO_FECHA_COLUMN_NAME: 'ds'})
        if 'holiday' in df.columns:
            df['holiday'] = df['holiday'].astype(str)
        
        df['ds_cleaned'] = df['ds'].astype(str).apply(lambda x: x.split(',')[0].strip())
        
        date_formats_to_try = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
        parsed_dates = None
        parsed_format = None
        for fmt in date_formats_to_try:
            try:
                current_parsed_dates = pd.to_datetime(df['ds_cleaned'], format=fmt, errors='coerce')
                if not current_parsed_dates.isnull().all(): 
                    parsed_dates = current_parsed_dates
                    parsed_format = fmt
                    break
            except ValueError:
                continue
        
        if parsed_dates is None or parsed_dates.isnull().all():
            return pd.DataFrame(columns=['ds', 'holiday', 'lower_window', 'upper_window'])

        df['ds'] = parsed_dates
        df = df.drop(columns=['ds_cleaned'], errors='ignore')
        
        original_rows = len(df)
        df.dropna(subset=['ds', 'holiday'], inplace=True)
        df = df[df['holiday'].str.lower() != 'nan']
        if len(df) < original_rows:
            pass

        df['lower_window'] = df.get('lower_window', pd.Series(0, index=df.index))
        df['upper_window'] = df.get('upper_window', pd.Series(0, index=df.index))
        df['lower_window'] = pd.to_numeric(df['lower_window'], errors='coerce').fillna(0).astype(int)
        df['upper_window'] = pd.to_numeric(df['upper_window'], errors='coerce').fillna(0).astype(int)

        if df.empty:
            pass
        print(f"Info cache: Feriados preprocesados. Formato detectado: {parsed_format if parsed_format else 'Varios/Error'}")
        return df[['ds', 'holiday', 'lower_window', 'upper_window']]
    except Exception as e:
        print(f"Error en cache (preprocess_feriados): {e}")
        return pd.DataFrame(columns=['ds', 'holiday', 'lower_window', 'upper_window'])

@st.cache_data(persist="disk")
def train_and_forecast_service(_df_prophet_train_arg, _df_feriados_eventos_processed_arg, service_type_arg, future_days_arg, changepoint_prior_scale_arg, seasonality_prior_scale_arg, holidays_prior_scale_arg):
    
    if _df_prophet_train_arg.empty or len(_df_prophet_train_arg) < 2:
        print(f"Advertencia de cache (train_and_forecast_service): Datos de entrenamiento insuficientes para '{service_type_arg}'.")
        return None, None, {'MAE': float('nan'), 'MAPE': float('nan')}, pd.DataFrame()

    model = Prophet(
        holidays=_df_feriados_eventos_processed_arg if (_df_feriados_eventos_processed_arg is not None and not _df_feriados_eventos_processed_arg.empty) else None,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale_arg,
        seasonality_prior_scale=seasonality_prior_scale_arg,
        holidays_prior_scale=holidays_prior_scale_arg
    )
    
    metrics = {'MAE': float('nan'), 'MAPE': float('nan')}
    forecast_df = None
    holiday_impacts_df = pd.DataFrame()

    try:
        model.fit(_df_prophet_train_arg)
        future = model.make_future_dataframe(periods=future_days_arg)
        forecast_df = model.predict(future)
        forecast_df = pd.merge(forecast_df, _df_prophet_train_arg[['ds', 'y']], on='ds', how='left')

        historical_fcst = forecast_df[forecast_df['y'].notna()]
        if not historical_fcst.empty and len(historical_fcst) > 1:
            metrics['MAE'] = mean_absolute_error(historical_fcst['y'], historical_fcst['yhat'])
            if not (historical_fcst['y'] == 0).any():
                 metrics['MAPE'] = mean_absolute_percentage_error(historical_fcst['y'], historical_fcst['yhat'])
            else: 
                y_true_mape = historical_fcst['y'][historical_fcst['y'] != 0]
                y_pred_mape = historical_fcst['yhat'][historical_fcst['y'] != 0]
                if not y_true_mape.empty:
                    metrics['MAPE'] = ((y_true_mape - y_pred_mape).abs() / y_true_mape.abs()).mean()
                else:
                    metrics['MAPE'] = float('inf') 

        if _df_feriados_eventos_processed_arg is not None and not _df_feriados_eventos_processed_arg.empty:
            service_holiday_impacts = []
            unique_holidays = _df_feriados_eventos_processed_arg['holiday'].unique()
            for hol_name in unique_holidays:
                col_name_options = [hol_name, hol_name.replace(' ', '_')] 
                actual_col_name = next((opt for opt in col_name_options if opt in forecast_df.columns), None)
                if actual_col_name:
                    holiday_effect = forecast_df[forecast_df[actual_col_name].abs() > 1e-6] 
                    if not holiday_effect.empty:
                        avg_impact = holiday_effect[actual_col_name].mean()
                        service_holiday_impacts.append({
                            'feriado': hol_name, 
                            'impacto_promedio': avg_impact, 
                            'dias_afectados': len(holiday_effect)
                        })
            if service_holiday_impacts:
                holiday_impacts_df = pd.DataFrame(service_holiday_impacts).sort_values(by='impacto_promedio', key=abs, ascending=False)
        
        print(f"Info cache (train_and_forecast_service): Modelo entrenado y pronóstico generado para '{service_type_arg}'.")

    except Exception as e:
        print(f"Error en cache (train_and_forecast_service) para '{service_type_arg}': {e}")
        return None, None, metrics, holiday_impacts_df 

    return model, forecast_df, metrics, holiday_impacts_df
# --- Configuración de la Página Streamlit ---
st.set_page_config(layout="wide", page_title="Panel de Pronóstico de Ventas", page_icon="📊")

st.title("📊 Panel Avanzado de Pronóstico de Ventas con Prophet")
st.markdown("""
Bienvenido al panel de pronóstico de ventas. Sube tus archivos de datos y de feriados,
selecciona los servicios y configura los parámetros para generar los pronósticos.
""")

# --- Barra Lateral para Carga de Archivos y Filtros Globales ---
st.sidebar.header("📁 Carga de Archivos")
uploaded_demanda_file = st.sidebar.file_uploader("Cargar archivo de Demanda (Data_prophet)", type=["csv"])
uploaded_feriados_file = st.sidebar.file_uploader("Cargar archivo de Feriados (ds_prophet - Feriados)", type=["csv"])

df_demanda_full = None
df_feriados_eventos_base = None 
df_feriados_eventos_processed = pd.DataFrame(columns=['ds', 'holiday', 'lower_window', 'upper_window']) 

if uploaded_demanda_file:
    df_demanda_full = load_data(uploaded_demanda_file)
    if df_demanda_full is None:
        st.sidebar.error(f"No se pudo cargar el archivo de demanda '{uploaded_demanda_file.name}'. Revisa los formatos o encodings.")
    else:
        st.sidebar.success(f"Archivo de demanda '{uploaded_demanda_file.name}' cargado.")

if uploaded_feriados_file:
    df_feriados_eventos_base = load_data(uploaded_feriados_file)
    if df_feriados_eventos_base is None:
        st.sidebar.error(f"No se pudo cargar el archivo de feriados '{uploaded_feriados_file.name}'.")
    else:
        st.sidebar.success(f"Archivo de feriados '{uploaded_feriados_file.name}' cargado. Procesando...")
        df_feriados_eventos_processed = preprocess_feriados(df_feriados_eventos_base)
        if df_feriados_eventos_processed.empty and not df_feriados_eventos_base.empty :
             st.sidebar.warning("El DataFrame de feriados quedó vacío después del preprocesamiento. Verifica formatos de fecha y contenido.")
        elif not df_feriados_eventos_processed.empty:
            st.sidebar.success("Feriados preprocesados.")


if df_demanda_full is not None:
    st.sidebar.header("⚙️ Filtros Globales")
    
    unique_service_types = []
    if SERVICE_TYPE_COLUMN_NAME in df_demanda_full.columns:
        df_demanda_full[SERVICE_TYPE_COLUMN_NAME] = df_demanda_full[SERVICE_TYPE_COLUMN_NAME].astype(str).str.strip()
        unique_service_types = sorted([s_type for s_type in df_demanda_full[SERVICE_TYPE_COLUMN_NAME].unique() if pd.notna(s_type) and s_type.lower() != 'nan' and s_type.strip() != ''])
    
    if not unique_service_types:
        st.sidebar.warning(f"No se encontraron tipos de servicio válidos en la columna '{SERVICE_TYPE_COLUMN_NAME}'.")
        selected_service_types = []
    else:
        st.sidebar.subheader("Tipos de Servicio a Analizar")
        container = st.sidebar.container()
        select_all_services_key = "select_all_services_checkbox"
        select_all_services = container.checkbox("Seleccionar Todos los Servicios", True, key=select_all_services_key)
        
        multiselect_key = "service_multiselect"
        if select_all_services:
            selected_service_types = container.multiselect(
                "Elige los servicios:",
                options=unique_service_types,
                default=unique_service_types,
                key=multiselect_key
            )
        else:
            selected_service_types = container.multiselect(
                "Elige los servicios:",
                options=unique_service_types,
                default=unique_service_types[0] if unique_service_types else None,
                key=multiselect_key
            )

    st.sidebar.subheader("🔧 Parámetros del Modelo Prophet")
    future_days_global = st.sidebar.slider("Días a predecir en el futuro:", 30, 730, 365, 5, key="future_days_slider")
    
    adv_settings = st.sidebar.expander("Parámetros de Regularización de Prophet")
    changepoint_prior_scale_global = adv_settings.number_input("Changepoint Prior Scale:", 0.001, 1.0, 0.05, 0.001, format="%.3f", key="cps_input")
    seasonality_prior_scale_global = adv_settings.number_input("Seasonality Prior Scale:", 0.01, 10.0, 1.0, 0.01, format="%.2f", key="sps_input")
    holidays_prior_scale_global = adv_settings.number_input("Holidays Prior Scale:", 0.01, 10.0, 1.0, 0.01, format="%.2f", key="hps_input")

    if selected_service_types:
        tab_overview, tab_service_detail, tab_data_explorer = st.tabs(["📈 Resumen General", "🔍 Análisis por Servicio", "📄 Explorador de Datos"])

        all_forecasts_list = []
        all_metrics_list = []
        all_holiday_impacts_list_global = []
        processed_services_forecasts = {} 
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_services_to_process = len(selected_service_types)

        for i, service_type in enumerate(selected_service_types):
            status_text.text(f"Procesando servicio: {service_type} ({i+1}/{total_services_to_process})...")
            
            df_demanda_current_service_raw = df_demanda_full[df_demanda_full[SERVICE_TYPE_COLUMN_NAME] == service_type].copy()
            if df_demanda_current_service_raw.empty:
                continue

            actual_demand_column_to_use = None
            if DEMANDA_COLUMN_NAME_PRINCIPAL in df_demanda_current_service_raw.columns:
                actual_demand_column_to_use = DEMANDA_COLUMN_NAME_PRINCIPAL
            elif DEMANDA_COLUMN_NAME_FALLBACK in df_demanda_current_service_raw.columns:
                actual_demand_column_to_use = DEMANDA_COLUMN_NAME_FALLBACK
            else:
                continue
            
            df_prophet_train_pre = df_demanda_current_service_raw.rename(columns={FECHA_COLUMN_NAME: 'ds', actual_demand_column_to_use: 'y'})
            
            try:
                parsed_dates = pd.to_datetime(df_prophet_train_pre['ds'], errors='coerce')
                formats_to_try = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%b-%Y', '%Y/%d/%m'] 
                idx_failed = parsed_dates.isnull()

                if idx_failed.any():
                    for fmt_date in formats_to_try:
                        if not idx_failed.any(): break 
                        parsed_dates.loc[idx_failed] = pd.to_datetime(df_prophet_train_pre.loc[idx_failed, 'ds'], format=fmt_date, errors='coerce')
                        idx_failed = parsed_dates.isnull() 
                
                df_prophet_train_pre['ds'] = parsed_dates
                num_failed_dates = df_prophet_train_pre['ds'].isnull().sum()
                df_prophet_train_pre.dropna(subset=['ds'], inplace=True)

            except Exception as e_date:
                continue

            if df_prophet_train_pre.empty:
                continue

            df_prophet_train_pre['y'] = pd.to_numeric(df_prophet_train_pre['y'], errors='coerce')
            df_prophet_train_pre.dropna(subset=['y'], inplace=True)
            df_prophet_train_pre = df_prophet_train_pre.sort_values(by='ds')

            if len(df_prophet_train_pre) < 2:
                continue
            
            df_prophet_train = df_prophet_train_pre[['ds', 'y']].copy()

            model, forecast, metrics, holiday_impacts_df_service = train_and_forecast_service(
                df_prophet_train, 
                df_feriados_eventos_processed, 
                service_type, 
                future_days_global,
                changepoint_prior_scale_global,
                seasonality_prior_scale_global,
                holidays_prior_scale_global
            )

            fig_components_service = None 
            
            if model and forecast is not None: 
                plt.close('all') 
                try:
                    fig_components_service = model.plot_components(forecast)
                except Exception as e_comps:
                    print(f"Advertencia: No se pudieron generar los componentes del modelo para '{service_type}': {e_comps}")
            
            if forecast is not None:
                processed_services_forecasts[service_type] = {
                    'model': model, 
                    'forecast_df': forecast, 
                    'metrics': metrics, 
                    'fig_components': fig_components_service, 
                    'train_data': df_prophet_train,
                    'holiday_impacts_df': holiday_impacts_df_service
                }
                all_forecasts_list.append(forecast[['ds', 'yhat']].copy()) 
                all_metrics_list.append({'servicio': service_type, **metrics, 'puntos_datos': len(df_prophet_train)})
                if not holiday_impacts_df_service.empty:
                     all_holiday_impacts_list_global.extend([{'servicio': service_type, **imp} for imp in holiday_impacts_df_service.to_dict('records')])
            progress_bar.progress((i + 1) / total_services_to_process)
        status_text.text(f"Procesamiento completado para {len(processed_services_forecasts)} de {total_services_to_process} servicios seleccionados.")


        # --- Pestaña: Resumen General ---
        with tab_overview:
            st.header("📈 Resumen General de Pronósticos")

            if not all_forecasts_list:
                st.warning("No hay pronósticos individuales para agregar. Revisa la selección de servicios o los datos de entrada.")
            else:
                st.subheader("Visión General Agregada: Demanda Total")
                try:
                    df_total_actuals_agg = None
                    actual_demand_col_for_total = None
                    if DEMANDA_COLUMN_NAME_PRINCIPAL in df_demanda_full.columns:
                        actual_demand_col_for_total = DEMANDA_COLUMN_NAME_PRINCIPAL
                    elif DEMANDA_COLUMN_NAME_FALLBACK in df_demanda_full.columns:
                        actual_demand_col_for_total = DEMANDA_COLUMN_NAME_FALLBACK

                    if actual_demand_col_for_total:
                        df_total_actuals_prep = df_demanda_full[[FECHA_COLUMN_NAME, actual_demand_col_for_total]].copy()
                        df_total_actuals_prep = df_total_actuals_prep.rename(columns={FECHA_COLUMN_NAME: 'ds', actual_demand_col_for_total: 'y'})
                        
                        parsed_ds_total_actuals = pd.to_datetime(df_total_actuals_prep['ds'], errors='coerce')
                        formats_to_try_total = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']
                        idx_failed_total = parsed_ds_total_actuals.isnull()
                        if idx_failed_total.any():
                            for fmt in formats_to_try_total:
                                if not idx_failed_total.any(): break
                                parsed_ds_total_actuals.loc[idx_failed_total] = pd.to_datetime(df_total_actuals_prep.loc[idx_failed_total, 'ds'], format=fmt, errors='coerce')
                                idx_failed_total = parsed_ds_total_actuals.isnull()
                        
                        df_total_actuals_prep['ds'] = parsed_ds_total_actuals
                        df_total_actuals_prep.dropna(subset=['ds'], inplace=True)
                        df_total_actuals_prep['y'] = pd.to_numeric(df_total_actuals_prep['y'], errors='coerce')
                        df_total_actuals_prep.dropna(subset=['y'], inplace=True)
                        df_total_actuals_agg = df_total_actuals_prep.groupby('ds')['y'].sum().reset_index().rename(columns={'y': 'y_total_actual'})
                        df_total_actuals_agg.sort_values('ds', inplace=True)
                    
                    df_total_prev_year_agg = None
                    if df_total_actuals_agg is not None and not df_total_actuals_agg.empty:
                        df_total_prev_year_agg = df_total_actuals_agg.copy()
                        df_total_prev_year_agg.rename(columns={'y_total_actual': 'y_total_prev_year'}, inplace=True)
                        df_total_prev_year_agg['ds'] = df_total_prev_year_agg['ds'] + pd.DateOffset(years=1)

                    df_all_forecasts_concat = pd.concat(all_forecasts_list)
                    df_total_forecast_agg = df_all_forecasts_concat.groupby('ds')['yhat'].sum().reset_index().rename(columns={'yhat': 'yhat_total_forecast'})
                    df_total_forecast_agg.sort_values('ds', inplace=True)

                    df_general_plot = df_total_forecast_agg
                    if df_total_actuals_agg is not None:
                        df_general_plot = pd.merge(df_general_plot, df_total_actuals_agg, on='ds', how='outer')
                    if df_total_prev_year_agg is not None:
                        df_general_plot = pd.merge(df_general_plot, df_total_prev_year_agg, on='ds', how='outer')
                    
                    df_general_plot.sort_values('ds', inplace=True)
                    
                    if not df_general_plot.empty:
                        min_date_overall = df_general_plot['ds'].min()
                        max_date_overall = df_general_plot['ds'].max()
                        
                        date_range_overall = st.slider(
                            "Rango de fechas para gráfico general:",
                            min_value=min_date_overall.to_pydatetime(),
                            max_value=max_date_overall.to_pydatetime(),
                            value=(max(min_date_overall.to_pydatetime(), (max_date_overall - pd.DateOffset(years=3)).to_pydatetime()), max_date_overall.to_pydatetime()), 
                            format="DD/MM/YYYY",
                            key="date_range_slider_overview_tab" 
                        )
                        start_date_overall, end_date_overall = date_range_overall
                        df_general_plot_filtered = df_general_plot[
                            (df_general_plot['ds'] >= pd.to_datetime(start_date_overall)) &
                            (df_general_plot['ds'] <= pd.to_datetime(end_date_overall))
                        ]

                        fig_plotly_general = go.Figure()
                        if 'y_total_actual' in df_general_plot_filtered.columns:
                            fig_plotly_general.add_trace(go.Scatter(x=df_general_plot_filtered['ds'], y=df_general_plot_filtered['y_total_actual'], mode='lines+markers', name='Demanda Real Total', marker=dict(size=5), line=dict(color='navy')))
                        if 'yhat_total_forecast' in df_general_plot_filtered.columns:
                            fig_plotly_general.add_trace(go.Scatter(x=df_general_plot_filtered['ds'], y=df_general_plot_filtered['yhat_total_forecast'], mode='lines', name='Pronóstico Total (Suma yhat)', line=dict(color='crimson')))
                        if 'y_total_prev_year' in df_general_plot_filtered.columns:
                             fig_plotly_general.add_trace(go.Scatter(x=df_general_plot_filtered['ds'], y=df_general_plot_filtered['y_total_prev_year'], mode='lines+markers', name='Demanda Total Año Anterior', marker=dict(size=5), line=dict(color='forestgreen', dash='dot')))
                        
                        fig_plotly_general.update_layout(
                            title=dict(text='Visión General Agregada: Demanda Total', x=0.5, font=dict(size=18)),
                            xaxis_title_text='Fecha', yaxis_title_text='Demanda Total Agregada',
                            legend_title_text='Leyenda', xaxis_rangeslider_visible=True,
                            hovermode='x unified', height=600,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_plotly_general, use_container_width=True)
                    else:
                        st.info("No hay suficientes datos agregados para generar el gráfico general.")
                except Exception as e:
                    st.error(f"Ocurrió un error al generar el gráfico general agregado: {e}")
            
            st.markdown("---")
            st.subheader("Resumen de Métricas de Ajuste Histórico por Servicio")
            if all_metrics_list:
                df_summary_metrics = pd.DataFrame(all_metrics_list).sort_values(by='MAPE', na_position='last')
                df_summary_metrics_display = df_summary_metrics.copy()
                df_summary_metrics_display['MAPE'] = df_summary_metrics_display['MAPE'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) and x != float('inf') else "N/A")
                df_summary_metrics_display['MAE'] = df_summary_metrics_display['MAE'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                st.dataframe(df_summary_metrics_display.set_index('servicio'), use_container_width=True)
            else:
                st.info("No hay métricas para mostrar. Procese algunos servicios.")

            st.markdown("---")
            st.subheader("Resumen del Impacto Promedio de Feriados (Top por Servicio)")
            if all_holiday_impacts_list_global:
                df_all_holiday_impacts = pd.DataFrame(all_holiday_impacts_list_global)
                if not df_all_holiday_impacts.empty:
                    df_all_holiday_impacts['impacto_abs'] = df_all_holiday_impacts['impacto_promedio'].abs()
                    summary_holiday_impact_df = df_all_holiday_impacts.sort_values(
                        by=['servicio', 'impacto_abs'], ascending=[True, False]
                    ).groupby('servicio').head(5).drop(columns=['impacto_abs']) 
                    
                    summary_holiday_impact_df['impacto_promedio'] = summary_holiday_impact_df['impacto_promedio'].map('{:.2f}'.format)
                    st.dataframe(summary_holiday_impact_df.set_index(['servicio', 'feriado']), use_container_width=True)
                else:
                    st.info("No se encontró impacto de feriados para los servicios procesados o no hay feriados definidos.")
            else:
                st.info("No hay datos de impacto de feriados para mostrar.")

        # --- Pestaña: Análisis por Servicio ---
        with tab_service_detail:
            st.header("🔍 Análisis Detallado por Servicio")
            
            if not processed_services_forecasts:
                st.warning("No hay servicios procesados para mostrar detalles. Asegúrate de que el procesamiento haya finalizado.")
            else:
                service_keys = list(processed_services_forecasts.keys())
                selected_service_for_detail = st.selectbox(
                    "Selecciona un servicio para ver detalles:",
                    options=service_keys,
                    key="service_detail_selectbox"
                )

                if selected_service_for_detail and selected_service_for_detail in processed_services_forecasts:
                    service_data = processed_services_forecasts[selected_service_for_detail]
                    model = service_data['model']
                    forecast_df = service_data['forecast_df']
                    metrics = service_data['metrics']
                    fig_components_service = service_data['fig_components']
                    df_prophet_train_service = service_data['train_data']
                    holiday_impacts_df_service_detail = service_data['holiday_impacts_df']

                    st.subheader(f"Detalles para: {selected_service_for_detail}")
                    
                    col1_metrics, col2_metrics = st.columns(2)
                    mae_val = metrics.get('MAE', float('nan'))
                    mape_val = metrics.get('MAPE', float('nan'))

                    col1_metrics.metric(label="MAE", value=f"{mae_val:.2f}" if pd.notnull(mae_val) else "N/A")
                    col2_metrics.metric(label="MAPE", value=f"{mape_val:.2%}" if pd.notnull(mape_val) and mape_val != float('inf') else "N/A")


                    st.markdown("---")
                    st.subheader("Comparación Anual y Pronóstico")
                    
                    try:
                        df_prev_year_demand = df_prophet_train_service[['ds', 'y']].copy()
                        df_prev_year_demand.rename(columns={'y': 'y_prev_year'}, inplace=True)
                        df_prev_year_demand['ds'] = df_prev_year_demand['ds'] + pd.DateOffset(years=1)
                        
                        forecast_with_prev_year = pd.merge(forecast_df, df_prev_year_demand, on='ds', how='left')

                        min_date_service = forecast_with_prev_year['ds'].min()
                        max_date_service = forecast_with_prev_year['ds'].max()
                        
                        slider_key_service_detail = f"date_range_slider_service_{selected_service_for_detail.replace(' ', '_')}"
                        date_range_service = st.slider(
                            f"Rango de fechas para '{selected_service_for_detail}':",
                            min_value=min_date_service.to_pydatetime(),
                            max_value=max_date_service.to_pydatetime(),
                            value=(max(min_date_service.to_pydatetime(), (max_date_service - pd.DateOffset(years=3)).to_pydatetime()), max_date_service.to_pydatetime()),
                            format="DD/MM/YYYY",
                            key=slider_key_service_detail 
                        )
                        start_date_service, end_date_service = date_range_service
                        plot_data_annual_comp = forecast_with_prev_year[
                            (forecast_with_prev_year['ds'] >= pd.to_datetime(start_date_service)) &
                            (forecast_with_prev_year['ds'] <= pd.to_datetime(end_date_service))
                        ]

                        fig_plotly_annual = go.Figure()
                        fig_plotly_annual.add_trace(go.Scatter(x=plot_data_annual_comp['ds'], y=plot_data_annual_comp['y'], mode='lines+markers', name='Demanda Actual (y)', marker=dict(size=5), line=dict(color='blue')))
                        fig_plotly_annual.add_trace(go.Scatter(x=plot_data_annual_comp['ds'], y=plot_data_annual_comp['yhat'], mode='lines', name='Pronóstico (yhat)', line=dict(color='red')))
                        fig_plotly_annual.add_trace(go.Scatter(x=plot_data_annual_comp['ds'], y=plot_data_annual_comp['yhat_upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False, name='Intervalo Sup.'))
                        fig_plotly_annual.add_trace(go.Scatter(x=plot_data_annual_comp['ds'], y=plot_data_annual_comp['yhat_lower'], mode='lines', line=dict(width=0), fillcolor='rgba(255,0,0,0.15)', fill='tonexty', hoverinfo='skip', showlegend=False, name='Intervalo Inf.'))
                        if 'y_prev_year' in plot_data_annual_comp.columns:
                             fig_plotly_annual.add_trace(go.Scatter(x=plot_data_annual_comp['ds'], y=plot_data_annual_comp['y_prev_year'], mode='lines+markers', name='Demanda Año Anterior', marker=dict(size=5, symbol='diamond-open'), line=dict(dash='dot', color='green')))
                        
                        fig_plotly_annual.update_layout(
                            title=dict(text=f'Comparación Anual: {selected_service_for_detail}', x=0.5, font=dict(size=16)),
                            xaxis_title_text='Fecha', yaxis_title_text='Demanda', legend_title_text='Leyenda',
                            xaxis_rangeslider_visible=True, hovermode='x unified', height=500,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_plotly_annual, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generando gráfico de comparación anual para '{selected_service_for_detail}': {e}")

                    st.markdown("---")
                    st.subheader("Componentes del Modelo Prophet")
                    if fig_components_service:
                        fig_components_service.set_size_inches(10, max(6, len(fig_components_service.axes) * 2)) 
                        st.pyplot(fig_components_service)
                        plt.close(fig_components_service) 
                    else:
                        st.info("No se generaron los componentes del modelo (posiblemente debido a un error previo o no se pudieron generar).")
                    
                    st.markdown("---")
                    st.subheader(f"Análisis de Impacto de Feriados para: {selected_service_for_detail}")
                    if not df_feriados_eventos_processed.empty:
                        if not holiday_impacts_df_service_detail.empty:
                            st.write("Impacto Promedio Estimado de Feriados (del modelo):")
                            st.dataframe(holiday_impacts_df_service_detail.set_index('feriado'), use_container_width=True)
                        else:
                            st.info(f"El modelo no estimó un impacto significativo para los feriados definidos en '{selected_service_for_detail}', o no hay feriados en el periodo de datos.")

                        with st.expander("Ver gráficos de demanda alrededor de cada feriado"):
                            num_feriados_plot_key = f"num_feriados_slider_{selected_service_for_detail.replace(' ', '_')}"
                            num_feriados_plot = st.slider("Máximo de feriados a graficar:", 1, len(df_feriados_eventos_processed), min(5, len(df_feriados_eventos_processed)), key=num_feriados_plot_key)
                            
                            for idx, feriado_row in df_feriados_eventos_processed.head(num_feriados_plot).iterrows():
                                holiday_name = str(feriado_row['holiday'])
                                holiday_date = feriado_row['ds']
                                lower_w = int(feriado_row['lower_window'])
                                upper_w = int(feriado_row['upper_window'])

                                analysis_start = holiday_date - pd.Timedelta(days=10)
                                analysis_end = holiday_date + pd.Timedelta(days=10)
                                holiday_view_df = forecast_df[(forecast_df['ds'] >= analysis_start) & (forecast_df['ds'] <= analysis_end)].copy()

                                if holiday_view_df.empty: continue

                                fig_h = go.Figure()
                                if 'y' in holiday_view_df.columns:
                                    fig_h.add_trace(go.Scatter(x=holiday_view_df['ds'], y=holiday_view_df['y'], mode='lines+markers', name='Real', line=dict(color='blue')))
                                fig_h.add_trace(go.Scatter(x=holiday_view_df['ds'], y=holiday_view_df['yhat'], mode='lines+markers', name='Pronóstico', line=dict(color='red', dash='dash')))
                                fig_h.add_trace(go.Scatter(x=holiday_view_df['ds'], y=holiday_view_df['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
                                fig_h.add_trace(go.Scatter(x=holiday_view_df['ds'], y=holiday_view_df['yhat_lower'], mode='lines', line=dict(width=0), fillcolor='rgba(255,0,0,0.1)', fill='tonexty', showlegend=False))
                                
                                fig_h.add_vline(x=holiday_date, line_width=2, line_dash="dash", line_color="green", annotation_text=f"{holiday_name}", annotation_position="top right")
                                effective_start_h = holiday_date + pd.Timedelta(days=lower_w); effective_end_h = holiday_date + pd.Timedelta(days=upper_w)
                                if effective_start_h < effective_end_h: 
                                     fig_h.add_vrect(x0=effective_start_h, x1=effective_end_h, fillcolor="orange", opacity=0.2, layer="below", line_width=0, annotation_text=f"Efecto Modelo ({lower_w}d a +{upper_w}d)")

                                fig_h.update_layout(title=f"Demanda alrededor de '{holiday_name}' para {selected_service_for_detail}", xaxis_title="Fecha", yaxis_title="Demanda", height=400, margin=dict(t=50, b=50))
                                st.plotly_chart(fig_h, use_container_width=True)
                    else:
                        st.info(f"No hay datos de feriados cargados para analizar el impacto en '{selected_service_for_detail}'.")

        # --- Pestaña: Explorador de Datos ---
        with tab_data_explorer:
            st.header("📄 Explorador de Datos Cargados")
            
            st.subheader("Datos de Demanda (Primeras 1000 filas)")
            if df_demanda_full is not None:
                st.dataframe(df_demanda_full.head(1000), height=300, use_container_width=True)
                display_df_info(df_demanda_full, "Datos de Demanda")
                if st.checkbox("Mostrar descripción estadística (demanda)", key="desc_demanda_chk"):
                    st.dataframe(df_demanda_full.describe(include='all'))
            else:
                st.info("No se han cargado datos de demanda.")

            st.subheader("Datos de Feriados (Procesados)")
            if df_feriados_eventos_processed is not None and not df_feriados_eventos_processed.empty:
                st.dataframe(df_feriados_eventos_processed.head(100), height=300, use_container_width=True)
                display_df_info(df_feriados_eventos_processed, "Datos de Feriados Procesados")
            else:
                st.info("No se han cargado o procesado datos de feriados.")
            
            # --- MODIFICACIÓN AQUÍ ---
            st.markdown("---") # Separador visual
            if processed_services_forecasts:
                # Mostrar el filtro (selectbox) primero
                service_to_show_fcst_key = "fcst_table_select_explorer"
                service_to_show_fcst = st.selectbox(
                    "Selecciona un servicio para ver su tabla de pronóstico:",
                    options=list(processed_services_forecasts.keys()),
                    key=service_to_show_fcst_key
                )

                # Luego el subencabezado y la tabla si se selecciona un servicio
                if service_to_show_fcst:
                    st.subheader(f"Tabla de Pronóstico para: {service_to_show_fcst} (Primeras 1000 filas)")
                    fcst_df_to_show = processed_services_forecasts[service_to_show_fcst]['forecast_df']
                    st.dataframe(fcst_df_to_show.head(1000), height=300, use_container_width=True)
                    
                    desc_fcst_key = f"desc_fcst_chk_{service_to_show_fcst.replace(' ', '_')}"
                    if st.checkbox(f"Mostrar descripción estadística del pronóstico para {service_to_show_fcst}", key=desc_fcst_key):
                        st.dataframe(fcst_df_to_show.describe(include='all'))
                else:
                    # Si hay pronósticos procesados pero no se selecciona ninguno (poco probable con selectbox a menos que las opciones estén vacías)
                    st.subheader("Pronósticos Generados")
                    st.info("Selecciona un servicio de la lista de arriba para ver su tabla de pronóstico.")
            
            else: # Si no hay pronósticos procesados en absoluto
                st.subheader("Pronósticos Generados")
                st.info("No se han generado pronósticos aún. Completa los pasos anteriores.")
            # --- FIN DE LA MODIFICACIÓN ---

    else: 
        st.info("Por favor, selecciona al menos un tipo de servicio en la barra lateral para comenzar el análisis.")
else: 
    st.info("📈 Por favor, carga el archivo de datos de demanda en la barra lateral para activar el panel.")

st.sidebar.markdown("---")
st.sidebar.info("Panel desarrollado para pronóstico de series temporales.")
