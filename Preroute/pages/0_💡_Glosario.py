# pages/0_💡_Glosario.py

import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Directorio de Recursos",
    page_icon="💡",
    layout="wide"
)

# --- DATOS (sin cambios) ---
CATEGORIES = [
    "All", "Analytics", "Marketing", "Productivity", "Sales", "Finance",
    "Communication", "Cloud Services", "Security", "Design", "Development",
    "Human Resources", "Customer Support", "E-commerce", "Social Media",
]
ICON_MAP = {
    "Zap": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>""",
    "ShoppingCart": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="21" r="1"></circle><circle cx="20" cy="21" r="1"></circle><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path></svg>""",
    "BarChart": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg>""",
    "Mail": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>""",
    "Calendar": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>""",
    "Database": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>""",
    "Cloud": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"></path></svg>""",
    "Shield": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>""",
}

@st.cache_data(ttl=60)
def load_data_from_sheet():
    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes
        )
        client = gspread.authorize(creds)
        spreadsheet = client.open("Glosario")
        worksheet = spreadsheet.sheet1
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        df = df.fillna('') 
        return df.to_dict('records')
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Error: No se encontró la hoja de cálculo 'Glosario'. Revisa el nombre y que la hayas compartido.")
        return []
    except Exception as e:
        st.error(f"Ocurrió un error al conectar con Google Sheets: {e}")
        return []

st.markdown("""
<style>
    @keyframes gradient_flow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #ff8c00, #000000, #e45826, #111111);
        background-size: 400% 400%;
        animation: gradient_flow 15s ease infinite;
    }
    .main .block-container { color: #e0e0e0; }
    .stRadio > div { flex-direction: row; flex-wrap: wrap; gap: 10px; }
    .stRadio > div > label {
        background-color: rgba(255, 255, 255, 0.1); border-radius: 20px;
        padding: 5px 15px; transition: all 0.2s ease-in-out;
        border: 1px solid rgba(255, 255, 255, 0.2); color: #d0d0d0;
    }
    .stRadio > div > label[data-baseweb="radio"]:has(> div > input:checked) {
        background-color: #ff8c00; color: black; border-color: #ff8c00; font-weight: 600;
    }
    .integration-card {
        background: rgba(20, 20, 20, 0.7); backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px); border-radius: 10px; padding: 20px;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        height: 100%; display: flex; flex-direction: column;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .integration-card:hover {
        transform: translateY(-5px); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-color: rgba(255, 140, 0, 0.5);
    }
    .icon-container {
        width: 50px; height: 50px; border-radius: 8px; display: flex;
        align-items: center; justify-content: center; margin-bottom: 15px;
    }
    .icon-container svg { stroke: white; width: 28px; height: 28px; }
    .integration-card h3 { margin: 0 0 10px 0; color: #ffffff; font-size: 1.1rem; }
    .integration-card p {
        color: #b0b0b0; font-size: 0.9rem; flex-grow: 1; margin-bottom: 15px;
    }
    .resource-expert {
        font-size: 0.85rem; color: #b0b0b0; margin-top: auto;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 10px; margin-top: 15px;
    }
    .action-button {
        display: block; text-align: center; padding: 10px; border-radius: 5px;
        text-decoration: none; color: white; font-weight: 600;
        margin-top: 15px; transition: filter 0.2s ease;
    }
    .action-button:hover { filter: brightness(1.2); }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: white;'>Directorio de Recursos y Herramientas</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #d0d0d0;'>Encuentra la herramienta que necesitas. Filtra por categoría y accede directamente o contacta al experto.</p>", unsafe_allow_html=True)
st.write("---")

if 'integrations_data' not in st.session_state:
    st.session_state.integrations_data = load_data_from_sheet()

all_integrations = st.session_state.integrations_data

selected_category = st.radio(
    "Filtrar por categoría:", CATEGORIES, horizontal=True, label_visibility="collapsed"
)

if selected_category == "All":
    filtered_integrations = all_integrations
else:
    filtered_integrations = [
        i for i in all_integrations if i.get("category") == selected_category
    ]

if not all_integrations:
    st.info("Cargando datos o la hoja de cálculo está vacía...")
elif not filtered_integrations:
    st.warning(f"No se encontraron recursos en la categoría '{selected_category}'.")
else:
    num_columns = 3
    cols = st.columns(num_columns)
    
    for i, integration in enumerate(filtered_integrations):
        col = cols[i % num_columns]
        with col:
            card_html = f"""
            <div class="integration-card">
                <div class="icon-container" style="background-color: {integration.get('color', '#607D8B')};">
                    {ICON_MAP.get(integration.get('icon', 'Shield'), ICON_MAP['Shield'])}
                </div>
                <h3>{integration.get('name', 'Sin Nombre')}</h3>
                <p>{integration.get('description', 'Sin Descripción')}</p>
                <div class="resource-expert">
                    <strong>Experto:</strong> {integration.get('expert', 'No asignado')}
                </div>
                <a href="{integration.get('action_url', '#')}" target="_blank" class="action-button" style="background-color: {integration.get('color', '#607D8B')};">
                    {integration.get('action_text', 'Acceder')}
                </a>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            st.write("")
