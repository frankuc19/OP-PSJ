# pages/0_💡_Glosario.py

import streamlit as st
import random

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Navega en las pestañas de la app",
    page_icon="💡",
    layout="wide"
)

# --- DATOS (sin cambios) ---

# 1. Categorías
CATEGORIES = [
    "All", "Analytics", "Marketing", "Productivity", "Sales", "Finance",
    "Communication", "Cloud Services", "Security", "Design", "Development",
    "Human Resources", "Customer Support", "E-commerce", "Social Media",
]

# 2. Paleta de Colores
COLOR_PALETTE = [
    "#FF4A00", "#96BF48", "#E37400", "#FFE01B", "#F06A6A", "#FFCC22",
    "#6772E5", "#F22F46", "#2D8CFF", "#0061FF", "#00A1E0", "#D32D27",
    "#4CAF50", "#9C27B0", "#FF9800", "#795548", "#607D8B", "#3F51B5",
    "#00BCD4", "#FFC107",
]

# 3. Mapa de Íconos (SVGs de Feather Icons, equivalentes a Lucide)
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

# --- FUNCIÓN PARA GENERAR DATOS ---
@st.cache_data
def generate_integrations(count: int):
    integrations = []
    icon_keys = list(ICON_MAP.keys())
    for i in range(count):
        category = random.choice(CATEGORIES[1:])
        integrations.append({
            "id": f"{i + 1}", "name": f"Integration {i + 1}",
            "description": f"This is a detailed description for Integration #{i+1}. It provides {category.lower()} services to streamline your workflow.",
            "category": category, "icon": random.choice(icon_keys), "color": random.choice(COLOR_PALETTE),
        })
    return integrations

# --- CSS PARA EL DISEÑO DE LA PÁGINA (TEMA OSCURO) ---
st.markdown("""
<style>
    /* Fondo animado oscuro */
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
    
    /* Color de texto principal */
    .main .block-container {
        color: #e0e0e0;
    }

    /* Estilo del contenedor de filtros */
    .stRadio > div {
        flex-direction: row;
        flex-wrap: wrap;
        gap: 10px;
    }
    .stRadio > div > label {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 5px 15px;
        transition: all 0.2s ease-in-out;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #d0d0d0; /* Letras de los filtros en gris claro */
    }
    /* Estilo del filtro seleccionado (naranja) */
    .stRadio > div > label[data-baseweb="radio"]:has(> div > input:checked) {
        background-color: #ff8c00;
        color: black; /* Letras negras para contraste con el fondo naranja */
        border-color: #ff8c00;
        font-weight: 600;
    }
    
    /* Diseño de la tarjeta de integración (efecto vidrio) */
    .integration-card {
        background: rgba(20, 20, 20, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 20px;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        height: 100%;
        display: flex;
        flex-direction: column;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .integration-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-color: rgba(255, 140, 0, 0.5);
    }
    .icon-container {
        width: 50px;
        height: 50px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 15px;
    }
    .icon-container svg {
        stroke: white;
        width: 28px;
        height: 28px;
    }
    .integration-card h3 {
        margin: 0 0 10px 0;
        color: #ffffff;
        font-size: 1.1rem;
    }
    .integration-card p {
        color: #b0b0b0;
        font-size: 0.9rem;
        flex-grow: 1;
        margin-bottom: 15px;
    }
    .category-tag {
        align-self: flex-start;
        background-color: rgba(255, 255, 255, 0.1);
        color: #d0d0d0;
        border-radius: 5px;
        padding: 4px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- RENDERIZADO DE LA PÁGINA ---

# Título y descripción con Markdown para asegurar el color blanco
st.markdown("<h1 style='color: white;'>Contenedores de Información</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #d0d0d0;'>Información que podras consultar buscando por contenedor, solo selecciona uno y vera información de relevancia</p>", unsafe_allow_html=True)
st.write("---")

# Generar y cachear los datos
if 'integrations_data' not in st.session_state:
    st.session_state.integrations_data = generate_integrations(50)

all_integrations = st.session_state.integrations_data

# 1. Filtro de Categorías
selected_category = st.radio(
    "Filter by category:",
    CATEGORIES,
    horizontal=True,
    label_visibility="collapsed"
)

# 2. Lógica de Filtrado
if selected_category == "All":
    filtered_integrations = all_integrations
else:
    filtered_integrations = [
        i for i in all_integrations if i["category"] == selected_category
    ]

if not filtered_integrations:
    st.warning(f"No integrations found in the category '{selected_category}'.")
else:
    # 3. Cuadrícula de Integraciones
    num_columns = 3
    cols = st.columns(num_columns)
    
    for i, integration in enumerate(filtered_integrations):
        col = cols[i % num_columns]
        with col:
            card_html = f"""
            <div class="integration-card">
                <div class="icon-container" style="background-color: {integration['color']};">
                    {ICON_MAP[integration['icon']]}
                </div>
                <h3>{integration['name']}</h3>
                <p>{integration['description']}</p>
                <div class="category-tag">{integration['category']}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            st.write("")