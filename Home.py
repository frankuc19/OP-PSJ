# Home.py
import streamlit as st
import bcrypt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Inicio de Sesión", page_icon="🔑", layout="centered")

# --- LÓGICA DE AUTENTICACIÓN ---
def login(username, password):
    """Verifica si el usuario y contraseña son válidos."""
    if username in st.secrets["users"]:
        stored_password_hash = st.secrets["users"][username]["password"]
        # Verifica la contraseña ingresada contra el hash almacenado
        if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            return True
    return False

# --- INICIALIZAR ESTADO DE SESIÓN ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None

# --- PÁGINA DE LOGIN ---
if not st.session_state['authenticated']:
    st.title("Bienvenido al Sistema de Análisis")
    st.write("Por favor, inicia sesión para continuar.")

    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    if st.button("Iniciar Sesión"):
        if login(username, password):
            st.session_state['authenticated'] = True
            st.session_state['user_role'] = st.secrets["users"][username]["role"]
            st.rerun() # Re-ejecuta el script para reflejar el estado de autenticado
        else:
            st.error("Usuario o contraseña incorrectos.")

# --- CONTENIDO POST-LOGIN Y NAVEGACIÓN ---
if st.session_state['authenticated']:
    st.sidebar.success(f"Sesión iniciada como: **{st.session_state['user_role']}**")
    
    # Obtener las páginas permitidas para el rol del usuario desde secrets.toml
    allowed_pages = st.secrets["roles"][st.session_state['user_role']]
    
    # Crear los objetos de página para la navegación
    pages_to_show = [
        st.Page("pages/1_Dashboard_General.py", title="Dashboard General", icon="📊"),
        st.Page("pages/2_Panel_Administrador.py", title="Panel Administrador", icon="⚙️"),
    ]
    
    # Filtrar las páginas basadas en los permisos del rol
    pg = st.navigation([page for page in pages_to_show if page.title in allowed_pages])
    
    st.title("Página Principal")
    st.write("¡Has iniciado sesión exitosamente!")
    st.write("Usa la barra lateral para navegar a las páginas a las que tienes acceso.")
    
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state['authenticated'] = False
        st.session_state['user_role'] = None
        st.rerun()
else:
    # Opcional: puedes dejar esto vacío o mostrar algo más
    pass