# Home.py
import streamlit as st
import bcrypt

# 1. st.set_page_config() es el PRIMER comando de Streamlit
st.set_page_config(page_title="Inicio de Sesión", page_icon="🔑", layout="centered")

# --- LÓGICA DE AUTENTICACIÓN ---
def login(username, password):
    if "users" in st.secrets and st.secrets.get("users"):
        if username in st.secrets["users"]:
            user_data = st.secrets["users"][username]
            if "password" in user_data:
                stored_password_hash = user_data["password"]
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
    st.title("Transvip")
    st.write("Por favor, inicia sesión para continuar.")

    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Iniciar Sesión")

        if submitted:
            if login(username, password):
                st.session_state['authenticated'] = True
                st.session_state['user_role'] = st.secrets["users"][username]["role"]
                st.info("DIAGNÓSTICO (Home.py): Login exitoso, st.rerun() será llamado.")
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

# --- LÓGICA POST-LOGIN Y REDIRECCIÓN ---
if st.session_state['authenticated']:
    st.info(f"DIAGNÓSTICO (Home.py): Usuario autenticado. Rol: '{st.session_state.get('user_role', 'Desconocido')}'")

    PAGES_CONFIG = {
        "1_🚀_Pre-Route": {"path": "pages/1_🚀_Pre-Route.py"},
        "2_📊_Dashboard": {"path": "pages/2_📊_Dashboard.py"}
    }
    
    user_role = st.session_state.get('user_role')
    allowed_pages_keys = []
    if "roles" in st.secrets and user_role in st.secrets.get("roles", {}):
        allowed_pages_keys = st.secrets["roles"][user_role]
    
    st.info(f"DIAGNÓSTICO (Home.py): Claves de páginas permitidas para '{user_role}': {allowed_pages_keys}")

    if allowed_pages_keys:
        first_page_key_to_redirect = allowed_pages_keys[0]
        st.info(f"DIAGNÓSTICO (Home.py): Primera clave de página para redirigir: '{first_page_key_to_redirect}'")

        if first_page_key_to_redirect in PAGES_CONFIG:
            path_to_switch = PAGES_CONFIG[first_page_key_to_redirect]["path"]
            st.success(f"DIAGNÓSTICO (Home.py): Intentando REDIRIGIR a: '{path_to_switch}'...")
            try:
                st.switch_page(path_to_switch)
                # Si st.switch_page tiene éxito, el script se detiene aquí y carga la nueva página.
                # El siguiente st.info no debería aparecer si la redirección es exitosa.
                st.info("DIAGNÓSTICO (Home.py): st.switch_page() fue llamado.")
            except Exception as e_switch:
                st.error(f"DIAGNÓSTICO (Home.py): FALLÓ la redirección a '{path_to_switch}'. Error: {e_switch}")
        else:
            st.error(f"DIAGNÓSTICO (Home.py): Error de Configuración - La clave de página '{first_page_key_to_redirect}' (del rol '{user_role}') no está definida en PAGES_CONFIG en Home.py.")
            st.warning("Contacte al administrador.")
    else:
        st.warning(f"DIAGNÓSTICO (Home.py): El rol '{user_role}' no tiene páginas asignadas en secrets.toml o el rol no está definido.")
        st.write("Contacte al administrador.")

    # Si la redirección falla o no hay páginas, mostramos el botón de logout
    # La barra lateral con las páginas la generará Streamlit automáticamente si estamos en una página de la carpeta 'pages/'
    # Aquí, en Home.py, solo nos preocupamos por el logout si la redirección no ocurre.
    if st.sidebar.button("Cerrar Sesión", key="logout_home_authenticated_fallback"):
        for key_session in st.session_state.keys():
            del st.session_state[key_session]
        st.rerun()
