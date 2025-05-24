# Home.py
import streamlit as st
import bcrypt

# --- CONFIGURACIÓN DE PÁGINA ---
# Es importante que st.set_page_config sea el primer comando de Streamlit en Home.py también
st.set_page_config(page_title="Transvip - Inicio de Sesión", page_icon="🔑", layout="wide")

# --- LÓGICA DE AUTENTICACIÓN (sin cambios) ---
def login(username, password):
    """Verifica si el usuario y contraseña son válidos."""
    if "users" in st.secrets and st.secrets.get("users"):
        if username in st.secrets["users"]:
            user_data = st.secrets["users"][username]
            if "password" in user_data:
                stored_password_hash = user_data["password"]
                if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
                    return True
    return False

# --- INICIALIZAR ESTADO DE SESIÓN (sin cambios) ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None

# --- PÁGINA DE LOGIN (MODIFICADA CON DISEÑO) ---
if not st.session_state['authenticated']:
    # Opcional: Añadir un logo
    # Intenta centrar el logo si es posible
    # Asegúrate de que la ruta al logo sea correcta
    # Ejemplo:
    # cols_logo_login = st.columns([1,1,1])
    # with cols_logo_login[1]:
    #     try:
    #         st.image("Preroute/transvip.png", width=150) # Ajusta la ruta y el ancho
    #     except Exception as e:
    #         st.caption(f"Logo no encontrado o error al cargar: {e}")
    # st.write("") # Espacio

    # Título y subtítulo con estilo naranja y centrado
    st.markdown("<h1 style='text-align: center; color: #FF6347;'>Transvip</h1>", unsafe_allow_html=True) # #FF6347 es un tono de naranja (Tomato)
    st.markdown("<p style='text-align: center; color: #FFA500;'>Por favor, inicia sesión para continuar.</p>", unsafe_allow_html=True) # #FFA500 es otro tono de naranja
    st.write("") # Añade un poco de espacio vertical

    # Usar columnas para centrar el formulario y darle un ancho más agradable
    col1, col_form, col2 = st.columns([0.5, 1.5, 0.5]) # Ajusta los ratios según necesites

    with col_form:
        # Opcional: Usar un contenedor con borde si tu versión de Streamlit lo soporta
        # with st.container(border=True):
        with st.form("login_form_styled"):
            st.markdown("### Ingresa tus Credenciales")
            username = st.text_input("👤 Usuario", placeholder="Tu nombre de usuario")
            password = st.text_input("🔒 Contraseña", type="password", placeholder="Tu contraseña")
            st.write("") # Espacio antes del botón
            submitted = st.form_submit_button("▶️ Iniciar Sesión", use_container_width=True, type="primary")

            if submitted:
                if login(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['user_role'] = st.secrets["users"][username]["role"]
                    # st.info("DIAGNÓSTICO (Home.py): Login exitoso, st.rerun() será llamado.") # Mantén esto comentado si ya funciona bien
                    st.rerun()
                else:
                    st.error("❌ Usuario o contraseña incorrectos.")
    
    # Pie de página opcional
    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: small; color: grey;'>© 2024 Transvip. Todos los derechos reservados.</p>", unsafe_allow_html=True)


# --- LÓGICA POST-LOGIN Y REDIRECCIÓN (SIN CAMBIOS IMPORTANTES AQUÍ, ASEGÚRATE QUE ESTÉ CORRECTA) ---
if st.session_state['authenticated']:
    st.sidebar.success(f"Sesión iniciada como: **{st.session_state.get('user_role', 'Desconocido')}**")
    # st.sidebar.title("Navegación") # Streamlit genera esto automáticamente si hay páginas

    PAGES_CONFIG = {
        "1_🚀_Pre-Route": {"icon": "🚀", "path": "pages/1_🚀_Pre-Route.py", "title": "Pre-Route"},
        "2_📊_Dashboard": {"icon": "📊", "path": "pages/2_📊_Dashboard.py", "title": "Dashboard"}
    }
    user_role = st.session_state.get('user_role', 'guest')
    allowed_pages_keys = st.secrets.get("roles", {}).get(user_role, [])

    # Redirección automática a la primera página permitida
    if allowed_pages_keys:
        first_page_key_to_redirect = allowed_pages_keys[0]
        if first_page_key_to_redirect in PAGES_CONFIG:
            st.switch_page(PAGES_CONFIG[first_page_key_to_redirect]["path"])
        else:
            st.error(f"Error de Configuración: '{first_page_key_to_redirect}' no definida en PAGES_CONFIG.")
            if st.button("Cerrar Sesión por Error"): # Botón de logout si hay error de config
                for key in st.session_state.keys(): del st.session_state[key]
                st.rerun()
    else:
        st.warning(f"Rol '{user_role}' sin páginas asignadas.")
        if st.button("Cerrar Sesión por Permisos"): # Botón de logout si no hay páginas
            for key in st.session_state.keys(): del st.session_state[key]
            st.rerun()
