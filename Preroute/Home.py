# Archivo: Home.py
# Propuesta 2: Diseño Minimalista con Gradiente Animado (Tema Naranjo y Negro)

import streamlit as st
import bcrypt
import streamlit.components.v1 as components

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Transvip Login", page_icon="Preroute/transvip.png", layout="wide")

# --- CSS PARA OCULTAR LA BARRA LATERAL EN ESTA PÁGINA ---
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- LÓGICA DE AUTENTICACIÓN ---
def login(username, password):
    # ... (misma función de login)
    if "users" in st.secrets and st.secrets.get("users"):
        if username in st.secrets["users"]:
            user_data = st.secrets["users"][username]
            if "password" in user_data:
                stored_password_hash = user_data["password"]
                if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
                    return True
    return False

# --- ESTADO DE SESIÓN ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# --- ESTILOS CSS PARA EL DISEÑO DEL LOGIN (MODIFICADO) ---
minimalist_style = """
<style>
@keyframes gradient_flow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

[data-testid="stHeader"] { display: none; }
[data-testid="stAppViewContainer"] {
    /* CAMBIO DE COLOR: Se usan tonos de naranjo y negro */
    background: linear-gradient(-45deg, #ff8c00, #000000, #e45826, #111111);
    background-size: 400% 400%;
    animation: gradient_flow 15s ease infinite;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}
/* ... (el resto de tu CSS para el login-box no cambia) ... */
.login-box {
    background: rgba(0, 0, 0, 0.4); /* Un poco más oscuro para mejor contraste */
    padding: 3rem;
    border-radius: 20px;
    text-align: center;
}
.login-box h1 {
    color: #FFFFFF;
    font-size: 3rem;
    font-weight: 700;
}
input[type="text"], input[type="password"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid #999;
    color: white;
    font-size: 1.1rem;
    padding: 0.5rem 0;
    transition: border-color 0.3s;
}
input[type="text"]:focus, input[type="password"]:focus {
    /* CAMBIO DE COLOR: El borde ahora es naranjo para coincidir con el tema */
    border-bottom-color: #ff8c00;
    outline: none;
}
</style>
"""
st.markdown(minimalist_style, unsafe_allow_html=True)


# --- RENDERIZADO DE LA PÁGINA ---
if st.session_state.get('authenticated', False):
    st.switch_page("pages/1_🚀_Pre-Route.py")
else:
    # ... (el código de tu formulario de login no cambia) ...
    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.markdown("<h1>Transvip</h1>", unsafe_allow_html=True)
        with st.form("login_form_minimalist"):
            st.text_input("Usuario", placeholder=" Usuario", key="user_input", label_visibility="collapsed")
            st.text_input("Contraseña", type="password", placeholder=" Contraseña", key="pass_input", label_visibility="collapsed")
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Iniciar Sesión", use_container_width=True, type="primary")

            if submitted:
                if login(st.session_state.user_input, st.session_state.pass_input):
                    st.session_state['authenticated'] = True
                    st.switch_page("pages/1_🚀_Pre-Route.py")
                else:
                    st.error("❌ Credenciales inválidas")