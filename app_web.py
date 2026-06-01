import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

# ==========================================
# --- 1. CONFIGURACIÓN DE LA PÁGINA (UI) ---
# ==========================================
st.set_page_config(
    page_title="EcoSmart ITGAM - WasteAI",
    page_icon="♻️",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Estilo CSS para que todo quepa en una sola pantalla sin scroll
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        h1 { margin-bottom: 0.2rem; font-size: 2rem !important; }
        h3 { margin-top: 0rem; font-size: 1.2rem !important; }
        .stMetric { padding: 0.5rem; background-color: #f0f2f6; border-radius: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

st.title("♻️ Clasificador EcoSmart — ITGAM WasteAI")

# ==========================================
# --- 2. INICIALIZACIÓN DE CONTADORES ------
# ==========================================
# Creamos las variables en la memoria de Streamlit para que no se reinicien a cero
if 'contador_comida' not in st.session_state: st.session_state.contador_comida = 0
if 'contador_cafe' not in st.session_state: st.session_state.contador_cafe = 0
if 'contador_plastico' not in st.session_state: st.session_state.contador_plastico = 0
if 'contador_carton' not in st.session_state: st.session_state.contador_carton = 0

clases_finales = ['Organico_Comida', 'Organico_Cafe', 'Plastico', 'Carton_Papel']

# ==========================================
# --- 3. CARGA DEL CEREBRO DE LA IA -------
# ==========================================
@st.cache_resource
def cargar_modelo():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)
    
    ruta_pesos = 'modelo_residuos (4).pth'
    if not os.path.exists(ruta_pesos):
        ruta_pesos = os.path.join(os.path.dirname(__file__), 'modelo_residuos (4).pth')

    if not os.path.exists(ruta_pesos):
        st.error("🚨 Error: No se encontró 'modelo_residuos (4).pth'.")
        return None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ruta_pesos, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model_data = cargar_modelo()

transformacion_ui = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================================
# --- 4. INTERFAZ LADO A LADO (SIN SCROLL) -
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Cámara de Reconocimiento")
    # Activamos el componente nativo de la cámara web
    foto_camara = st.camera_input("Enfoque el residuo frente a la cámara:")

with col2:
    st.subheader("🧠 Diagnóstico de la Inteligencia Artificial")
    
    if foto_camara is not None and model_data is not None:
        model, device = model_data
        
        # Procesar los bits de la cámara
        imagen = Image.open(foto_camara).convert('RGB')
        img_t = transformacion_ui(imagen).unsqueeze(0).to(device)
        
        # Inferencia
        with torch.no_grad():
            outputs = model(img_t)
            probabilities = F.softmax(outputs, dim=1)
            confianza, predicted = torch.max(probabilities, 1)
            
        clase_detectada = clases_finales[predicted.item()]
        porcentaje = confianza.item() * 100
        
        # Despliegue visual del veredicto
        st.markdown(f"### 📍 Predicción: **`{clase_detectada}`**")
        st.progress(confianza.item())
        st.metric(label="Porcentaje de Certeza Técnica", value=f"{porcentaje:.2f}%")
        
        # --- LÓGICA DEL BOTÓN INTELIGENTE (> 70%) ---
        if porcentaje >= 70.0:
            st.success(f"🎯 ¡Certeza alta ({porcentaje:.1f}%)! Puedes registrar este objeto.")
            
            # Botón para sumar al inventario escolar
            if st.button(f"📥 Registrar en Contador de {clase_detectada}"):
                if clase_detectada == 'Organico_Comida': st.session_state.contador_comida += 1
                elif clase_detectada == 'Organico_Cafe': st.session_state.contador_cafe += 1
                elif clase_detectada == 'Plastico': st.session_state.contador_plastico += 1
                elif clase_detectada == 'Carton_Papel': st.session_state.contador_carton += 1
                st.toast(f"¡+1 añadido a {clase_detectada}! 🚀", icon="✅")
        else:
            st.warning("⚠️ Confianza menor al 70%. Estabiliza el objeto frente a la cámara para poder registrarlo.")
            
        # --- DESPLIEGUE TABLA DE MÉTRICAS / CONTADORES ---
        st.markdown("---")
        st.markdown("#### 📊 Inventario Escolar de Residuos Clasificados:")
        
        # Mostramos los contadores lado a lado de forma muy visual
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🍏 Comida", st.session_state.contador_comida)
        m2.metric("☕ Café", st.session_state.contador_cafe)
        m3.metric("🍼 Plástico", st.session_state.contador_plastico)
        m4.metric("📦 Cartón", st.session_state.contador_carton)
            
    else:
        st.info("💡 Active la cámara en la izquierda para iniciar el análisis en tiempo real.")