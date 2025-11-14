import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="Detector de Objetos Urbanos",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .title-section {
        background: rgba(255,255,255,0.95);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    h1 {
        color: #667eea;
        font-weight: 700;
        margin: 0;
    }
    h3 {
        color: #764ba2;
    }
    </style>
""", unsafe_allow_html=True)

# Colores para diferentes tipos de objetos
COLORS = {
    "car": (0, 255, 0),           # Verde
    "person": (255, 0, 0),        # Rojo
    "traffic light": (0, 165, 255),  # Naranja
    "bus": (0, 255, 255),         # Amarillo
    "truck": (128, 0, 255),       # Magenta
}

# Mapeo de nombres de clases de COCO
CLASS_NAMES = {
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
    0: "person",
    9: "traffic light",
}

# Cargar modelo YOLO
@st.cache_resource
def load_model():
    """Cargar modelo YOLO11 con caché"""
    try:
        model = YOLO("yolo11n.pt")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def process_image(image, model):
    """Procesar imagen y detectar objetos"""
    # Convertir PIL Image a numpy array
    img_np = np.array(image)
    
    # Realizar predicción
    results = model.predict(img_np, conf=0.25, verbose=False)
    
    return results

def draw_detections(image, results):
    """Dibujar cajas de detección en la imagen"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    detections = {
        "car": 0,
        "person": 0,
        "traffic light": 0,
        "bus": 0,
        "truck": 0,
        "motorbike": 0,
    }
    
    # Procesar detecciones
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            # Obtener coordenadas
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = int(box.cls[0])
            
            # Obtener nombre de clase
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            if class_name in detections:
                detections[class_name] += 1
            
            # Seleccionar color
            color = COLORS.get(class_name, (255, 255, 255))
            
            # Dibujar rectángulo
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Dibujar etiqueta
            label = f"{class_name} {conf:.2f}"
            draw.text((x1, y1 - 10), label, fill=color, font=None)
    
    return img_draw, detections

# Título y descripción
st.markdown("""
    <div class="title-section">
        <h1>🚗 Detector de Objetos en Escenas Urbanas</h1>
        <p style="color: #666; font-size: 16px; margin: 10px 0 0 0;">
            Detecta automáticamente coches, peatones, semáforos y más en tus imágenes de calles.
        </p>
    </div>
""", unsafe_allow_html=True)

# Cargar modelo
model = load_model()

if model is None:
    st.error("No se pudo cargar el modelo. Verifica que ultralytics esté instalado correctamente.")
    st.stop()

# Sección de carga de imagen
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Cargar Imagen")
    uploaded_file = st.file_uploader(
        "Sube una imagen de una calle",
        type=["jpg", "jpeg", "png", "bmp"],
        help="La imagen será procesada para detectar objetos urbanos"
    )

with col2:
    st.subheader("⚙️ Configuración")
    confidence = st.slider("Confianza mínima", 0.1, 1.0, 0.25, 0.05)

# Procesar imagen
if uploaded_file is not None:
    # Cargar imagen
    image = Image.open(uploaded_file).convert("RGB")
    
    # Mostrar imagen original
    col_original, col_result = st.columns(2)
    
    with col_original:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True)
    
    # Procesamiento
    with st.spinner("🔍 Detectando objetos..."):
        results = model.predict(np.array(image), conf=confidence, verbose=False)
        image_result, detections = draw_detections(image, results)
    
    with col_result:
        st.subheader("Objetos Detectados")
        st.image(image_result, use_column_width=True)
    
    # Mostrar resumen de detecciones
    st.markdown("---")
    st.subheader("📊 Resumen de Detecciones")
    
    # Crear columnas para las métricas
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    metrics_data = [
        ("🚗 Carros", detections["car"], col1),
        ("👥 Peatones", detections["person"], col2),
        ("🚦 Semáforos", detections["traffic light"], col3),
        ("🚌 Buses", detections["bus"], col4),
        ("🚚 Camiones", detections["truck"], col5),
        ("🏍️ Motos", detections["motorbike"], col6),
    ]
    
    for label, count, col in metrics_data:
        with col:
            st.metric(label, count)
    
    # Resumen en texto
    total = sum(detections.values())
    detected_items = ", ".join([
        f"{count} {label.lower()}" 
        for label, count in detections.items() 
        if count > 0
    ])
    
    if detected_items:
        summary = f"**Resumen:** Detectados {total} objetos - {detected_items}"
    else:
        summary = "**Resumen:** No se detectaron objetos en la imagen"
    
    st.info(summary)
    
    # Detalles técnicos
    with st.expander("ℹ️ Detalles Técnicos"):
        col_tech1, col_tech2 = st.columns(2)
        with col_tech1:
            st.write(f"**Modelo:** YOLO11n")
            st.write(f"**Confianza:** {confidence}")
        with col_tech2:
            st.write(f"**Tamaño de imagen:** {image.size[0]}x{image.size[1]}")
            st.write(f"**Total detectado:** {total} objetos")

else:
    # Mensaje inicial
    st.markdown("""
        <div style="text-align: center; padding: 50px 20px; background: rgba(255,255,255,0.1); border-radius: 10px; color: white;">
            <h3>👆 Comienza subiendo una imagen de una calle</h3>
            <p>Soporta formatos: JPG, JPEG, PNG, BMP</p>
            <p>La app detectará automáticamente:</p>
            <ul style="list-style: none; padding: 20px 0;">
                <li>🚗 Coches y vehículos</li>
                <li>👥 Peatones</li>
                <li>🚦 Semáforos</li>
                <li>🚌 Buses</li>
                <li>🚚 Camiones</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: white; padding: 20px;">
        <p>Powered by YOLO11 • Detección de objetos en tiempo real</p>
    </div>
    """,
    unsafe_allow_html=True
)