import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import io

# <CHANGE> Reemplazar cv2 con PIL para evitar errores de libGL
# PIL no requiere dependencias gráficas del sistema

# Configurar página de Streamlit
st.set_page_config(
    page_title="Detección de Objetos Urbanos",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            color: white;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("<h1>🚗 Detección de Objetos en Escenas Urbanas</h1>", unsafe_allow_html=True)

# Cargar modelo YOLO
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Mapeo de clases relevantes para escenas urbanas
URBAN_CLASSES = {
    2: {"nombre": "Carro", "color": (255, 0, 0)},
    3: {"nombre": "Moto", "color": (0, 255, 0)},
    5: {"nombre": "Bus", "color": (255, 165, 0)},
    7: {"nombre": "Camión", "color": (255, 0, 255)},
    0: {"nombre": "Persona", "color": (0, 0, 255)},
    9: {"nombre": "Semáforo", "color": (255, 255, 0)},
}

# Sidebar para control de parámetros
st.sidebar.markdown("## ⚙️ Configuración")
confidence = st.sidebar.slider(
    "Nivel de Confianza",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Ajusta la sensibilidad de detección"
)

# Subir imagen
st.markdown("<h2>📤 Carga tu imagen</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Selecciona una imagen de una calle",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    # Abrir imagen con PIL
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
    
    # Realizar detección
    with st.spinner("Procesando imagen..."):
        results = model(image_array, conf=confidence)
    
    # Procesar resultados
    detections = {}
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])
            
            if class_id in URBAN_CLASSES:
                class_name = URBAN_CLASSES[class_id]["nombre"]
                if class_name not in detections:
                    detections[class_name] = []
                
                # Guardar coordenadas normalizadas
                coords = box.xyxy[0].cpu().numpy()
                detections[class_name].append({
                    "coords": coords,
                    "confidence": confidence_score
                })
    
    # Dibujar cajas en la imagen
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    
    for class_name, boxes in detections.items():
        color = URBAN_CLASSES[list(URBAN_CLASSES.keys())[list([v["nombre"] for v in URBAN_CLASSES.values()]).index(class_name)]]["color"]
        
        for box_data in boxes:
            x1, y1, x2, y2 = box_data["coords"]
            
            # Dibujar rectángulo
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Dibujar etiqueta
            label = f"{class_name} ({box_data['confidence']:.2f})"
            draw.text((x1, y1 - 10), label, fill=color)
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>📸 Imagen Original</h3>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("<h3>✅ Detecciones</h3>", unsafe_allow_html=True)
        st.image(image_with_boxes, use_column_width=True)
    
    # Resumen de detecciones
    st.markdown("<h2>📊 Resumen de Objetos Detectados</h2>", unsafe_allow_html=True)
    
    if detections:
        # Crear columnas para métricas
        cols = st.columns(len(detections))
        
        resumen_text = "Detectados: "
        for idx, (class_name, boxes) in enumerate(detections.items()):
            count = len(boxes)
            resumen_text += f"{count} {class_name}{'s' if count != 1 else ''}"
            if idx < len(detections) - 1:
                resumen_text += ", "
            
            # Mostrar métrica
            with cols[idx]:
                st.metric(
                    label=class_name,
                    value=count,
                    label_visibility="visible"
                )
        
        # Mostrar resumen en texto
        st.info(f"✨ {resumen_text}")
        
        # Tabla detallada
        st.markdown("<h3>📋 Detalles de Confianza</h3>", unsafe_allow_html=True)
        
        detailed_data = []
        for class_name, boxes in detections.items():
            for i, box_data in enumerate(boxes):
                detailed_data.append({
                    "Objeto": class_name,
                    "Número": i + 1,
                    "Confianza": f"{box_data['confidence']:.2%}"
                })
        
        if detailed_data:
            st.dataframe(detailed_data, use_container_width=True)
    
    else:
        st.warning("⚠️ No se detectaron objetos en la imagen. Intenta con otra o ajusta el nivel de confianza.")

else:
    st.info("👈 Carga una imagen para comenzar la detección de objetos")