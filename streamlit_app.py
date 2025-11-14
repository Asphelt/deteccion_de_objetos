import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import io

# Configuración de la página
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
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #00d4ff;
            color: white;
        }
        h1, h2, h3 {
            color: #00d4ff;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("🚗 Detección de Objetos en Escenas Urbanas")
st.markdown("Carga una imagen de una calle y detecta automáticamente carros, peatones, semáforos y más")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    confidence_threshold = st.slider(
        "Confianza de detección",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Mayor valor = detecciones más precisas pero menos sensibles"
    )
    
    st.markdown("---")
    st.markdown("""
    ### Información del modelo
    - **Modelo:** YOLOv11
    - **Objetos detectados:**
      - Carros (car)
      - Autobuses (bus)
      - Camiones (truck)
      - Peatones (person)
      - Semáforos (traffic light)
      - Motocicletas (motorcycle)
    """)

# Cargar modelo YOLO
@st.cache_resource
def load_model():
    return YOLO("yolov11n.pt")

model = load_model()

# Mapeo de clases a colores y etiquetas en español
class_mapping = {
    2: {"name": "Carro", "color": (255, 0, 0)},  # Rojo
    5: {"name": "Autobús", "color": (0, 255, 255)},  # Cian
    7: {"name": "Camión", "color": (255, 165, 0)},  # Naranja
    0: {"name": "Persona", "color": (0, 255, 0)},  # Verde
    9: {"name": "Semáforo", "color": (255, 255, 0)},  # Amarillo
    3: {"name": "Motocicleta", "color": (255, 0, 255)},  # Magenta
}

# Área de carga de imagen
st.markdown("### 📸 Cargar Imagen")
uploaded_file = st.file_uploader(
    "Selecciona una imagen de una calle",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Soporta formatos: JPG, PNG, BMP"
)

if uploaded_file is not None:
    # Cargar imagen
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
    
    # Realizar detección
    with st.spinner("Detectando objetos..."):
        results = model(image_array, conf=confidence_threshold, verbose=False)
    
    # Procesar resultados
    detection_counts = {}
    detections_list = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id in class_mapping:
                class_name = class_mapping[class_id]["name"]
                
                # Contar objetos
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                
                # Guardar detalles
                detections_list.append({
                    "Objeto": class_name,
                    "Confianza": f"{confidence:.2%}",
                    "Coordenadas": f"({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])})"
                })
    
    # Dibujar cajas de detección
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id in class_mapping:
                color = class_mapping[class_id]["color"]
                class_name = class_mapping[class_id]["name"]
                
                # Coordenadas
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Dibujar rectángulo
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Dibujar etiqueta
                label = f"{class_name} {confidence:.2%}"
                draw.text((x1, y1 - 10), label, fill=color)
    
    # Layout de dos columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Imagen Original")
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("### Imagen con Detecciones")
        st.image(image_with_boxes, use_column_width=True)
    
    # Resumen de detecciones
    st.markdown("---")
    st.markdown("### 📊 Resumen de Detecciones")
    
    if detection_counts:
        # Mostrar contadores en tarjetas
        cols = st.columns(len(detection_counts))
        for (obj_name, count), col in zip(detection_counts.items(), cols):
            with col:
                st.metric(label=obj_name, value=count)
        
        # Resumen en texto
        summary_text = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" 
                                  for name, count in detection_counts.items()])
        st.success(f"✅ Detectados: {summary_text}")
        
        # Tabla detallada
        if detections_list:
            st.markdown("### 📋 Detalles de Detecciones")
            st.dataframe(detections_list, use_container_width=True)
    else:
        st.warning("⚠️ No se detectaron objetos en la imagen. Intenta ajustar la confianza de detección.")
else:
    st.info("👆 Carga una imagen para comenzar la detección de objetos")