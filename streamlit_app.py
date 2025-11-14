import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# Configuración de la página
st.set_page_config(
    page_title="Detector de Objetos Urbanos",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Detector de Objetos en Escenas Urbanas")
st.write("Sube una imagen de una calle para detectar automáticamente coches, peatones y semáforos")

# Cargar modelo YOLO11 (se descarga automáticamente la primera vez)
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Mapeo de clases de interés
CLASES_INTERES = {
    2: "Carros",      # car
    3: "Motos",       # motorcycle
    5: "Autobuses",   # bus
    7: "Camiones",    # truck
    0: "Personas",    # person
    9: "Semáforos"    # traffic light
}

# Interfaz de upload
uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Puedes subir imágenes en formato JPG, PNG o BMP"
)

if uploaded_file is not None:
    # Convertir archivo a imagen
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    # Realizar predicción
    with st.spinner("Analizando imagen..."):
        results = model.predict(
            source=image_np,
            conf=0.25,  # Confianza mínima del 25%
            verbose=False
        )
    
    result = results[0]
    
    # Procesar detecciones
    detections_count = {}
    boxes_data = []
    
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # Contar solo las clases de interés
            if class_id in CLASES_INTERES:
                clase_español = CLASES_INTERES[class_id]
                detections_count[clase_español] = detections_count.get(clase_español, 0) + 1
            
            # Guardar datos de cajas para dibujar
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            boxes_data.append({
                'xyxy': xyxy,
                'class_name': class_name,
                'conf': conf,
                'class_id': class_id
            })
    
    # Dibujar cajas en la imagen
    image_with_boxes = image.copy()
    draw = Image.new('RGBA', image_with_boxes.size, (255, 255, 255, 0))
    
    # Usar cv2 para dibujar cajas
    image_cv2 = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)
    
    colors = {
        0: (0, 255, 0),      # Personas - Verde
        2: (255, 0, 0),      # Carros - Azul
        3: (255, 165, 0),    # Motos - Naranja
        5: (255, 0, 255),    # Autobuses - Magenta
        7: (0, 255, 255),    # Camiones - Cian
        9: (0, 0, 255)       # Semáforos - Rojo
    }
    
    for box_info in boxes_data:
        xyxy = box_info['xyxy']
        x1, y1, x2, y2 = map(int, xyxy)
        class_id = box_info['class_id']
        class_name = box_info['class_name']
        conf = box_info['conf']
        
        color = colors.get(class_id, (200, 200, 200))
        
        # Dibujar caja
        cv2.rectangle(image_cv2, (x1, y1), (x2, y2), color, 2)
        
        # Dibujar etiqueta
        label = f"{class_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Fondo para el texto
        cv2.rectangle(
            image_cv2,
            (x1, y1 - text_size[1] - 4),
            (x1 + text_size[0] + 4, y1),
            color,
            -1
        )
        
        # Texto
        cv2.putText(
            image_cv2,
            label,
            (x1 + 2, y1 - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
    
    # Convertir de vuelta a RGB
    image_result = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    image_result = Image.fromarray(image_result)
    
    # Mostrar resultados
    st.divider()
    
    # Columnas para layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Imagen Analizada")
        st.image(image_result, use_container_width=True)
    
    with col2:
        st.subheader("📊 Resumen de Detecciones")
        
        if detections_count:
            # Mostrar conteos
            total = sum(detections_count.values())
            st.metric("Total de Objetos", total)
            
            st.write("---")
            
            for objeto, cantidad in sorted(detections_count.items(), key=lambda x: x[1], reverse=True):
                st.metric(objeto, cantidad)
            
            # Crear texto de resumen
            resumen = "Detectados: " + ", ".join(
                [f"{cantidad} {objeto}" for objeto, cantidad in sorted(detections_count.items(), key=lambda x: x[1], reverse=True)]
            )
            
            st.write("---")
            st.info(f"✓ {resumen}")
        else:
            st.warning("No se detectaron objetos de interés en la imagen")
    
    # Mostrar información adicional
    with st.expander("ℹ️ Información técnica"):
        st.write(f"**Modelo:** YOLOv11n")
        st.write(f"**Total de detecciones:** {len(boxes_data)}")
        st.write(f"**Confianza mínima:** 25%")
        st.write(f"**Dimensiones de imagen:** {image_np.shape}")

else:
    # Mostrar instrucciones
    st.info("""
    👈 **Cómo usar:**
    
    1. Haz clic en "Browse files" para seleccionar una imagen
    2. Sube una foto de una calle con coches, peatones o semáforos
    3. La app detectará automáticamente los objetos y mostrará:
       - La imagen con cajas alrededor de cada objeto
       - Un resumen del conteo de objetos
    
    **Objetos que detecta:**
    - 🚗 Carros
    - 🏍️ Motos
    - 🚌 Autobuses
    - 🚐 Camiones
    - 👥 Personas
    - 🚦 Semáforos
    """)
    
    st.write("---")
    
    # Ejemplo con descripción
    st.subheader("Ejemplo de uso")
    st.write("""
    Sube una foto como esta:
    - Una calle con vehículos
    - Una intersección con peatones
    - Una escena urbana con semáforos
    
    Y obtendrás automáticamente el análisis con detecciones.
    """)