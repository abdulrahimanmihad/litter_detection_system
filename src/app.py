import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Litter Detection AI", page_icon="♻️")

# Resolve the model path relative to this file so the app runs from any
# working directory (e.g. `streamlit run src/app.py` or a systemd service).
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

# 2. Load Model (with caching so it doesn't reload every time)
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# 3. UI Layout
st.title("♻️ Real-Time Litter Detection System")
st.write("Upload an image to detect **Plastic**, **Cigarettes**, **Glass**, and more.")

# 4. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 5. Run Detection Button
    if st.button('🔍 Detect Litter'):
        with st.spinner('Analyzing...'):
            # Run inference with OPTIMIZED confidence threshold
            results = model.predict(image, conf=0.189)
            
            # Plot results on the image
            res_plotted = results[0].plot()
            
            # Show Result
            st.image(res_plotted, caption='Detected Litter', use_column_width=True)
            
            # Show Stats
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success(f"✅ Detected {len(boxes)} object(s)!")
                # Optional: List classes detected
                cls_ids = boxes.cls.cpu().numpy()
                detected_classes = [model.names[int(cls)] for cls in cls_ids]
                st.write("Found:", ", ".join(set(detected_classes)))
            else:
                st.warning("No litter detected (try a lower confidence threshold if needed).")