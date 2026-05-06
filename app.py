import streamlit as st
import numpy as np
import cv2
import os
from utils.predictor import SemanticSegmentationModel
from utils.visualization import draw_semantic_mask
from utils.mask_utils import calculate_area_percentages

st.set_page_config(page_title="VisionSeg AI", layout="wide", page_icon="🧩")

@st.cache_resource
def load_model():
    return SemanticSegmentationModel()

model = load_model()

st.title("🧩 VisionSeg AI - Gemini Powered")
st.sidebar.header("Controls")
opacity = st.sidebar.slider("Mask Opacity", 0.0, 1.0, 0.5)
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

image = None
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Run Segmentation", type="primary"):
        with st.spinner("Processing..."):
            # Preprocess
            input_img = cv2.resize(image, (128, 128))
            input_img = input_img.astype(np.float32) / 255.0
            input_img = np.expand_dims(input_img, axis=0)

            prediction = model.predict(input_img, original_shape=image.shape[:2])

            if prediction:
                blended_image, colors = draw_semantic_mask(image, prediction, model.classes, opacity=opacity)
                
                with col2:
                    st.subheader("Segmentation Output")
                    st.image(blended_image, use_container_width=True)

                # Analysis Summary
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Classes Detected:**")
                    for cls in prediction['classes_detected']:
                        st.write(f"- {cls}")
                with c2:
                    st.write("**Area Distribution:**")
                    stats = calculate_area_percentages(prediction['mask'], model.classes)
                    for cls, pct in stats.items():
                        st.progress(pct/100, text=f"{cls}: {pct:.1f}%")

                # Safe Legend Generation
                st.subheader("🎨 Legend")
                legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
                for idx, cls in enumerate(model.classes):
                    if cls in prediction['classes_detected']:
                        color = colors[idx]
                        hex_c = '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))
                        legend_html += f"<div style='padding:5px 10px; background:{hex_c}; color:white; border-radius:5px;'>{cls}</div>"
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)
