import streamlit as st
import numpy as np
from PIL import Image
import os
from utils.predictor import SemanticSegmentationModel
from utils.visualization import draw_semantic_mask
from utils.mask_utils import calculate_area_percentages

# Must be the first Streamlit command
st.set_page_config(page_title="Semantic Segmentation", layout="wide", page_icon="🧩")

st.title("🧩 Semantic Segmentation App")
st.markdown("Upload an image to perform pixel-wise semantic segmentation.")

@st.cache_resource
def load_model():
    model_dir = "model"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5', '.pt', '.pth', '.onnx'))]
    model_path = os.path.join(model_dir, model_files[0]) if model_files else "model/unet_model (1).keras"
    return SemanticSegmentationModel(model_path=model_path)

# Load model via caching to prevent reload on every interaction
model = load_model()

st.sidebar.header("Controls")
opacity = st.sidebar.slider("Mask Opacity", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
demo_path = "assets/demo.png"
use_demo = st.sidebar.checkbox("Use Demo Image", value=True)

image = None
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.sidebar.error("Invalid image upload.")
elif use_demo and os.path.exists(demo_path):
    image = Image.open(demo_path)
else:
    st.info("Please upload an image or check 'Use Demo Image'.")

if image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    st.markdown("---")
    
    if st.button("Run Segmentation", type="primary", use_container_width=True):
        with st.spinner("Running deep learning inference..."):
            try:
                prediction = model.predict(image)
                blended_image, colors = draw_semantic_mask(image, prediction, model.classes, opacity=opacity)
                
                with col2:
                    st.subheader("Segmentation Output")
                    st.image(blended_image, use_container_width=True)
                    
                st.subheader("📊 Analysis Summary")
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.markdown("**Classes Detected:**")
                    for cls in prediction['classes_detected']:
                        st.markdown(f"- ✅ {cls}")
                        
                with stats_col2:
                    st.markdown("**Area Distribution:**")
                    percentages = calculate_area_percentages(prediction['mask'], model.classes)
                    for cls, pct in percentages.items():
                        # Exclude pure background from progress if preferred, or include it
                        st.progress(pct / 100.0, text=f"{cls}: {pct:.1f}%")
                        
                st.markdown("---")
                st.subheader("🎨 Legend")
                # Generate clean HTML legend
                legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 15px; margin-top: 10px;'>"
                for idx, cls in enumerate(model.classes):
                    if cls in prediction['classes_detected']:
                        color = colors[idx]
                        # OpenCV uses BGR natively, but we converted it. PIL is RGB. Our colors are RGB.
                        hex_color = '#%02x%02x%02x' % tuple(color)
                        legend_html += f'''
                        <div style='display: flex; align-items: center; background-color: #f0f2f6; padding: 5px 10px; border-radius: 5px;'>
                            <div style='width: 20px; height: 20px; background-color: {hex_color}; margin-right: 10px; border-radius: 3px; border: 1px solid #ccc;'></div>
                            <span style='font-weight: 500; color: #31333F;'>{cls}</span>
                        </div>
                        '''
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during segmentation: {str(e)}")
