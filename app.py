import os
import cv2
import numpy as np
from PIL import Image
from utils.predictor import SemanticSegmentationModel
from utils.visualization import draw_semantic_mask

def main():
    print("Loading Semantic Segmentation Model...")
    
    # Dynamically find the model in the model directory
    model_dir = "model"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(('.keras', '.h5', '.pt', '.pth', '.onnx'))]
    model_path = os.path.join(model_dir, model_files[0]) if model_files else "model/unet_model (1).keras"
    
    try:
        model = SemanticSegmentationModel(model_path=model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    demo_image_path = "assets/demo.png"
    if not os.path.exists(demo_image_path):
        print(f"Demo image not found at {demo_image_path}. Please place an image there to test.")
        return

    print(f"Loading image from {demo_image_path}...")
    img_bgr = cv2.imread(demo_image_path)
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    print("Running inference...")
    try:
        image_np = np.array(image)
        image_np = cv2.resize(image_np, (128, 128))
        image_np = image_np.astype(np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=0)

        prediction = model.predict(image_np, original_shape=image.shape[:2])
        print("Classes detected:", prediction['classes_detected'])
        
        print("Generating visualization...")
        blended_image, _ = draw_semantic_mask(image, prediction, model.classes, opacity=0.5)
        
        output_path = "output_segmented.png"
        blended_image.save(output_path)
        print(f"Success! Segmented image saved to {output_path}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
