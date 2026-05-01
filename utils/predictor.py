import os
import numpy as np
import tensorflow as tf
from PIL import Image

class SemanticSegmentationModel:
    def __init__(self, model_path='model/unet_model (1).keras', labels_path='model/labels.txt'):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.classes = []
        
        self.load_labels()
        self.load_model()
        
    def load_labels(self):
        try:
            with open(self.labels_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            print(f"Labels file not found at {self.labels_path}. Using default labels.")
            self.classes = ['Background', 'Foreground']
            
    def load_model(self):
        try:
            # Auto-detect CPU/GPU is handled by TensorFlow implicitly
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
            
        # Resize to typical U-Net input size (e.g., 256x256). 
        # Using tf.image.resize handles dynamic aspect ratios safely.
        target_size = (256, 256)
        image_resized = tf.image.resize(image, target_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized / 255.0
        
        # Add batch dimension
        input_tensor = tf.expand_dims(image_normalized, 0)
        return input_tensor, image.shape[:2]

    def predict(self, image):
        if self.model is None:
            raise ValueError("Model is not loaded.")
            
        input_tensor, original_shape = self.preprocess(image)
        
        # Run inference
        preds = self.model.predict(input_tensor)
        
        # Post-process: Get the class with the highest probability per pixel
        # Assumes output shape is (1, H, W, num_classes)
        mask = np.argmax(preds, axis=-1)[0] # Shape (H, W)
        
        # Resize mask back to original image shape
        mask_resized = tf.image.resize(mask[..., tf.newaxis], original_shape, method='nearest').numpy()
        mask_resized = mask_resized[..., 0].astype(np.uint8)
        
        # Get detected classes
        unique_classes = np.unique(mask_resized)
        classes_detected = [self.classes[c] if c < len(self.classes) else f"Class {c}" for c in unique_classes]
        
        return {
            "mask": mask_resized,
            "class_map": mask_resized,
            "classes_detected": classes_detected
        }
