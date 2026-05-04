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

    def predict(self, preprocessed_image, original_shape=None):
        if self.model is None:
            raise ValueError("Model is not loaded.")
            
        # Run inference
        preds = self.model.predict(preprocessed_image)
        if preds is None or len(preds) == 0:
            raise ValueError("Model prediction returned None or empty.")
        
        # Post-process: Get the class with the highest probability per pixel
        # Assumes output shape is (1, H, W, num_classes)
        mask = np.argmax(preds, axis=-1)[0] # Shape (H, W)
        
        if mask is None:
            raise ValueError("Computed mask is None.")
        
        # Validate original_shape
        is_valid_shape = False
        if original_shape is not None:
            if isinstance(original_shape, (tuple, list)) and len(original_shape) == 2:
                if original_shape[0] is not None and original_shape[1] is not None:
                    try:
                        # Ensure they are valid integers
                        original_shape = (int(original_shape[0]), int(original_shape[1]))
                        is_valid_shape = True
                    except (ValueError, TypeError):
                        pass
        
        # Resize mask back to original image shape
        if is_valid_shape:
            mask_resized = tf.image.resize(mask[..., tf.newaxis], original_shape, method='nearest').numpy()
            mask_resized = mask_resized[..., 0].astype(np.uint8)
        else:
            mask_resized = mask.astype(np.uint8)
        
        # Get detected classes
        unique_classes = np.unique(mask_resized)
        classes_detected = [self.classes[c] if c < len(self.classes) else f"Class {c}" for c in unique_classes]
        
        return {
            "mask": mask_resized,
            "class_map": mask_resized,
            "classes_detected": classes_detected
        }
