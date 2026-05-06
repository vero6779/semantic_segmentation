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
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, preprocessed_image, original_shape=None):
        if self.model is None:
            raise ValueError("Model is not loaded.")

        # 🔥 تأكيد input
        if not isinstance(preprocessed_image, np.ndarray):
            raise ValueError("Input must be numpy array")

        if preprocessed_image.ndim != 4:
            raise ValueError(f"Expected shape (1, H, W, 3), got {preprocessed_image.shape}")

        # 🔥 inference
        preds = self.model.predict(preprocessed_image)

        if preds is None or len(preds) == 0:
            raise ValueError("Model returned empty predictions")

        print("Preds shape:", preds.shape)

        # 🔥 التعامل مع كل الحالات
        if preds.ndim == 4 and preds.shape[-1] > 1:
            mask = np.argmax(preds, axis=-1)[0]

        elif preds.ndim == 4 and preds.shape[-1] == 1:
            mask = (preds[0, :, :, 0] > 0.5).astype(np.uint8)

        elif preds.ndim == 3:
            mask = preds[0]

        else:
            raise ValueError(f"Unexpected prediction shape: {preds.shape}")

        if mask is None:
            raise ValueError("Mask is None")

        # 🔥 validation قوي لـ original_shape
        valid_shape = False
        if original_shape is not None:
            try:
                h, w = original_shape
                if h is not None and w is not None:
                    h, w = int(h), int(w)
                    if h > 0 and w > 0:
                        valid_shape = True
                        original_shape = (h, w)
            except Exception:
                valid_shape = False

        # 🔥 resize آمن
        if valid_shape:
            try:
                mask_resized = tf.image.resize(
                    mask[..., tf.newaxis],
                    original_shape,
                    method='nearest'
                ).numpy()[..., 0].astype(np.uint8)
            except Exception as e:
                print("Resize failed:", e)
                mask_resized = mask.astype(np.uint8)
        else:
            mask_resized = mask.astype(np.uint8)

        # 🔥 استخراج الكلاسات
        unique_classes = np.unique(mask_resized)

        classes_detected = [
            self.classes[c] if c < len(self.classes) else f"Class {c}"
            for c in unique_classes
        ]

        return {
            "mask": mask_resized,
            "class_map": mask_resized,
            "classes_detected": classes_detected
        }
