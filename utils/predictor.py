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

        # 🔥 تأكيد إن input صح
        if not isinstance(preprocessed_image, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if preprocessed_image.ndim != 4:
            raise ValueError(f"Expected input shape (1, H, W, 3), got {preprocessed_image.shape}")

        # Run inference
        preds = self.model.predict(preprocessed_image)

        if preds is None or len(preds) == 0:
            raise ValueError("Model prediction returned None or empty.")

        print("Preds shape:", preds.shape)

        # 🔥 التعامل مع كل أنواع الموديلات
        if preds.ndim == 4 and preds.shape[-1] > 1:
            # multi-class segmentation
            mask = np.argmax(preds, axis=-1)[0]

        elif preds.ndim == 4 and preds.shape[-1] == 1:
            # binary segmentation (sigmoid)
            mask = (preds[0, :, :, 0] > 0.5).astype(np.uint8)

        elif preds.ndim == 3:
            # shape (1, H, W)
            mask = preds[0]

        else:
            raise ValueError(f"Unexpected prediction shape: {preds.shape}")

        if mask is None:
            raise ValueError("Mask computation failed.")

        # 🔥 تأكيد original_shape
        if (
            original_shape is not None and
            isinstance(original_shape, (tuple, list)) and
            len(original_shape) == 2 and
            original_shape[0] is not None and
            original_shape[1] is not None
        ):
            try:
                original_shape = (int(original_shape[0]), int(original_shape[1]))

                mask_resized = tf.image.resize(
                    mask[..., tf.newaxis],
                    original_shape,
                    method='nearest'
                ).numpy()

                mask_resized = mask_resized[..., 0].astype(np.uint8)

            except Exception as e:
                print("Resize failed, using original mask:", e)
                mask_resized = mask.astype(np.uint8)
        else:
            mask_resized = mask.astype(np.uint8)

        # استخراج الكلاسات
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
