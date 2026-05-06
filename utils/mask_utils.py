import os
import numpy as np
import tensorflow as tf

class SemanticSegmentationModel:
    def __init__(self, model_path='model/unet_model (1).keras', labels_path='model/labels.txt'):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.classes = []
        self.load_labels()
        self.load_model()

    def load_labels(self):
        if os.path.exists(self.labels_path):
            with open(self.labels_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines() if line.strip()]
        else:
            self.classes = ['Background', 'Foreground']

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, preprocessed_image, original_shape=None):
        if self.model is None:
            return None

        preds = self.model.predict(preprocessed_image)
        
        # معالجة مخرجات الموديل (Multi-class vs Binary)
        if preds.shape[-1] > 1:
            mask = np.argmax(preds, axis=-1)[0]
        else:
            mask = (preds[0, :, :, 0] > 0.5).astype(np.uint8)

        # تغيير الحجم للحجم الأصلي بدقة
        if original_shape:
            mask_resized = tf.image.resize(
                mask[..., tf.newaxis], 
                original_shape, 
                method='nearest'
            ).numpy()[..., 0].astype(np.uint8)
        else:
            mask_resized = mask.astype(np.uint8)

        unique_ids = np.unique(mask_resized)
        classes_detected = [self.classes[int(i)] if int(i) < len(self.classes) else f"Class {i}" for i in unique_ids]

        return {
            "mask": mask_resized,
            "classes_detected": classes_detected
        }
