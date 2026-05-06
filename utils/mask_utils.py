import cv2
import numpy as np

def get_class_colors(num_classes):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(max(num_classes, 20), 3), dtype=np.uint8)
    colors[0] = [0, 0, 0] # Background black
    return colors

def colorize_mask(mask, colors):
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in np.unique(mask):
        idx = int(class_id)
        if idx < len(colors):
            colored_mask[mask == idx] = colors[idx]
    return colored_mask

def blend_masks(image, colored_mask, alpha=0.5):
    # تحويل الصورة لمصفوفة إذا لم تكن كذلك
    img_np = np.array(image)
    # دمج فقط المناطق التي تحتوي على ماسك (ليست سوداء)
    mask_indices = np.any(colored_mask != [0, 0, 0], axis=-1)
    blended = img_np.copy()
    blended[mask_indices] = cv2.addWeighted(
        img_np[mask_indices], 1 - alpha,
        colored_mask[mask_indices], alpha, 0
    )
    return blended

def calculate_area_percentages(mask, classes):
    total_pixels = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    percentages = {}
    for val, count in zip(unique, counts):
        idx = int(val)
        name = classes[idx] if idx < len(classes) else f"Class {idx}"
        percentages[name] = (count / total_pixels) * 100
    return dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))
