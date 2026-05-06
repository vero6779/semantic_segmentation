import cv2
import numpy as np

def get_class_colors(num_classes):
    """
    Generate a distinct color for each class using a random seed.
    """
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

    # Background = black
    if num_classes > 0:
        colors[0] = [0, 0, 0]

    return colors


def colorize_mask(mask, colors):
    """
    Convert a 2D class mask into an RGB color mask.
    """
    if mask is None:
        raise ValueError("Mask is None in colorize_mask")

    if not isinstance(mask, np.ndarray):
        raise ValueError("Mask must be numpy array")

    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    unique_classes = np.unique(mask)

    for class_id in unique_classes:
        try:
            class_id = int(class_id)
        except:
            continue

        if class_id < len(colors):
            colored_mask[mask == class_id] = colors[class_id]

    return colored_mask


def blend_masks(image, colored_mask, alpha=0.5):
    """
    Blend the original image and the colored mask using alpha blending.
    """
    if image is None or colored_mask is None:
        raise ValueError("Image or colored_mask is None in blend_masks")

    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be numpy array")

    if not isinstance(colored_mask, np.ndarray):
        raise ValueError("colored_mask must be numpy array")

    mask_indices = np.any(colored_mask != [0, 0, 0], axis=-1)

    blended = image.copy()

    blended[mask_indices] = cv2.addWeighted(
        image[mask_indices], 1 - alpha,
        colored_mask[mask_indices], alpha, 0
    )

    return blended


def calculate_area_percentages(mask, classes):
    """
    Calculate percentage of each class safely.
    """
    if mask is None:
        raise ValueError("Mask is None in calculate_area_percentages")

    if not isinstance(mask, np.ndarray):
        raise ValueError("Mask must be numpy array")

    if mask.size == 0:
        raise ValueError("Mask is empty")

    total_pixels = mask.size

    unique, counts = np.unique(mask, return_counts=True)
    percentages = {}

    for val, count in zip(unique, counts):
        if val is None:
            continue

        try:
            class_index = int(val)
        except:
            continue

        class_name = classes[class_index] if class_index < len(classes) else f"Class {class_index}"
        percentages[class_name] = (count / total_pixels) * 100

    return dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))
