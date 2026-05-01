import cv2
import numpy as np

def get_class_colors(num_classes):
    """
    Generate a distinct color for each class using a random seed.
    """
    np.random.seed(42) # Consistent colors across runs
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    
    # Ensure background is purely black or transparent if it's the first class
    if num_classes > 0:
        colors[0] = [0, 0, 0]
    return colors

def colorize_mask(mask, colors):
    """
    Convert a 2D class mask into an RGB color mask.
    """
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id in np.unique(mask):
        if class_id < len(colors):
            colored_mask[mask == class_id] = colors[class_id]
    return colored_mask

def blend_masks(image, colored_mask, alpha=0.5):
    """
    Blend the original image and the colored mask using alpha blending.
    """
    # Create mask where colored_mask is not black to avoid darkening the original image on background
    mask_indices = np.any(colored_mask != [0, 0, 0], axis=-1)
    
    blended = image.copy()
    blended[mask_indices] = cv2.addWeighted(
        image[mask_indices], 1 - alpha, 
        colored_mask[mask_indices], alpha, 0
    )
    return blended
    
def calculate_area_percentages(mask, classes):
    """
    Calculate the percentage area of each class in the mask.
    Ignores background (class 0) usually, but we include it for completeness.
    """
    total_pixels = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    percentages = {}
    
    for val, count in zip(unique, counts):
        class_name = classes[val] if val < len(classes) else f"Class {val}"
        percentages[class_name] = (count / total_pixels) * 100
        
    # Sort by percentage descending
    sorted_percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))
    return sorted_percentages
