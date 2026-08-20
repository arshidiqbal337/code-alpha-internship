"""
Image preprocessing and segmentation utilities.
Includes thresholding, aspect-ratio preserving resizing, character padding,
and contour-based bounding box segmentation for handwritten text images.
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


def load_and_preprocess_image(image_path_or_array, target_size=(28, 28)) -> np.ndarray:
    """
    Loads an image file path or numpy array, converts to grayscale,
    applies Otsu binarization, handles background inversion (ensuring white text on black background),
    and resizes with padding to match target_size (28x28).
    
    Returns:
        np.ndarray: Preprocessed grayscale image normalized between [0, 255].
    """
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_path_or_array}")
    elif isinstance(image_path_or_array, np.ndarray):
        if len(image_path_or_array.shape) == 3:
            img = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY)
        else:
            img = image_path_or_array.copy()
    else:
        raise TypeError("Input must be a file path string or numpy array.")

    # Apply Gaussian Blur to smooth out noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Otsu's thresholding for optimal binarization
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect if background was inverted: check average pixel value
    # If the majority of pixels are white (mean > 127), invert to ensure text is white on black background
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    return resize_and_pad(thresh, target_size=target_size)


def resize_and_pad(img: np.ndarray, target_size=(28, 28), pad_value=0) -> np.ndarray:
    """
    Resizes an image preserving aspect ratio and pads it to target_size (28, 28).
    Paddings are added symmetrically to center the character.
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size

    # Compute scaling factor maintaining aspect ratio
    scale = min(target_h / float(h), target_w / float(w))
    new_h, new_w = int(h * scale), int(w * scale)

    # Prevent zero dimension
    new_h, new_w = max(1, new_h), max(1, new_w)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded square container
    padded = np.full((target_h, target_w), pad_value, dtype=np.uint8)

    # Compute top-left placement coordinates for centering
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return padded


def prepare_tensor(preprocessed_img: np.ndarray) -> torch.Tensor:
    """
    Converts a preprocessed (28, 28) numpy image into a normalized PyTorch tensor ready for model input.
    Shape: (1, 1, 28, 28)
    """
    pil_img = Image.fromarray(preprocessed_img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3267,))
    ])
    tensor = transform(pil_img)  # Shape: (1, 28, 28)
    return tensor.unsqueeze(0)  # Shape: (1, 1, 28, 28)


def segment_characters(image_path_or_array, min_area=30, max_area=10000) -> list:
    """
    Segments individual character bounding boxes from a handwritten text image (word or sentence).
    Sorts detected characters left-to-right (x-axis order).

    Returns:
        List of dicts: [{'box': (x, y, w, h), 'crop': np.ndarray (28, 28 preprocessed character image)}]
    """
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from: {image_path_or_array}")
    else:
        if len(image_path_or_array.shape) == 3:
            img = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY)
        else:
            img = image_path_or_array.copy()

    # Preprocessing for contour extraction
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # If background is bright, invert
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if min_area <= area <= max_area and w > 2 and h > 5:
            bounding_boxes.append((x, y, w, h))

    # Sort boxes left-to-right (by X coordinate)
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])

    segmented_chars = []
    for (x, y, w, h) in bounding_boxes:
        # Extract character ROI with slight padding margin
        margin = 2
        y1 = max(0, y - margin)
        y2 = min(thresh.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(thresh.shape[1], x + w + margin)

        char_crop = thresh[y1:y2, x1:x2]
        char_28x28 = resize_and_pad(char_crop, target_size=(28, 28))

        segmented_chars.append({
            'box': (x, y, w, h),
            'crop': char_28x28
        })

    return segmented_chars
