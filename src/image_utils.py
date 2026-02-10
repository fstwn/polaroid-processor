"""
Image utility functions for Polaroid processing.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Disable PIL decompression bomb warning for large TIFF files
Image.MAX_IMAGE_PIXELS = None


def apply_barrel_distortion_correction(
    image: np.ndarray,
    k1: float = -0.15,
    k2: float = 0.0,
    k3: float = 0.0
) -> np.ndarray:
    """
    Apply radial distortion correction to fix barrel/pincushion distortion
    caused by the Polaroid holder not keeping the photo perfectly flat.

    Barrel distortion (bulge outward) is corrected with negative k1 values.
    Pincushion distortion (pinch inward) is corrected with positive k1 values.

    Args:
        image: Input image (BGR or grayscale)
        k1: Primary radial distortion coefficient (-0.3 to 0.3 typical)
        k2: Secondary radial distortion coefficient (usually 0)
        k3: Tertiary radial distortion coefficient (rarely needed)

    Returns:
        Corrected image with same dimensions as input
    """
    if k1 == 0 and k2 == 0 and k3 == 0:
        logger.info("Distortion correction skipped (all coefficients are 0)")
        return image

    height, width = image.shape[:2]

    # Create camera matrix (assuming image center is optical center)
    # Focal length estimate based on image dimensions
    focal_length = max(width, height)
    cx = width / 2.0
    cy = height / 2.0

    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Distortion coefficients: [k1, k2, p1, p2, k3]
    # p1, p2 are tangential distortion (set to 0 for pure radial correction)
    dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float64)

    # Get optimal new camera matrix to avoid cropping
    # alpha=1 means all pixels are retained (some black borders may appear)
    # alpha=0 means only valid pixels are retained (some image may be cropped)
    # We use alpha=0.5 for a balance
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), alpha=0.5
    )

    # Apply undistortion
    corrected = cv2.undistort(
        image, camera_matrix, dist_coeffs, None, new_camera_matrix
    )

    # Optionally crop to ROI to remove black borders
    # For Polaroid processing, we'll keep the full image and let
    # subsequent cropping handle any edge artifacts
    x, y, w, h = roi
    if w > 0 and h > 0 and w < width and h < height:
        # Only crop if ROI is valid and smaller than original
        # We add a small margin to avoid cutting too much
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2 * margin)
        h = min(height - y, h + 2 * margin)
        corrected = corrected[y:y+h, x:x+w]

        # Resize back to original dimensions to maintain consistency
        corrected = cv2.resize(
            corrected, (width, height), interpolation=cv2.INTER_LANCZOS4
        )

    logger.info(
        f"Applied barrel distortion correction: k1={k1}, k2={k2}, k3={k3}"
    )
    return corrected


def get_distortion_correction_config(
    config: Dict
) -> Tuple[bool, float, float, float]:
    """
    Extract distortion correction parameters from config dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (enabled, k1, k2, k3)
    """
    distortion_config = config.get('distortion_correction', {})
    enabled = distortion_config.get('enabled', False)
    k1 = distortion_config.get('k1', -0.15)
    k2 = distortion_config.get('k2', 0.0)
    k3 = distortion_config.get('k3', 0.0)
    return enabled, k1, k2, k3


def calculate_quality_score(
    image: np.ndarray,
    corners: np.ndarray,
    target_aspect_ratio: float = 0.822
) -> float:
    """
    Calculate a quality score for the detected Polaroid.

    Args:
        image: Original image
        corners: Detected corner points
        target_aspect_ratio: Expected aspect ratio (width/height)

    Returns:
        Quality score between 0 and 1
    """
    if corners is None or len(corners) != 4:
        return 0.0

    # Calculate area of detected rectangle
    area = cv2.contourArea(corners)
    image_area = image.shape[0] * image.shape[1]
    area_ratio = area / image_area

    # Calculate aspect ratio accuracy
    x, y, w, h = cv2.boundingRect(corners)
    aspect_ratio = w / h
    aspect_diff = abs(aspect_ratio - target_aspect_ratio)
    aspect_accuracy = 1.0 - aspect_diff / target_aspect_ratio

    # Calculate corner angle accuracy (should be close to 90 degrees)
    angles = []
    for i in range(4):
        p1 = corners[i][0]
        p2 = corners[(i + 1) % 4][0]
        p3 = corners[(i + 2) % 4][0]

        v1 = p1 - p2
        v2 = p3 - p2
        dot_product = np.dot(v1, v2)
        norms_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_angle = dot_product / norms_product
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(abs(angle - 90))

    angle_accuracy = 1.0 - np.mean(angles) / 90

    # Combined score
    score = 0.4 * area_ratio + 0.3 * aspect_accuracy + 0.3 * angle_accuracy
    return min(1.0, max(0.0, score))


def save_with_metadata(
    image: np.ndarray,
    output_path: str,
    dpi: int = 600
) -> None:
    """
    Save image with proper metadata.

    Args:
        image: Image to save
        output_path: Output file path
        dpi: DPI for the image
    """
    # Convert BGR to RGB for PIL
    if len(image.shape) == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image

    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)

    # Save with DPI metadata
    pil_image.save(output_path, "TIFF", compression="lzw", dpi=(dpi, dpi))


def apply_safety_margin_crop(
    image: np.ndarray,
    margin_percent: float = 1.0
) -> np.ndarray:
    """
    Crop edges of an image by a percentage to remove artifacts.

    This should be applied AFTER distortion correction to remove any
    black edges introduced by the distortion correction process.

    Args:
        image: Input image
        margin_percent: Percentage of each edge to crop (e.g., 1.0 = 1%)

    Returns:
        Cropped image
    """
    if margin_percent <= 0:
        return image

    height, width = image.shape[:2]
    # Calculate pixels to crop from each edge
    crop_x = int(width * margin_percent / 100.0)
    crop_y = int(height * margin_percent / 100.0)

    # Ensure we don't crop too much
    if crop_x * 2 >= width or crop_y * 2 >= height:
        logger.warning(
            f"Safety margin {margin_percent}% too large, skipping crop"
        )
        return image

    # Crop the image
    cropped = image[crop_y:height-crop_y, crop_x:width-crop_x]

    logger.info(
        f"Applied safety margin crop: {margin_percent}% "
        f"({width}x{height} -> {cropped.shape[1]}x{cropped.shape[0]})"
    )

    return cropped


def create_debug_visualization(
    image: np.ndarray,
    corners: np.ndarray,
    output_path: str
) -> None:
    """
    Create a debug visualization showing detected corners.

    Args:
        image: Original image
        corners: Detected corner points
        output_path: Path to save debug image
    """
    debug_image = image.copy()

    if corners is not None:
        # Draw detected corners
        for i, corner in enumerate(corners):
            x, y = corner[0]
            cv2.circle(debug_image, (int(x), int(y)), 10, (0, 255, 0), -1)
            cv2.putText(
                debug_image, str(i), (int(x) + 15, int(y) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # Draw rectangle outline
        cv2.polylines(debug_image, [corners], True, (0, 255, 0), 3)

    cv2.imwrite(output_path, debug_image)
