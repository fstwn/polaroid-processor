"""
Shadow correction for Polaroid scans.
Handles holder-induced shadows and vignetting at image edges.
"""

import cv2
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ShadowCorrector:
    """
    Corrects shadow artifacts at the edges of Polaroid scans.
    These shadows typically result from the Polaroid holder casting
    shadows during scanning.
    """

    def __init__(
        self,
        border_region_percent: float = 12.0,
        corner_region_percent: float = 15.0,
        max_brightness_boost: int = 40,
        detection_threshold: float = 0.85,
        blend_falloff: float = 0.7
    ):
        """
        Initialize the shadow corrector.

        Args:
            border_region_percent: Width of border region to analyze (% of dim)
            corner_region_percent: Size of corner regions for correction
            max_brightness_boost: Maximum brightness increase (0-255)
            detection_threshold: Brightness ratio below which to correct
            blend_falloff: How smoothly correction blends (0-1)
        """
        self.border_region_percent = border_region_percent
        self.corner_region_percent = corner_region_percent
        self.max_brightness_boost = max_brightness_boost
        self.detection_threshold = detection_threshold
        self.blend_falloff = blend_falloff

    def analyze_border_brightness(
        self,
        image: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Analyze the brightness of the white border regions.

        Args:
            image: Input image (BGR)

        Returns:
            Tuple of (reference_brightness, top, bottom, left, right averages)
        """
        height, width = image.shape[:2]

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Calculate border region sizes
        border_h = int(height * self.border_region_percent / 100)
        border_w = int(width * self.border_region_percent / 100)

        # Sample from the CENTER of each edge (where shadows are least likely)
        center_sample_ratio = 0.3  # Sample middle 30% of each edge

        # Top edge - center portion
        top_start = int(width * (0.5 - center_sample_ratio / 2))
        top_end = int(width * (0.5 + center_sample_ratio / 2))
        top_region = gray[0:border_h, top_start:top_end]

        # Bottom edge - center portion
        bottom_region = gray[height-border_h:height, top_start:top_end]

        # Left edge - center portion
        left_start = int(height * (0.5 - center_sample_ratio / 2))
        left_end = int(height * (0.5 + center_sample_ratio / 2))
        left_region = gray[left_start:left_end, 0:border_w]

        # Right edge - center portion
        right_region = gray[left_start:left_end, width-border_w:width]

        # Calculate average brightness of each region
        top_avg = np.mean(top_region) if top_region.size > 0 else 255
        bottom_avg = np.mean(bottom_region) if bottom_region.size > 0 else 255
        left_avg = np.mean(left_region) if left_region.size > 0 else 255
        right_avg = np.mean(right_region) if right_region.size > 0 else 255

        # Reference brightness is the maximum (brightest = least shadowed)
        reference = max(top_avg, bottom_avg, left_avg, right_avg)

        logger.debug(
            f"Border brightness - Ref: {reference:.1f}, "
            f"T: {top_avg:.1f}, B: {bottom_avg:.1f}, "
            f"L: {left_avg:.1f}, R: {right_avg:.1f}"
        )

        return reference, top_avg, bottom_avg, left_avg, right_avg

    def create_correction_mask(
        self,
        height: int,
        width: int,
        reference_brightness: float,
        edge_brightness: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Create a correction mask that brightens shadowed edges.

        Args:
            height: Image height
            width: Image width
            reference_brightness: Target brightness level
            edge_brightness: (top, bottom, left, right) brightness values

        Returns:
            Correction mask (float32, values 0-1 indicating boost amount)
        """
        top_avg, bottom_avg, left_avg, right_avg = edge_brightness

        # Calculate correction factors for each edge
        def calc_correction(edge_avg):
            if reference_brightness > 0:
                ratio = edge_avg / reference_brightness
            else:
                ratio = 1
            if ratio < self.detection_threshold:
                # Amount of correction needed (0 to 1)
                return min(1.0, (self.detection_threshold - ratio) / 0.3)
            return 0.0

        top_corr = calc_correction(top_avg)
        bottom_corr = calc_correction(bottom_avg)
        left_corr = calc_correction(left_avg)
        right_corr = calc_correction(right_avg)

        logger.debug(
            f"Correction factors - T: {top_corr:.2f}, B: {bottom_corr:.2f}, "
            f"L: {left_corr:.2f}, R: {right_corr:.2f}"
        )

        # Create base mask
        mask = np.zeros((height, width), dtype=np.float32)

        # Border region sizes
        border_h = int(height * self.border_region_percent / 100)
        border_w = int(width * self.border_region_percent / 100)

        # Create gradient masks for each edge
        # Top edge
        if top_corr > 0:
            for y in range(border_h):
                # Gradient from full correction at edge to 0 at border_h
                gradient = 1.0 - (y / border_h) ** self.blend_falloff
                mask[y, :] = np.maximum(mask[y, :], gradient * top_corr)

        # Bottom edge
        if bottom_corr > 0:
            for y in range(height - border_h, height):
                dist_from_edge = height - 1 - y
                grad = 1.0 - (dist_from_edge / border_h) ** self.blend_falloff
                mask[y, :] = np.maximum(mask[y, :], grad * bottom_corr)

        # Left edge
        if left_corr > 0:
            for x in range(border_w):
                gradient = 1.0 - (x / border_w) ** self.blend_falloff
                mask[:, x] = np.maximum(mask[:, x], gradient * left_corr)

        # Right edge
        if right_corr > 0:
            for x in range(width - border_w, width):
                dist_from_edge = width - 1 - x
                grad = 1.0 - (dist_from_edge / border_w) ** self.blend_falloff
                mask[:, x] = np.maximum(mask[:, x], grad * right_corr)

        # Apply extra correction to corners (often most affected)
        corner_size_h = int(height * self.corner_region_percent / 100)
        corner_size_w = int(width * self.corner_region_percent / 100)

        # Corner boost factor
        corner_boost = 1.3

        # Top-left corner
        if top_corr > 0 or left_corr > 0:
            for y in range(corner_size_h):
                for x in range(corner_size_w):
                    y_ratio = y / corner_size_h
                    x_ratio = x / corner_size_w
                    dist = np.sqrt(y_ratio ** 2 + x_ratio ** 2)
                    if dist < 1.0:
                        corner_grad = (1.0 - dist) ** self.blend_falloff
                        base_corr = max(top_corr, left_corr)
                        val = corner_grad * base_corr * corner_boost
                        mask[y, x] = max(mask[y, x], val)

        # Top-right corner
        if top_corr > 0 or right_corr > 0:
            for y in range(corner_size_h):
                for x in range(width - corner_size_w, width):
                    dx = width - 1 - x
                    y_ratio = y / corner_size_h
                    x_ratio = dx / corner_size_w
                    dist = np.sqrt(y_ratio ** 2 + x_ratio ** 2)
                    if dist < 1.0:
                        corner_grad = (1.0 - dist) ** self.blend_falloff
                        base_corr = max(top_corr, right_corr)
                        val = corner_grad * base_corr * corner_boost
                        mask[y, x] = max(mask[y, x], val)

        # Bottom-left corner
        if bottom_corr > 0 or left_corr > 0:
            for y in range(height - corner_size_h, height):
                dy = height - 1 - y
                for x in range(corner_size_w):
                    y_ratio = dy / corner_size_h
                    x_ratio = x / corner_size_w
                    dist = np.sqrt(y_ratio ** 2 + x_ratio ** 2)
                    if dist < 1.0:
                        corner_grad = (1.0 - dist) ** self.blend_falloff
                        base_corr = max(bottom_corr, left_corr)
                        val = corner_grad * base_corr * corner_boost
                        mask[y, x] = max(mask[y, x], val)

        # Bottom-right corner
        if bottom_corr > 0 or right_corr > 0:
            for y in range(height - corner_size_h, height):
                dy = height - 1 - y
                for x in range(width - corner_size_w, width):
                    dx = width - 1 - x
                    y_ratio = dy / corner_size_h
                    x_ratio = dx / corner_size_w
                    dist = np.sqrt(y_ratio ** 2 + x_ratio ** 2)
                    if dist < 1.0:
                        corner_grad = (1.0 - dist) ** self.blend_falloff
                        base_corr = max(bottom_corr, right_corr)
                        val = corner_grad * base_corr * corner_boost
                        mask[y, x] = max(mask[y, x], val)

        # Clip mask to [0, 1]
        mask = np.clip(mask, 0, 1)

        return mask

    def apply_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply shadow correction to the image.

        Args:
            image: Input image (BGR)

        Returns:
            Corrected image
        """
        height, width = image.shape[:2]

        # Analyze border brightness
        result = self.analyze_border_brightness(image)
        ref_brightness, top, bottom, left, right = result

        # Check if correction is needed
        min_edge = min(top, bottom, left, right)
        if min_edge / ref_brightness >= self.detection_threshold:
            logger.info(
                "No significant edge shadows detected, skipping correction"
            )
            return image

        # Create correction mask
        mask = self.create_correction_mask(
            height, width, ref_brightness, (top, bottom, left, right)
        )

        # Apply correction
        # Convert mask to 3-channel for color images
        if len(image.shape) == 3:
            mask_3d = np.stack([mask] * 3, axis=2)
        else:
            mask_3d = mask

        # Calculate brightness boost
        boost = mask_3d * self.max_brightness_boost

        # Apply boost (additive)
        corrected = image.astype(np.float32) + boost
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        affected_pct = np.mean(mask > 0.1) * 100
        logger.info(
            f"Shadow correction: boost={self.max_brightness_boost}, "
            f"affected={affected_pct:.1f}%"
        )

        return corrected


def get_shadow_correction_config(config: Dict) -> Tuple[bool, Dict]:
    """
    Extract shadow correction parameters from config dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (enabled, params_dict)
    """
    shadow_config = config.get('shadow_correction', {})
    enabled = shadow_config.get('enabled', False)

    params = {
        'border_region_percent': shadow_config.get(
            'border_region_percent', 12.0
        ),
        'corner_region_percent': shadow_config.get(
            'corner_region_percent', 15.0
        ),
        'max_brightness_boost': shadow_config.get(
            'max_brightness_boost', 40
        ),
        'detection_threshold': shadow_config.get(
            'detection_threshold', 0.85
        ),
        'blend_falloff': shadow_config.get('blend_falloff', 0.7)
    }

    return enabled, params
