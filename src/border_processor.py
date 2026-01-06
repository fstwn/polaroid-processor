"""
Border processing for removing dark edges and creating clean white frames.
"""

import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class BorderProcessor:
    """
    Border processor that detects and removes dark edges,
    then adds clean white borders.
    """

    def __init__(self,
                 border_threshold: int = 50,
                 min_border_width: int = 5,
                 padding_method: str = 'white',
                 synthesis_sample_width: int = 20,
                 content_threshold: float = 0.1):
        """
        Initialize border processor.

        Args:
            border_threshold: Maximum pixel value to consider as dark
            min_border_width: Minimum width of border to process
            padding_method: Method for padding ('white', 'synthesize', etc.)
            synthesis_sample_width: Width of border area to sample
            content_threshold: Min fraction of row/column with content
        """
        self.border_threshold = border_threshold
        self.min_border_width = min_border_width
        self.padding_method = padding_method
        self.synthesis_sample_width = synthesis_sample_width
        self.content_threshold = content_threshold

    def detect_black_borders(
            self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect dark borders by finding where actual content starts.
        Uses gradient-based approach to find content edges accurately.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Tuple of (top, bottom, left, right) content boundaries
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape

        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Calculate gradient to find strong edges (content boundaries)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold for strong edges
        edge_threshold = np.percentile(gradient_magnitude, 75)

        # Find top boundary - scan from top, look for content
        top = 0
        for row in range(height):
            # Check both brightness (content) and edge strength
            row_pixels = blurred[row, :]
            row_edges = gradient_magnitude[row, :]

            # Count pixels that are either bright enough or have edges
            bright_pixels = np.sum(row_pixels > self.border_threshold)
            strong_edges = np.sum(row_edges > edge_threshold)

            # Content found when we have enough bright pixels OR edges
            threshold_pixels = width * (1 - self.content_threshold)
            if bright_pixels > threshold_pixels or strong_edges > width * 0.1:
                # Make sure we're past any initial dark border
                if row_pixels.mean() > self.border_threshold:
                    top = row
                    break

        # Find bottom boundary - scan from bottom
        bottom = height
        for row in range(height - 1, -1, -1):
            row_pixels = blurred[row, :]
            row_edges = gradient_magnitude[row, :]

            bright_pixels = np.sum(row_pixels > self.border_threshold)
            strong_edges = np.sum(row_edges > edge_threshold)

            threshold_pixels = width * (1 - self.content_threshold)
            if bright_pixels > threshold_pixels or strong_edges > width * 0.1:
                if row_pixels.mean() > self.border_threshold:
                    bottom = row + 1
                    break

        # Find left boundary - scan from left
        left = 0
        for col in range(width):
            col_pixels = blurred[:, col]
            col_edges = gradient_magnitude[:, col]

            bright_pixels = np.sum(col_pixels > self.border_threshold)
            strong_edges = np.sum(col_edges > edge_threshold)

            threshold_pixels = height * (1 - self.content_threshold)
            if bright_pixels > threshold_pixels or strong_edges > height * 0.1:
                if col_pixels.mean() > self.border_threshold:
                    left = col
                    break

        # Find right boundary - scan from right
        right = width
        for col in range(width - 1, -1, -1):
            col_pixels = blurred[:, col]
            col_edges = gradient_magnitude[:, col]

            bright_pixels = np.sum(col_pixels > self.border_threshold)
            strong_edges = np.sum(col_edges > edge_threshold)

            threshold_pixels = height * (1 - self.content_threshold)
            if bright_pixels > threshold_pixels or strong_edges > height * 0.1:
                if col_pixels.mean() > self.border_threshold:
                    right = col + 1
                    break

        # Ensure we have valid boundaries
        if bottom <= top or right <= left:
            logger.warning(
                "No valid content detected, returning original bounds")
            return 0, height, 0, width

        # Add small safety margin to ensure no border pixels
        margin = 2
        top = max(0, top - margin)
        bottom = min(height, bottom + margin)
        left = max(0, left - margin)
        right = min(width, right + margin)

        return top, bottom, left, right

    def create_white_border(self, content: np.ndarray,
                            target_size: Tuple[int, int]) -> np.ndarray:
        """
        Create an image with clean white borders around content.
        Ensures even padding when possible.

        Args:
            content: Content image (without borders)
            target_size: Target (width, height) for output

        Returns:
            Image with white borders
        """
        content_height, content_width = content.shape[:2]
        target_width, target_height = target_size

        # Calculate padding needed - ensure even padding when possible
        total_pad_height = target_height - content_height
        total_pad_width = target_width - content_width

        # Distribute padding evenly
        pad_top = total_pad_height // 2
        pad_left = total_pad_width // 2

        # Create output image filled with white
        if len(content.shape) == 3:
            output = np.full(
                (target_height, target_width, content.shape[2]),
                255, dtype=content.dtype)
        else:
            output = np.full(
                (target_height, target_width), 255, dtype=content.dtype)

        # Place content in center
        output[pad_top:pad_top + content_height,
               pad_left:pad_left + content_width] = content

        return output

    def synthesize_border_pixels(
            self, image: np.ndarray, target_size: Tuple[int, int],
            content_region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create borders using specified method.

        Args:
            image: Original image
            target_size: Target (width, height) for output
            content_region: (top, bottom, left, right) of content area

        Returns:
            Image with borders
        """
        top, bottom, left, right = content_region
        target_width, target_height = target_size

        # Extract content
        content = image[top:bottom, left:right]
        content_height, content_width = content.shape[:2]

        # Calculate padding needed
        total_pad_height = target_height - content_height
        total_pad_width = target_width - content_width

        if total_pad_height < 0 or total_pad_width < 0:
            logger.warning(
                "Target size smaller than content, returning cropped")
            return content

        if total_pad_height == 0 and total_pad_width == 0:
            return content

        # Use white borders by default for clean Polaroid look
        if (self.padding_method == 'white' or
                self.padding_method == 'synthesize'):
            return self.create_white_border(content, target_size)

        # Create output image
        if len(image.shape) == 3:
            output = np.zeros(
                (target_height, target_width, image.shape[2]),
                dtype=image.dtype)
        else:
            output = np.zeros((target_height, target_width),
                              dtype=image.dtype)

        # Distribute padding evenly - center the content
        pad_top = total_pad_height // 2
        pad_left = total_pad_width // 2

        # Place content in center
        output[pad_top:pad_top + content_height,
               pad_left:pad_left + content_width] = content

        # For black padding, output is already zeros
        if self.padding_method == 'black':
            pass  # Already initialized with zeros

        return output

    def process_borders(
            self, image: np.ndarray, target_aspect_ratio: float = 0.822,
            force_aspect_ratio: bool = True) -> np.ndarray:
        """
        Process dark borders in the image and add clean white frame.

        Args:
            image: Input image
            target_aspect_ratio: Target aspect ratio to maintain
            force_aspect_ratio: Whether to force the exact aspect ratio

        Returns:
            Processed image with borders handled
        """
        # Detect dark borders and find content region
        top, bottom, left, right = self.detect_black_borders(image)

        # Check if we have significant borders to process
        content_height = bottom - top
        content_width = right - left

        if (content_height < self.min_border_width or
                content_width < self.min_border_width):
            logger.info("No significant borders detected")
            return image

        # Extract content without borders
        content = image[top:bottom, left:right]

        if not force_aspect_ratio:
            return content

        # Calculate target dimensions maintaining aspect ratio
        current_ratio = content_width / content_height

        if abs(current_ratio - target_aspect_ratio) < 0.01:
            # Already correct aspect ratio - add small even border
            border_size = max(10, int(min(content_width, content_height) *
                                      0.02))
            target_height = content_height + 2 * border_size
            target_width = content_width + 2 * border_size
        elif current_ratio > target_aspect_ratio:
            # Content is too wide, need to add height
            target_height = int(content_width / target_aspect_ratio)
            target_width = content_width
        else:
            # Content is too tall, need to add width
            target_width = int(content_height * target_aspect_ratio)
            target_height = content_height

        # Create image with clean white borders
        result = self.synthesize_border_pixels(
            image, (target_width, target_height),
            (top, bottom, left, right))

        logger.info(f"Border processing: {image.shape} -> {result.shape}")
        return result

    def crop_to_content(self, image: np.ndarray) -> np.ndarray:
        """
        Simple crop to remove dark borders without aspect ratio correction.

        Args:
            image: Input image

        Returns:
            Cropped image
        """
        top, bottom, left, right = self.detect_black_borders(image)
        return image[top:bottom, left:right]
