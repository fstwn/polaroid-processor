"""
Polaroid Image Processor
Automatically crops and corrects high-resolution Polaroid scans.
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Optional, Dict
import logging

# Disable PIL decompression bomb warning for large TIFF files
Image.MAX_IMAGE_PIXELS = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolaroidProcessor:
    """
    Processes high-resolution Polaroid scans to extract and correct
    the Polaroid image.
    """

    # Default Polaroid dimensions in mm (width x height)
    DEFAULT_WIDTH_MM = 88
    DEFAULT_HEIGHT_MM = 107

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the processor.

        Args:
            config: Configuration dictionary (from YAML file)
        """
        config = config or {}

        # Get image settings from config
        image_config = config.get('image', {})
        self.dpi = image_config.get('dpi', 600)
        self.polaroid_width_mm = image_config.get(
            'target_width_mm', self.DEFAULT_WIDTH_MM
        )
        self.polaroid_height_mm = image_config.get(
            'target_height_mm', self.DEFAULT_HEIGHT_MM
        )
        self.aspect_ratio = self.polaroid_width_mm / self.polaroid_height_mm

        # Calculate target dimensions in pixels
        self.target_width_px = int(self.polaroid_width_mm * self.dpi / 25.4)
        self.target_height_px = int(self.polaroid_height_mm * self.dpi / 25.4)

        # Get JPG output settings from config
        jpg_config = config.get('jpg_output', {})
        self.jpg_dpi = jpg_config.get('dpi', 350)
        self.jpg_size_multiplier = jpg_config.get('size_multiplier', 2.0)
        self.jpg_quality = jpg_config.get('quality', 95)
        self.jpg_width_px = int(
            self.polaroid_width_mm * self.jpg_size_multiplier *
            self.jpg_dpi / 25.4
        )
        self.jpg_height_px = int(
            self.polaroid_height_mm * self.jpg_size_multiplier *
            self.jpg_dpi / 25.4
        )

        # Get preprocessing settings
        preprocess_config = config.get('preprocessing', {})
        self.threshold_value = preprocess_config.get('threshold_value', 100)

        # Get contour detection settings
        contour_config = config.get('contour_detection', {})
        self.min_contour_area = contour_config.get('min_area', 10000)
        self.min_aspect_ratio = contour_config.get('min_aspect_ratio', 0.7)
        self.max_aspect_ratio = contour_config.get('max_aspect_ratio', 1.0)

        logger.info(
            f"Polaroid dimensions: {self.polaroid_width_mm}"
            f"x{self.polaroid_height_mm}mm "
            f"(aspect ratio: {self.aspect_ratio:.3f})"
        )
        logger.info(
            f"Target output size: {self.target_width_px}"
            f"x{self.target_height_px} pixels"
        )
        logger.info(
            f"JPG output: {self.jpg_width_px}x{self.jpg_height_px} pixels "
            f"at {self.jpg_dpi} DPI ({self.jpg_size_multiplier}x size)"
        )

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image using PIL and convert to OpenCV format."""
        pil_image = Image.open(image_path)
        # Convert PIL to OpenCV format (RGB -> BGR)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return cv_image

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for edge detection - find white Polaroid
        on black background.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Simple thresholding - white Polaroid should be
        # much brighter than background
        if isinstance(self.threshold_value, str) and self.threshold_value == 'OTSU':
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_OTSU
            )
        else: 
            _, thresh = cv2.threshold(
                gray, self.threshold_value, 255, cv2.THRESH_BINARY
            )

        return thresh

    def find_polaroid_contour(
        self, processed_image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Find the white Polaroid rectangle on black background.
        """
        # Find contours
        contours, _ = cv2.findContours(
            processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Get image dimensions
        img_height, img_width = processed_image.shape
        img_area = img_height * img_width

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # The largest white area should be the Polaroid
        for contour in contours:
            area = cv2.contourArea(contour)

            # Skip very small contours (use config value)
            if area < self.min_contour_area:
                continue

            # Skip contours that are too large (likely the full image)
            if area > img_area * 0.95:
                continue

            # Approximate the contour to a polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if it's roughly rectangular (4 corners)
            if len(approx) == 4:
                # Check aspect ratio (use config values)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h

                ar_ok = (self.min_aspect_ratio <= aspect_ratio <=
                         self.max_aspect_ratio)
                if ar_ok:
                    print(
                        f"Found potential Polaroid: area={area:.0f}, "
                        f"size={w}x{h}, aspect={aspect_ratio:.3f}"
                    )
                    return approx

        # If no 4-corner contour found, try with more corners and simplify
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area or area > img_area * 0.95:
                continue

            # Try different epsilon values for approximation
            for eps_factor in [0.005, 0.01, 0.02, 0.03]:
                epsilon = eps_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) >= 4:
                    # Check if it's roughly rectangular
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h

                    ar_ok = (self.min_aspect_ratio <=
                             aspect_ratio <=
                             self.max_aspect_ratio)
                    if ar_ok:
                        print(
                            "Found potential Polaroid (simplified): "
                            f"area={area:.0f}, "
                            f"size={w}x{h}, aspect={aspect_ratio:.3f}, "
                            f"corners={len(approx)}"
                        )
                        return approx

        print("No suitable Polaroid contour found")
        return None

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in the format:
        top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype="float32")

        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def apply_safety_margin_to_corners(
        self,
        corners: np.ndarray,
        margin_percent: float = 0.5
    ) -> np.ndarray:
        """
        Apply a safety margin to corners, moving them slightly inward.
        This helps avoid black edge artifacts from imperfect detection.

        Args:
            corners: Original corner points (4x2 array, ordered)
            margin_percent: Percentage to move corners inward (0.5 = 0.5%)

        Returns:
            Adjusted corner points
        """
        if margin_percent <= 0:
            return corners

        # Calculate the center of the quadrilateral
        center = np.mean(corners, axis=0)

        # Move each corner slightly toward the center
        margin_factor = margin_percent / 100.0
        adjusted_corners = np.zeros_like(corners)

        for i in range(4):
            # Vector from corner to center
            to_center = center - corners[i]
            # Move corner inward by margin_factor of the distance
            adjustment = to_center * margin_factor
            adjusted_corners[i] = corners[i] + adjustment

        logger.info(
            f"Applied safety margin: {margin_percent}% inward adjustment"
        )
        return adjusted_corners

    def perspective_correct(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        safety_margin_percent: float = 0.0
    ) -> np.ndarray:
        """
        Apply perspective correction to align the Polaroid properly.

        Args:
            image: Input image
            corners: Detected corner points
            safety_margin_percent: Margin to crop inward (0-2% typical)

        Returns:
            Perspective-corrected image
        """
        # Order the corner points
        ordered_corners = self.order_points(corners.reshape(4, 2))

        # Apply safety margin if specified
        if safety_margin_percent > 0:
            ordered_corners = self.apply_safety_margin_to_corners(
                ordered_corners, safety_margin_percent
            )

        # Calculate the actual dimensions of the detected Polaroid
        width1 = np.linalg.norm(ordered_corners[0] - ordered_corners[1])
        width2 = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
        height1 = np.linalg.norm(ordered_corners[0] - ordered_corners[3])
        height2 = np.linalg.norm(ordered_corners[1] - ordered_corners[2])

        # Use the average dimensions
        avg_width = int((width1 + width2) / 2)
        avg_height = int((height1 + height2) / 2)

        # Ensure the aspect ratio is correct (use config values)
        if avg_width / avg_height > self.aspect_ratio:
            # Too wide, adjust height
            target_height = int(avg_width / self.aspect_ratio)
            target_width = avg_width
        else:
            # Too tall, adjust width
            target_width = int(avg_height * self.aspect_ratio)
            target_height = avg_height

        # Define destination points (target rectangle with correct AR)
        dst = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype="float32")

        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners, dst)

        # Apply perspective transformation
        corrected = cv2.warpPerspective(
            image, matrix, (target_width, target_height)
        )

        logger.info(
            f"Perspective correction: {image.shape} -> {corrected.shape}"
        )
        return corrected

    def create_jpg_output(self, corrected_image: np.ndarray) -> np.ndarray:
        """
        Create JPG output at configured DPI and size.
        """
        # Resize to JPG dimensions
        jpg_resized = cv2.resize(
            corrected_image,
            (self.jpg_width_px, self.jpg_height_px),
            interpolation=cv2.INTER_LANCZOS4
        )

        logger.info(
            f"JPG resize: {corrected_image.shape} -> {jpg_resized.shape}"
        )
        return jpg_resized

    def process_image(
        self, input_path: str, output_path: str, create_jpg: bool = True
    ) -> bool:
        """
        Process a single Polaroid scan.

        Args:
            input_path: Path to input TIFF file
            output_path: Path to save processed image
            create_jpg: Whether to also create JPG output

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing: {input_path}")

            # Load image
            image = self.load_image(input_path)
            original_height, original_width = image.shape[:2]
            logger.info(f"Original size: {original_width}x{original_height}")

            # Preprocess for edge detection
            processed = self.preprocess_image(image)
            logger.info(
                f"Preprocessing complete, thresh shape: {processed.shape}"
            )

            # Find Polaroid contour
            corners = self.find_polaroid_contour(processed)
            if corners is None:
                logger.error(f"Could not detect Polaroid in {input_path}")
                # Save debug image to help diagnose
                debug_path = output_path.replace('.tif', '_debug_thresh.tif')
                cv2.imwrite(debug_path, processed)
                logger.info(
                    f"Saved thresholded image for debugging: {debug_path}"
                )
                return False

            logger.info(f"Found Polaroid corners: {len(corners)} points")

            # Apply perspective correction
            corrected = self.perspective_correct(image, corners)
            logger.info(
                "Perspective correction complete, "
                f"output size: {corrected.shape}"
            )

            # Convert back to PIL and save TIFF
            corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(corrected_rgb)

            # Save as TIFF to preserve quality
            pil_image.save(output_path, "TIFF", compression="lzw")
            logger.info(f"Successfully processed TIFF: {output_path}")

            # Create JPG output if requested
            if create_jpg:
                jpg_output = self.create_jpg_output(corrected)
                jpg_rgb = cv2.cvtColor(jpg_output, cv2.COLOR_BGR2RGB)
                jpg_pil = Image.fromarray(jpg_rgb)

                # Create JPG filename
                jpg_path = output_path.replace('.tif', '.jpg').replace(
                    '.tiff', '.jpg'
                )
                jpg_pil.save(
                    jpg_path, "JPEG",
                    quality=self.jpg_quality,
                    dpi=(self.jpg_dpi, self.jpg_dpi)
                )
                logger.info(f"Successfully processed JPG: {jpg_path}")

            return True

        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def process_batch(
        self, input_dir: str, output_dir: str, create_jpg: bool = True
    ) -> None:
        """
        Process all TIFF files in a directory.

        Args:
            input_dir: Directory containing input TIFF files
            output_dir: Directory to save processed images
            create_jpg: Whether to also create JPG output
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Find all TIFF files
        tiff_files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.tif', '.tiff'))
        ]

        if not tiff_files:
            logger.warning(f"No TIFF files found in {input_dir}")
            return

        logger.info(f"Found {len(tiff_files)} TIFF files to process")

        successful = 0
        failed = 0

        for filename in tiff_files:
            input_path = os.path.join(input_dir, filename)
            output_filename = f"processed_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            if self.process_image(input_path, output_path, create_jpg):
                successful += 1
            else:
                failed += 1

        logger.info(
            f"Batch processing complete: {successful} "
            f"successful, {failed} failed"
        )
