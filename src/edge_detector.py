"""
Enhanced edge detection utilities for Polaroid processing.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class EdgeDetector:
    """
    Advanced edge detection for finding Polaroid boundaries.
    """
    
    def __init__(self, canny_low: int = 50, canny_high: int = 150):
        """
        Initialize edge detector with Canny parameters.
        
        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny edge detection.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Binary image with detected edges
        """
        # Apply Canny edge detection
        edges = cv2.Canny(image, self.canny_low, self.canny_high)
        
        # Apply morphological operations to clean up edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def find_rectangular_contours(self, edges: np.ndarray, min_area: int = 1000) -> list:
        """
        Find rectangular contours from edge image.
        
        Args:
            edges: Binary edge image
            min_area: Minimum area for contour filtering
            
        Returns:
            List of rectangular contours
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_contours = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) == 4:
                rectangular_contours.append(approx)
        
        return rectangular_contours
    
    def find_largest_rectangle(self, contours: list, target_aspect_ratio: float = 0.822) -> Optional[np.ndarray]:
        """
        Find the largest rectangle that matches the target aspect ratio.
        
        Args:
            contours: List of rectangular contours
            target_aspect_ratio: Expected aspect ratio (width/height)
            
        Returns:
            Best matching rectangular contour or None
        """
        if not contours:
            return None
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Calculate score based on area and aspect ratio match
            area = cv2.contourArea(contour)
            aspect_score = 1.0 - abs(aspect_ratio - target_aspect_ratio) / target_aspect_ratio
            area_score = area / 1000000  # Normalize area
            
            # Combined score (weighted)
            score = 0.7 * area_score + 0.3 * aspect_score
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        return best_contour
    
    def refine_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Refine corner detection using sub-pixel accuracy.
        
        Args:
            image: Original grayscale image
            corners: Initial corner points
            
        Returns:
            Refined corner points
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Refine corners using cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        refined_corners = cv2.cornerSubPix(
            gray, 
            corners.astype(np.float32), 
            (5, 5), 
            (-1, -1), 
            criteria
        )
        
        return refined_corners
