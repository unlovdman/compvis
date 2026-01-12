"""
Motion Detector Module
======================
Deteksi area gerakan menggunakan kontur dan bounding box.
"""

import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class MotionRegion:
    """Representasi satu area gerakan yang terdeteksi."""
    x: int
    y: int
    width: int
    height: int
    area: int
    contour: np.ndarray
    
    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class MotionDetector:
    """
    Deteksi gerakan dari foreground mask.
    
    Pipeline:
    1. Threshold → Binary mask
    2. Morphology → Clean mask
    3. Find contours → Detect regions
    4. Filter by area → Remove noise
    """
    
    def __init__(
        self,
        threshold: int = 25,
        min_area: int = 500,
        morph_kernel_size: int = 5
    ):
        self.threshold = threshold
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
    
    def apply_threshold(self, diff: np.ndarray) -> np.ndarray:
        """Binarisasi: piksel > threshold = 255, else 0."""
        _, binary = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Erosion lalu dilation untuk bersihkan noise."""
        eroded = cv2.erode(mask, self.kernel, iterations=1)
        dilated = cv2.dilate(eroded, self.kernel, iterations=2)
        return dilated
    
    def find_motion_regions(self, mask: np.ndarray) -> List[MotionRegion]:
        """Cari kontur dan filter berdasarkan area minimum."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            regions.append(MotionRegion(
                x=x, y=y, width=w, height=h, area=int(area), contour=contour
            ))
        
        return regions
    
    def detect(self, diff_image: np.ndarray) -> Tuple[np.ndarray, List[MotionRegion]]:
        """
        Pipeline lengkap deteksi gerakan.
        
        Returns: (clean_mask, list of MotionRegion)
        """
        binary = self.apply_threshold(diff_image)
        clean = self.apply_morphology(binary)
        regions = self.find_motion_regions(clean)
        return clean, regions
