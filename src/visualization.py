"""
Visualization Module
====================
Fungsi-fungsi untuk menampilkan hasil deteksi gerakan.
"""

import cv2
import numpy as np
from typing import List, Tuple
from .motion_detector import MotionRegion


def draw_motion_regions(
    frame: np.ndarray,
    regions: List[MotionRegion],
    box_color: Tuple[int, int, int] = (0, 255, 0),
    box_thickness: int = 2,
    show_area: bool = True
) -> np.ndarray:
    """Gambar bounding box di setiap area gerakan."""
    output = frame.copy()
    
    for region in regions:
        x, y, w, h = region.bounding_box
        cv2.rectangle(output, (x, y), (x + w, y + h), box_color, box_thickness)
        
        if show_area:
            text = f"Area: {region.area}"
            cv2.putText(output, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    
    return output


def draw_contours(
    frame: np.ndarray,
    regions: List[MotionRegion],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """Gambar kontur di setiap area gerakan."""
    output = frame.copy()
    contours = [r.contour for r in regions]
    cv2.drawContours(output, contours, -1, color, thickness)
    return output


def create_debug_view(
    original: np.ndarray,
    diff: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray
) -> np.ndarray:
    """
    Buat tampilan 4 panel untuk debugging:
    [Original] [Difference]
    [Mask]     [Result]
    """
    h, w = original.shape[:2]
    
    # Konversi grayscale ke BGR jika perlu
    if len(diff.shape) == 2:
        diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Resize semua ke ukuran yang sama
    size = (w // 2, h // 2)
    orig_small = cv2.resize(original, size)
    diff_small = cv2.resize(diff, size)
    mask_small = cv2.resize(mask, size)
    result_small = cv2.resize(result, size)
    
    # Gabungkan
    top = np.hstack([orig_small, diff_small])
    bottom = np.hstack([mask_small, result_small])
    combined = np.vstack([top, bottom])
    
    # Tambah label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(combined, "Difference", (w//2 + 10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(combined, "Mask", (10, h//2 + 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(combined, "Result", (w//2 + 10, h//2 + 20), font, 0.5, (255, 255, 255), 1)
    
    return combined


def add_status_bar(
    frame: np.ndarray,
    motion_count: int,
    fps: float = 0.0
) -> np.ndarray:
    """Tambah status bar di bagian atas frame."""
    output = frame.copy()
    h, w = output.shape[:2]
    
    # Background hitam semi-transparan
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)
    
    # Text status
    status = f"Motion Regions: {motion_count}"
    if fps > 0:
        status += f" | FPS: {fps:.1f}"
    
    cv2.putText(output, status, (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return output
