"""
Visualization Module
====================
Fungsi-fungsi untuk menampilkan hasil deteksi gerakan dengan UI modern.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from .motion_detector import MotionRegion
import math
import time


# === Warna Modern ===
COLORS = {
    "primary": (0, 255, 200),      # Cyan/Teal
    "secondary": (255, 100, 100),   # Coral
    "accent": (200, 100, 255),      # Purple
    "success": (100, 255, 100),     # Green
    "warning": (0, 200, 255),       # Orange/Yellow
    "danger": (0, 0, 255),          # Red
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
}


def draw_motion_regions(
    frame: np.ndarray,
    regions: List[MotionRegion],
    box_color: Optional[Tuple[int, int, int]] = None,
    box_thickness: int = 2,
    show_area: bool = True,
    glow_effect: bool = True,
    animated_corners: bool = True
) -> np.ndarray:
    """
    Gambar bounding box modern di setiap area gerakan.
    
    Features:
    - Glow effect untuk visibility
    - Animated corners (bracket style)
    - Color-coded berdasarkan ukuran area
    """
    output = frame.copy()
    
    for region in regions:
        x, y, w, h = region.bounding_box
        
        # Color based on area size
        if box_color is None:
            if region.area > 10000:
                color = COLORS["danger"]
            elif region.area > 3000:
                color = COLORS["warning"]
            else:
                color = COLORS["primary"]
        else:
            color = box_color
        
        # Glow effect (larger, more transparent box behind)
        if glow_effect:
            overlay = output.copy()
            cv2.rectangle(overlay, (x-3, y-3), (x+w+3, y+h+3), color, 4)
            output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
        
        # Main box
        cv2.rectangle(output, (x, y), (x+w, y+h), color, box_thickness)
        
        # Animated corners (bracket style)
        if animated_corners:
            corner_len = min(20, w // 4, h // 4)
            t = 2  # thickness
            
            # Top-left
            cv2.line(output, (x, y), (x + corner_len, y), COLORS["white"], t)
            cv2.line(output, (x, y), (x, y + corner_len), COLORS["white"], t)
            
            # Top-right
            cv2.line(output, (x+w, y), (x+w - corner_len, y), COLORS["white"], t)
            cv2.line(output, (x+w, y), (x+w, y + corner_len), COLORS["white"], t)
            
            # Bottom-left
            cv2.line(output, (x, y+h), (x + corner_len, y+h), COLORS["white"], t)
            cv2.line(output, (x, y+h), (x, y+h - corner_len), COLORS["white"], t)
            
            # Bottom-right
            cv2.line(output, (x+w, y+h), (x+w - corner_len, y+h), COLORS["white"], t)
            cv2.line(output, (x+w, y+h), (x+w, y+h - corner_len), COLORS["white"], t)
        
        # Area label dengan background
        if show_area:
            text = f"{region.area:,}px"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            # Label background
            cv2.rectangle(output, (x, y - th - 8), (x + tw + 8, y), color, -1)
            cv2.putText(output, text, (x + 4, y - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["white"], 1)
    
    return output


def draw_contours(
    frame: np.ndarray,
    regions: List[MotionRegion],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    fill_alpha: float = 0.2
) -> np.ndarray:
    """Gambar kontur dengan optional fill."""
    output = frame.copy()
    contours = [r.contour for r in regions]
    
    # Fill dengan transparansi
    if fill_alpha > 0:
        overlay = output.copy()
        cv2.drawContours(overlay, contours, -1, color, -1)
        output = cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0)
    
    cv2.drawContours(output, contours, -1, color, thickness)
    return output


def create_debug_view(
    original: np.ndarray,
    diff: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray
) -> np.ndarray:
    """
    Buat tampilan 4 panel untuk debugging dengan label modern.
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
    
    # Tambah label pada setiap panel
    panels = [
        (orig_small, "ORIGINAL", COLORS["white"]),
        (diff_small, "DIFFERENCE", COLORS["warning"]),
        (mask_small, "MOTION MASK", COLORS["primary"]),
        (result_small, "RESULT", COLORS["success"]),
    ]
    
    labeled = []
    for panel, label, color in panels:
        p = panel.copy()
        # Label background
        cv2.rectangle(p, (0, 0), (120, 25), (0, 0, 0), -1)
        cv2.putText(p, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        labeled.append(p)
    
    # Gabungkan
    top = np.hstack([labeled[0], labeled[1]])
    bottom = np.hstack([labeled[2], labeled[3]])
    combined = np.vstack([top, bottom])
    
    return combined


def add_status_bar(
    frame: np.ndarray,
    motion_count: int,
    fps: float = 0.0,
    is_webcam: bool = False,
    recording: bool = False,
    heatmap_on: bool = False,
    tracking_on: bool = False,
    paused: bool = False
) -> np.ndarray:
    """
    Tambah status bar modern di bagian atas frame.
    
    Menampilkan:
    - Motion count dengan indikator
    - FPS
    - Status recording, heatmap, tracking
    """
    output = frame.copy()
    h, w = output.shape[:2]
    bar_height = 40
    
    # Background gradient-like effect
    overlay = output.copy()
    for i in range(bar_height):
        alpha = 1 - (i / bar_height) * 0.5
        cv2.line(overlay, (0, i), (w, i), (20, 20, 30), 1)
    output = cv2.addWeighted(overlay, 0.8, output, 0.2, 0)
    
    # Motion indicator (pulsing circle when motion detected)
    indicator_color = COLORS["success"] if motion_count > 0 else COLORS["gray"]
    cv2.circle(output, (20, 20), 8, indicator_color, -1)
    cv2.circle(output, (20, 20), 8, COLORS["white"], 1)
    
    # Motion count
    motion_text = f"MOTION: {motion_count}"
    cv2.putText(output, motion_text, (35, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["white"], 1)
    
    # FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(output, fps_text, (170, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["primary"], 1)
    
    # Status indicators (right side)
    x_pos = w - 20
    indicators = []
    
    if recording:
        indicators.append(("REC", COLORS["danger"]))
    if heatmap_on:
        indicators.append(("HEAT", COLORS["warning"]))
    if tracking_on:
        indicators.append(("TRACK", COLORS["accent"]))
    if paused:
        indicators.append(("PAUSED", COLORS["warning"]))
    if is_webcam:
        indicators.append(("LIVE", COLORS["success"]))
    
    for label, color in reversed(indicators):
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x_pos -= tw + 15
        cv2.rectangle(output, (x_pos - 3, 8), (x_pos + tw + 3, 32), color, -1)
        cv2.putText(output, label, (x_pos, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["black"], 1)
    
    return output


def add_keyboard_legend(
    frame: np.ndarray,
    show: bool = True,
    position: str = "bottom"
) -> np.ndarray:
    """
    Tambah legend keyboard shortcuts di frame.
    """
    if not show:
        return frame
    
    output = frame.copy()
    h, w = output.shape[:2]
    
    shortcuts = [
        ("Q", "Quit"),
        ("D", "Debug"),
        ("H", "Heatmap"),
        ("T", "Track"),
        ("S", "Snapshot"),
        ("R", "Record"),
        ("Space", "Pause"),
    ]
    
    # Calculate legend size
    legend_h = 30
    start_y = h - legend_h - 5 if position == "bottom" else 45
    
    # Background
    overlay = output.copy()
    cv2.rectangle(overlay, (0, start_y), (w, start_y + legend_h), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.6, output, 0.4, 0)
    
    # Draw shortcuts
    x = 10
    y = start_y + 20
    for key, desc in shortcuts:
        text = f"[{key}] {desc}"
        cv2.putText(output, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["gray"], 1)
        tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        x += tw + 15
        if x > w - 100:
            break
    
    return output


def add_progress_bar(
    frame: np.ndarray,
    progress: float,
    height: int = 5
) -> np.ndarray:
    """
    Tambah progress bar untuk video playback.
    
    Parameters
    ----------
    progress : float
        Progress 0.0 - 1.0
    """
    output = frame.copy()
    h, w = output.shape[:2]
    
    # Background
    cv2.rectangle(output, (0, h - height), (w, h), COLORS["gray"], -1)
    
    # Progress
    progress_w = int(w * min(1.0, max(0.0, progress)))
    cv2.rectangle(output, (0, h - height), (progress_w, h), COLORS["primary"], -1)
    
    return output


def draw_center_message(
    frame: np.ndarray,
    message: str,
    submessage: str = "",
    color: Tuple[int, int, int] = None
) -> np.ndarray:
    """
    Gambar pesan besar di tengah frame.
    Berguna untuk PAUSED, RECORDING STARTED, dll.
    """
    output = frame.copy()
    h, w = output.shape[:2]
    
    if color is None:
        color = COLORS["white"]
    
    # Main message
    font_scale = 1.5
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    x = (w - tw) // 2
    y = (h + th) // 2
    
    # Shadow
    cv2.putText(output, message, (x+2, y+2), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLORS["black"], 3)
    cv2.putText(output, message, (x, y),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    
    # Submessage
    if submessage:
        (tw2, th2), _ = cv2.getTextSize(submessage, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        x2 = (w - tw2) // 2
        cv2.putText(output, submessage, (x2, y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["gray"], 1)
    
    return output

