"""
Motion Heatmap Module
=====================

Modul untuk membuat visualisasi heatmap dari area yang sering ada gerakan.
Berguna untuk analisis pola gerakan jangka panjang.

Konsep:
- Akumulasi mask gerakan dari waktu ke waktu
- Aplikasikan decay untuk gradual fade
- Visualisasi dengan colormap (cold → hot)
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class MotionHeatmap:
    """
    Membuat heatmap akumulatif dari deteksi gerakan.
    
    Cara kerja:
    1. Setiap frame, tambahkan mask gerakan ke accumulator
    2. Aplikasikan decay agar gerakan lama memudar
    3. Normalize dan aplikasikan colormap untuk visualisasi
    """
    
    def __init__(
        self,
        shape: Optional[Tuple[int, int]] = None,
        decay_rate: float = 0.995,
        intensity: float = 5.0,
        colormap: int = cv2.COLORMAP_JET
    ):
        """
        Parameters
        ----------
        shape : Optional[Tuple[int, int]]
            Ukuran heatmap (height, width). Jika None, akan di-set otomatis
        decay_rate : float
            Rate decay per frame (0.99 = 1% decay per frame)
        intensity : float
            Multiplier untuk intensitas gerakan
        colormap : int
            OpenCV colormap untuk visualisasi
        """
        self.decay_rate = decay_rate
        self.intensity = intensity
        self.colormap = colormap
        
        self.accumulator: Optional[np.ndarray] = None
        self.shape = shape
    
    def update(self, motion_mask: np.ndarray) -> None:
        """
        Update heatmap dengan mask gerakan baru.
        
        Parameters
        ----------
        motion_mask : np.ndarray
            Binary mask dari gerakan (nilai 0 atau 255)
        """
        # Inisialisasi accumulator jika belum ada
        if self.accumulator is None:
            self.shape = motion_mask.shape[:2]
            self.accumulator = np.zeros(self.shape, dtype=np.float32)
        
        # Resize mask jika ukuran berbeda
        if motion_mask.shape[:2] != self.shape:
            motion_mask = cv2.resize(motion_mask, (self.shape[1], self.shape[0]))
        
        # Normalize mask ke 0-1
        normalized = motion_mask.astype(np.float32) / 255.0
        
        # Decay yang ada
        self.accumulator *= self.decay_rate
        
        # Tambahkan gerakan baru
        self.accumulator += normalized * self.intensity
        
        # Clamp ke range yang reasonable
        np.clip(self.accumulator, 0, 255, out=self.accumulator)
    
    def get_heatmap(self, normalized: bool = True) -> np.ndarray:
        """
        Dapatkan heatmap sebagai image BGR.
        
        Parameters
        ----------
        normalized : bool
            Jika True, normalize ke full range untuk kontras maksimal
        
        Returns
        -------
        np.ndarray
            Heatmap dalam format BGR
        """
        if self.accumulator is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        if normalized and self.accumulator.max() > 0:
            display = (self.accumulator / self.accumulator.max() * 255).astype(np.uint8)
        else:
            display = np.clip(self.accumulator, 0, 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(display, self.colormap)
        
        return colored
    
    def overlay_on_frame(
        self,
        frame: np.ndarray,
        alpha: float = 0.5,
        threshold: float = 10.0
    ) -> np.ndarray:
        """
        Overlay heatmap pada frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Frame BGR asli
        alpha : float
            Transparansi overlay (0 = tidak terlihat, 1 = solid)
        threshold : float
            Minimum nilai heatmap untuk ditampilkan
        
        Returns
        -------
        np.ndarray
            Frame dengan heatmap overlay
        """
        if self.accumulator is None:
            return frame
        
        # Dapatkan heatmap berwarna
        heatmap = self.get_heatmap(normalized=False)
        
        # Resize jika perlu
        if heatmap.shape[:2] != frame.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # Buat mask berdasarkan threshold
        mask = self.accumulator > threshold
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0])
            ).astype(bool)
        
        # Blend hanya di area yang memiliki gerakan
        output = frame.copy()
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        
        output = np.where(
            mask_3ch,
            cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0),
            frame
        )
        
        return output.astype(np.uint8)
    
    def reset(self) -> None:
        """Reset heatmap ke kosong."""
        if self.accumulator is not None:
            self.accumulator.fill(0)
    
    def get_hotspots(
        self,
        top_n: int = 5,
        min_distance: int = 50
    ) -> list:
        """
        Dapatkan titik-titik dengan aktivitas tertinggi.
        
        Parameters
        ----------
        top_n : int
            Jumlah hotspot yang dikembalikan
        min_distance : int
            Jarak minimum antara hotspot
        
        Returns
        -------
        list
            List of (x, y, intensity) tuples
        """
        if self.accumulator is None:
            return []
        
        hotspots = []
        temp = self.accumulator.copy()
        
        for _ in range(top_n):
            # Cari nilai maksimum
            max_val = temp.max()
            if max_val <= 0:
                break
            
            # Cari lokasi
            y, x = np.unravel_index(temp.argmax(), temp.shape)
            hotspots.append((int(x), int(y), float(max_val)))
            
            # Zero-out area sekitar untuk mencari hotspot berikutnya
            y_min = max(0, y - min_distance)
            y_max = min(temp.shape[0], y + min_distance)
            x_min = max(0, x - min_distance)
            x_max = min(temp.shape[1], x + min_distance)
            temp[y_min:y_max, x_min:x_max] = 0
        
        return hotspots


def create_heatmap_legend(
    height: int = 20,
    width: int = 200,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Buat legend bar untuk heatmap.
    
    Returns
    -------
    np.ndarray
        Image BGR untuk legend
    """
    gradient = np.linspace(0, 255, width).astype(np.uint8)
    gradient = np.tile(gradient, (height, 1))
    legend = cv2.applyColorMap(gradient, colormap)
    
    # Tambah label
    cv2.putText(legend, "Low", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(legend, "High", (width - 35, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return legend
