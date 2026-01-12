"""
Video Reader Module
===================

Modul ini menangani pembacaan video dari file.
Menggunakan OpenCV VideoCapture untuk membaca frame per frame.

Konsep Penting:
- Video adalah urutan gambar (frame) yang ditampilkan dengan kecepatan tertentu (FPS)
- Setiap frame adalah array numpy dengan shape (height, width, channels)
- Format warna default OpenCV adalah BGR, bukan RGB
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator


class VideoReader:
    """
    Kelas untuk membaca video dari file.
    
    Menggunakan context manager (with statement) untuk memastikan
    resource video selalu di-release dengan benar.
    
    Contoh penggunaan:
    -----------------
    >>> with VideoReader("video.mp4") as reader:
    ...     for frame in reader.frames():
    ...         process(frame)
    """
    
    def __init__(self, video_path: str):
        """
        Inisialisasi VideoReader.
        
        Parameters
        ----------
        video_path : str
            Path ke file video (mp4, avi, mkv, dll)
        """
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        
    def __enter__(self) -> "VideoReader":
        """
        Dipanggil saat masuk blok 'with'.
        Membuka koneksi ke file video.
        """
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Dipanggil saat keluar blok 'with'.
        Memastikan video selalu di-release.
        """
        self.release()
        
    def open(self) -> bool:
        """
        Membuka file video.
        
        Returns
        -------
        bool
            True jika berhasil membuka video, False jika gagal
        """
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise FileNotFoundError(
                f"Tidak dapat membuka video: {self.video_path}\n"
                "Pastikan path benar dan format video didukung."
            )
        
        return True
    
    def release(self) -> None:
        """
        Melepas resource video.
        Penting untuk menghindari memory leak.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Membaca satu frame dari video.
        
        Returns
        -------
        Tuple[bool, Optional[np.ndarray]]
            - ret: True jika berhasil membaca, False jika video habis
            - frame: Array numpy (H, W, 3) jika berhasil, None jika gagal
        
        Penjelasan:
        - Frame adalah gambar dalam format BGR
        - Jika video sudah habis, ret akan False
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator untuk iterasi semua frame dalam video.
        
        Yields
        ------
        np.ndarray
            Frame video dalam format BGR (H, W, 3)
        
        Contoh:
        -------
        >>> for frame in reader.frames():
        ...     cv2.imshow("Frame", frame)
        """
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            yield frame
            
    @property
    def fps(self) -> float:
        """
        Mendapatkan Frame Per Second video.
        
        Digunakan untuk menghitung delay yang tepat saat menampilkan video.
        """
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    @property
    def frame_count(self) -> int:
        """Total jumlah frame dalam video."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def width(self) -> int:
        """Lebar frame dalam piksel."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property
    def height(self) -> int:
        """Tinggi frame dalam piksel."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Resolusi video dalam format (width, height)."""
        return (self.width, self.height)
    
    def get_info(self) -> dict:
        """
        Mendapatkan informasi lengkap tentang video.
        
        Returns
        -------
        dict
            Dictionary berisi metadata video
        """
        return {
            "path": self.video_path,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.frame_count / self.fps if self.fps > 0 else 0
        }


# === Testing langsung (opsional) ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_reader.py <path_to_video>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    with VideoReader(video_path) as reader:
        print("Video Info:")
        for key, value in reader.get_info().items():
            print(f"  {key}: {value}")
        
        print("\nMenampilkan video (tekan 'q' untuk keluar)...")
        
        for frame in reader.frames():
            cv2.imshow("Video", frame)
            
            # Tunggu sesuai FPS
            delay = int(1000 / reader.fps)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
