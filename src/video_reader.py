"""
Video Reader Module
===================

Modul ini menangani pembacaan video dari file ATAU webcam.
Menggunakan OpenCV VideoCapture untuk membaca frame per frame.

Konsep Penting:
- Video adalah urutan gambar (frame) yang ditampilkan dengan kecepatan tertentu (FPS)
- Setiap frame adalah array numpy dengan shape (height, width, channels)
- Format warna default OpenCV adalah BGR, bukan RGB
- Webcam dapat diakses dengan index (0, 1, 2...) atau path device
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator, Union


class VideoReader:
    """
    Kelas untuk membaca video dari file atau webcam.
    
    Menggunakan context manager (with statement) untuk memastikan
    resource video selalu di-release dengan benar.
    
    Contoh penggunaan:
    -----------------
    >>> # Dari file video
    >>> with VideoReader("video.mp4") as reader:
    ...     for frame in reader.frames():
    ...         process(frame)
    
    >>> # Dari webcam
    >>> with VideoReader(0) as reader:  # 0 = webcam default
    ...     for frame in reader.frames():
    ...         process(frame)
    """
    
    def __init__(
        self, 
        source: Union[str, int],
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None
    ):
        """
        Inisialisasi VideoReader.
        
        Parameters
        ----------
        source : Union[str, int]
            Path ke file video (mp4, avi, mkv, dll) ATAU
            Index webcam (0, 1, 2...) untuk kamera
        width : Optional[int]
            Lebar frame untuk webcam (default: 640)
        height : Optional[int]
            Tinggi frame untuk webcam (default: 480)
        fps : Optional[int]
            FPS untuk webcam (default: 30)
        """
        self.source = source
        self._is_webcam = isinstance(source, int)
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Webcam settings
        self._webcam_width = width or 640
        self._webcam_height = height or 480
        self._webcam_fps = fps or 30
        
    def __enter__(self) -> "VideoReader":
        """
        Dipanggil saat masuk blok 'with'.
        Membuka koneksi ke file video atau webcam.
        """
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Dipanggil saat keluar blok 'with'.
        Memastikan video selalu di-release.
        """
        self.release()
    
    @property
    def is_webcam(self) -> bool:
        """Cek apakah sumber adalah webcam."""
        return self._is_webcam
        
    def open(self) -> bool:
        """
        Membuka file video atau webcam.
        
        Returns
        -------
        bool
            True jika berhasil membuka video, False jika gagal
        """
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            source_type = "webcam" if self._is_webcam else "video"
            raise FileNotFoundError(
                f"Tidak dapat membuka {source_type}: {self.source}\n"
                "Pastikan path benar atau webcam terpasang."
            )
        
        # Konfigurasi webcam jika diperlukan
        if self._is_webcam:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._webcam_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._webcam_height)
            self.cap.set(cv2.CAP_PROP_FPS, self._webcam_fps)
            # Reduce buffer untuk latency rendah
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
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
        - Untuk webcam, ret False biasanya berarti kamera disconnect
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
        Untuk webcam, mengembalikan nilai yang di-set atau aktual dari kamera.
        """
        if self.cap is None:
            return 0.0
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Webcam kadang return 0, gunakan nilai default
        if self._is_webcam and actual_fps <= 0:
            return float(self._webcam_fps)
        return actual_fps
    
    @property
    def frame_count(self) -> int:
        """
        Total jumlah frame dalam video.
        Untuk webcam, mengembalikan 0 (tidak terbatas).
        """
        if self.cap is None or self._is_webcam:
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
        info = {
            "source": str(self.source),
            "is_webcam": self._is_webcam,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }
        
        if not self._is_webcam:
            info["frame_count"] = self.frame_count
            info["duration_seconds"] = self.frame_count / self.fps if self.fps > 0 else 0
        else:
            info["frame_count"] = "∞"
            info["duration_seconds"] = "∞"
            
        return info


# === Testing langsung (opsional) ===
if __name__ == "__main__":
    import sys
    
    # Default ke webcam jika tidak ada argument
    if len(sys.argv) < 2:
        print("No argument provided. Using webcam (index 0)...")
        source = 0
    else:
        arg = sys.argv[1]
        # Cek apakah argument adalah angka (webcam index)
        source = int(arg) if arg.isdigit() else arg
    
    with VideoReader(source) as reader:
        print("Video Info:")
        for key, value in reader.get_info().items():
            print(f"  {key}: {value}")
        
        print("\nMenampilkan video (tekan 'q' untuk keluar)...")
        
        for frame in reader.frames():
            cv2.imshow("Video", frame)
            
            # Tunggu sesuai FPS
            delay = max(1, int(1000 / reader.fps)) if reader.fps > 0 else 30
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()

