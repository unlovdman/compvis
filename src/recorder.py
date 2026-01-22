"""
Video Recorder Module
=====================

Modul untuk merekam output video dan mengambil snapshot.
Fitur auto-record saat gerakan terdeteksi.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from datetime import datetime
import os


class VideoRecorder:
    """
    Merekam frame ke file video.
    
    Mendukung:
    - Recording manual (start/stop)
    - Auto-record saat ada event (motion detected)
    - Berbagai codec (MJPG, XVID, H264)
    """
    
    def __init__(
        self,
        output_dir: str = "recordings",
        fps: float = 30.0,
        codec: str = "MJPG",
        auto_filename: bool = True
    ):
        """
        Parameters
        ----------
        output_dir : str
            Direktori untuk menyimpan rekaman
        fps : float
            Frame per second untuk output
        codec : str
            Codec video (MJPG, XVID, mp4v)
        auto_filename : bool
            Jika True, buat nama file otomatis dengan timestamp
        """
        self.output_dir = output_dir
        self.fps = fps
        self.codec = codec
        self.auto_filename = auto_filename
        
        self.writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False
        self.current_file: Optional[str] = None
        self.frame_count = 0
        
        # Buat direktori jika belum ada
        os.makedirs(output_dir, exist_ok=True)
    
    def _generate_filename(self, prefix: str = "recording") -> str:
        """Generate nama file dengan timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"{prefix}_{timestamp}.avi")
    
    def start(
        self,
        frame_size: Tuple[int, int],
        filename: Optional[str] = None
    ) -> bool:
        """
        Mulai recording.
        
        Parameters
        ----------
        frame_size : Tuple[int, int]
            Ukuran frame (width, height)
        filename : Optional[str]
            Path file output. Jika None, generate otomatis.
        
        Returns
        -------
        bool
            True jika berhasil memulai recording
        """
        if self.is_recording:
            return False
        
        # Generate atau gunakan filename yang diberikan
        if filename is None:
            if self.auto_filename:
                self.current_file = self._generate_filename()
            else:
                self.current_file = os.path.join(self.output_dir, "output.avi")
        else:
            self.current_file = filename
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.current_file,
            fourcc,
            self.fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            print(f"Error: Tidak dapat membuat file video: {self.current_file}")
            return False
        
        self.is_recording = True
        self.frame_count = 0
        print(f"🔴 Recording started: {self.current_file}")
        return True
    
    def write(self, frame: np.ndarray) -> bool:
        """
        Tulis frame ke video.
        
        Parameters
        ----------
        frame : np.ndarray
            Frame BGR untuk direkam
        
        Returns
        -------
        bool
            True jika berhasil menulis
        """
        if not self.is_recording or self.writer is None:
            return False
        
        self.writer.write(frame)
        self.frame_count += 1
        return True
    
    def stop(self) -> Optional[str]:
        """
        Berhenti recording.
        
        Returns
        -------
        Optional[str]
            Path file yang direkam, atau None jika tidak recording
        """
        if not self.is_recording or self.writer is None:
            return None
        
        self.writer.release()
        self.writer = None
        self.is_recording = False
        
        saved_file = self.current_file
        print(f"⏹️  Recording stopped: {saved_file} ({self.frame_count} frames)")
        
        self.current_file = None
        self.frame_count = 0
        
        return saved_file
    
    def toggle(self, frame_size: Tuple[int, int]) -> bool:
        """Toggle recording on/off."""
        if self.is_recording:
            self.stop()
            return False
        else:
            self.start(frame_size)
            return True


class SnapshotCapture:
    """
    Mengambil screenshot dari frame.
    """
    
    def __init__(
        self,
        output_dir: str = "snapshots",
        format: str = "jpg",
        quality: int = 95
    ):
        """
        Parameters
        ----------
        output_dir : str
            Direktori untuk menyimpan snapshot
        format : str
            Format gambar (jpg, png)
        quality : int
            Kualitas kompresi (untuk jpg)
        """
        self.output_dir = output_dir
        self.format = format
        self.quality = quality
        
        os.makedirs(output_dir, exist_ok=True)
    
    def capture(
        self,
        frame: np.ndarray,
        prefix: str = "snapshot",
        filename: Optional[str] = None
    ) -> str:
        """
        Ambil snapshot.
        
        Parameters
        ----------
        frame : np.ndarray
            Frame untuk di-capture
        prefix : str
            Prefix nama file
        filename : Optional[str]
            Nama file custom
        
        Returns
        -------
        str
            Path file yang disimpan
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{prefix}_{timestamp}.{self.format}"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Set quality params
        if self.format.lower() == "jpg":
            params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        elif self.format.lower() == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        else:
            params = []
        
        cv2.imwrite(filepath, frame, params)
        print(f"📸 Snapshot saved: {filepath}")
        
        return filepath


class MotionRecorder:
    """
    Auto-record saat gerakan terdeteksi.
    
    Fitur:
    - Pre-buffer: Simpan N frame sebelum gerakan untuk context
    - Post-buffer: Terus rekam N frame setelah gerakan berhenti
    """
    
    def __init__(
        self,
        output_dir: str = "motion_clips",
        pre_buffer_seconds: float = 2.0,
        post_buffer_seconds: float = 3.0,
        fps: float = 30.0
    ):
        """
        Parameters
        ----------
        output_dir : str
            Direktori output
        pre_buffer_seconds : float
            Durasi buffer sebelum gerakan
        post_buffer_seconds : float
            Durasi buffer setelah gerakan berhenti
        fps : float
            FPS video
        """
        self.output_dir = output_dir
        self.pre_buffer_frames = int(pre_buffer_seconds * fps)
        self.post_buffer_frames = int(post_buffer_seconds * fps)
        self.fps = fps
        
        self.pre_buffer: list = []
        self.recorder = VideoRecorder(output_dir, fps)
        self.frames_since_last_motion = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def update(
        self,
        frame: np.ndarray,
        motion_detected: bool
    ) -> Optional[str]:
        """
        Update dengan frame baru.
        
        Parameters
        ----------
        frame : np.ndarray
            Frame saat ini
        motion_detected : bool
            Apakah ada gerakan di frame ini
        
        Returns
        -------
        Optional[str]
            Path file jika recording selesai, None otherwise
        """
        saved_file = None
        
        if motion_detected:
            self.frames_since_last_motion = 0
            
            # Mulai recording jika belum
            if not self.recorder.is_recording:
                h, w = frame.shape[:2]
                self.recorder.start((w, h))
                
                # Tulis pre-buffer
                for buffered_frame in self.pre_buffer:
                    self.recorder.write(buffered_frame)
            
            self.recorder.write(frame)
        else:
            self.frames_since_last_motion += 1
            
            if self.recorder.is_recording:
                # Post-buffer: terus rekam sebentar
                if self.frames_since_last_motion <= self.post_buffer_frames:
                    self.recorder.write(frame)
                else:
                    saved_file = self.recorder.stop()
        
        # Update pre-buffer
        self.pre_buffer.append(frame.copy())
        if len(self.pre_buffer) > self.pre_buffer_frames:
            self.pre_buffer.pop(0)
        
        return saved_file
