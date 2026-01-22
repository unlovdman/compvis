"""
Alert Module
============

Modul untuk notifikasi audio saat gerakan terdeteksi.
Menggunakan system beep untuk kesederhanaan (tanpa dependency tambahan).
"""

import os
import sys
import time
from typing import Optional
import subprocess


class MotionAlert:
    """
    Notifikasi audio saat gerakan terdeteksi.
    
    Fitur:
    - System beep (cross-platform)
    - Cooldown untuk mencegah spam
    - Volume control (jika didukung)
    """
    
    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        enabled: bool = True,
        min_regions: int = 1
    ):
        """
        Parameters
        ----------
        cooldown_seconds : float
            Waktu minimum antara alert
        enabled : bool
            Apakah alert aktif
        min_regions : int
            Minimum jumlah region gerakan untuk trigger alert
        """
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self.min_regions = min_regions
        
        self.last_alert_time: float = 0
        self._detect_platform()
    
    def _detect_platform(self):
        """Deteksi platform untuk menentukan metode beep."""
        self.platform = sys.platform
    
    def _beep(self, frequency: int = 1000, duration_ms: int = 200):
        """
        Mainkan beep sound.
        
        Parameters
        ----------
        frequency : int
            Frekuensi beep dalam Hz
        duration_ms : int
            Durasi dalam milliseconds
        """
        try:
            if self.platform == "linux":
                # Linux: gunakan paplay atau speaker-test
                # Fallback ke print bell character
                try:
                    # Coba paplay dengan file beep jika ada
                    subprocess.run(
                        ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=1
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    # Fallback ke terminal bell
                    print("\a", end="", flush=True)
            
            elif self.platform == "darwin":
                # macOS: gunakan afplay atau osascript
                try:
                    subprocess.run(
                        ["afplay", "/System/Library/Sounds/Ping.aiff"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=1
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    print("\a", end="", flush=True)
            
            elif self.platform == "win32":
                # Windows: gunakan winsound
                try:
                    import winsound
                    winsound.Beep(frequency, duration_ms)
                except ImportError:
                    print("\a", end="", flush=True)
            
            else:
                # Fallback universal
                print("\a", end="", flush=True)
                
        except Exception:
            # Silent fail - jangan crash karena audio
            pass
    
    def trigger(self, num_regions: int = 1, force: bool = False) -> bool:
        """
        Trigger alert jika kondisi terpenuhi.
        
        Parameters
        ----------
        num_regions : int
            Jumlah region gerakan terdeteksi
        force : bool
            Abaikan cooldown
        
        Returns
        -------
        bool
            True jika alert dibunyikan
        """
        if not self.enabled:
            return False
        
        if num_regions < self.min_regions:
            return False
        
        current_time = time.time()
        time_since_last = current_time - self.last_alert_time
        
        if not force and time_since_last < self.cooldown_seconds:
            return False
        
        self._beep()
        self.last_alert_time = current_time
        return True
    
    def toggle(self) -> bool:
        """Toggle alert on/off."""
        self.enabled = not self.enabled
        return self.enabled


class MotionAlertAdvanced(MotionAlert):
    """
    Alert dengan fitur tambahan: berbagai level urgency.
    """
    
    LEVEL_LOW = "low"
    LEVEL_MEDIUM = "medium"
    LEVEL_HIGH = "high"
    
    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        enabled: bool = True,
        min_regions: int = 1,
        area_threshold_medium: int = 5000,
        area_threshold_high: int = 20000
    ):
        """
        Parameters tambahan:
        - area_threshold_medium: area minimum untuk level medium
        - area_threshold_high: area minimum untuk level high
        """
        super().__init__(cooldown_seconds, enabled, min_regions)
        self.area_threshold_medium = area_threshold_medium
        self.area_threshold_high = area_threshold_high
    
    def _get_level(self, total_area: int) -> str:
        """Tentukan level urgency berdasarkan total area gerakan."""
        if total_area >= self.area_threshold_high:
            return self.LEVEL_HIGH
        elif total_area >= self.area_threshold_medium:
            return self.LEVEL_MEDIUM
        else:
            return self.LEVEL_LOW
    
    def trigger_with_level(
        self,
        num_regions: int,
        total_area: int,
        force: bool = False
    ) -> Optional[str]:
        """
        Trigger alert dengan level urgency.
        
        Returns
        -------
        Optional[str]
            Level yang di-trigger, atau None jika tidak ada alert
        """
        if not self.enabled or num_regions < self.min_regions:
            return None
        
        current_time = time.time()
        time_since_last = current_time - self.last_alert_time
        
        if not force and time_since_last < self.cooldown_seconds:
            return None
        
        level = self._get_level(total_area)
        
        # Beep berbeda untuk setiap level
        if level == self.LEVEL_HIGH:
            # Triple beep untuk high urgency
            for _ in range(3):
                self._beep(1500, 150)
                time.sleep(0.1)
        elif level == self.LEVEL_MEDIUM:
            # Double beep
            self._beep(1200, 200)
            time.sleep(0.15)
            self._beep(1200, 200)
        else:
            # Single beep
            self._beep(1000, 200)
        
        self.last_alert_time = current_time
        return level
