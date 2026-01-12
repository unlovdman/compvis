"""
Background Model Module
=======================
Model background untuk deteksi gerakan:
1. Frame Differencing - sederhana
2. Running Average - lebih robust
3. MOG2 - advanced (improvement)
"""

import cv2
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod


class BackgroundModel(ABC):
    """Abstract base class untuk model background."""
    
    @abstractmethod
    def update(self, frame: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_background(self) -> Optional[np.ndarray]:
        pass


class FrameDifferencingModel(BackgroundModel):
    """
    Frame Differencing: bandingkan frame saat ini dengan sebelumnya.
    Kelebihan: Sederhana dan cepat
    Kekurangan: Sensitif noise, ghosting effect
    """
    
    def __init__(self):
        self.prev_frame: Optional[np.ndarray] = None
        
    def update(self, frame: np.ndarray) -> None:
        self.prev_frame = frame.copy()
        
    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return np.zeros_like(frame)
        return cv2.absdiff(frame, self.prev_frame)
    
    def get_background(self) -> Optional[np.ndarray]:
        return self.prev_frame


class RunningAverageModel(BackgroundModel):
    """
    Running Average: background = alpha*current + (1-alpha)*background
    Alpha kecil = adaptasi lambat, background stabil
    """
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.background: Optional[np.ndarray] = None
        
    def update(self, frame: np.ndarray) -> None:
        frame_float = frame.astype(np.float32)
        if self.background is None:
            self.background = frame_float.copy()
        else:
            cv2.accumulateWeighted(frame_float, self.background, self.alpha)
        
    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        if self.background is None:
            return np.zeros_like(frame)
        return cv2.absdiff(frame, self.background.astype(np.uint8))
    
    def get_background(self) -> Optional[np.ndarray]:
        if self.background is None:
            return None
        return self.background.astype(np.uint8)


class MOG2Model(BackgroundModel):
    """MOG2: Mixture of Gaussians - handles multimodal backgrounds."""
    
    def __init__(self, history: int = 500, var_threshold: float = 16):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        self.background: Optional[np.ndarray] = None
        
    def update(self, frame: np.ndarray) -> None:
        pass
        
    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        mask = self.subtractor.apply(frame)
        self.background = self.subtractor.getBackgroundImage()
        return mask
    
    def get_background(self) -> Optional[np.ndarray]:
        return self.background


def create_background_model(method: str = "running_average", **kwargs):
    """Factory function untuk membuat background model."""
    models = {
        "frame_diff": FrameDifferencingModel,
        "running_average": RunningAverageModel,
        "mog2": MOG2Model
    }
    if method not in models:
        raise ValueError(f"Method '{method}' tidak dikenali")
    return models[method](**kwargs)
