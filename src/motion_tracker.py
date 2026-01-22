"""
Motion Tracker Module
=====================

Modul untuk melacak objek bergerak antar frame.
Menggunakan centroid tracking untuk menjaga identitas objek.

Konsep:
- Setiap objek dilacak berdasarkan posisi centroid-nya
- Objek baru diberi ID unik
- Trail disimpan untuk visualisasi path gerakan
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class TrackedObject:
    """Representasi objek yang sedang dilacak."""
    id: int
    centroid: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    trail: deque = field(default_factory=lambda: deque(maxlen=30))
    age: int = 0  # Frame sejak terakhir terdeteksi
    total_frames: int = 0  # Total frame objek terdeteksi
    velocity: Tuple[float, float] = (0.0, 0.0)  # Kecepatan (dx, dy)
    
    def __post_init__(self):
        self.trail.append(self.centroid)


class MotionTracker:
    """
    Melacak objek bergerak menggunakan centroid-based tracking.
    
    Algoritma:
    1. Terima list centroid dari frame saat ini
    2. Hitung jarak setiap centroid baru ke objek yang sudah dilacak
    3. Pasangkan berdasarkan jarak terdekat (Hungarian algorithm simplified)
    4. Objek baru → buat ID baru
    5. Objek yang hilang terlalu lama → hapus dari tracking
    """
    
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 100.0,
        trail_length: int = 30
    ):
        """
        Parameters
        ----------
        max_disappeared : int
            Maksimum frame sebelum objek dianggap hilang
        max_distance : float
            Jarak maksimum untuk dianggap objek yang sama
        trail_length : int
            Panjang trail untuk visualisasi
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trail_length = trail_length
        
        self.next_id = 0
        self.objects: Dict[int, TrackedObject] = {}
    
    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Hitung jarak Euclidean antara dua titik."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def update(
        self,
        detections: List[Tuple[Tuple[int, int], Tuple[int, int, int, int]]]
    ) -> Dict[int, TrackedObject]:
        """
        Update tracking dengan deteksi baru.
        
        Parameters
        ----------
        detections : List[Tuple[centroid, bounding_box]]
            List of (centroid, bounding_box) dari deteksi frame ini
        
        Returns
        -------
        Dict[int, TrackedObject]
            Dictionary objek yang sedang dilacak
        """
        # Jika tidak ada deteksi, increment age semua objek
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].age += 1
                if self.objects[obj_id].age > self.max_disappeared:
                    del self.objects[obj_id]
            return self.objects
        
        # Jika belum ada objek, register semua deteksi sebagai objek baru
        if len(self.objects) == 0:
            for centroid, bbox in detections:
                self._register(centroid, bbox)
            return self.objects
        
        # Hitung jarak antara objek yang ada dengan deteksi baru
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid].centroid for oid in object_ids]
        detection_centroids = [d[0] for d in detections]
        
        # Matrix jarak
        distances = np.zeros((len(object_centroids), len(detection_centroids)))
        for i, oc in enumerate(object_centroids):
            for j, dc in enumerate(detection_centroids):
                distances[i, j] = self._distance(oc, dc)
        
        # Greedy matching (simplified Hungarian)
        used_rows = set()
        used_cols = set()
        matches = []
        
        # Sort by distance
        indices = np.argsort(distances.flatten())
        for idx in indices:
            row = idx // len(detection_centroids)
            col = idx % len(detection_centroids)
            
            if row in used_rows or col in used_cols:
                continue
            
            if distances[row, col] > self.max_distance:
                continue
            
            matches.append((row, col))
            used_rows.add(row)
            used_cols.add(col)
        
        # Update matched objects
        for row, col in matches:
            obj_id = object_ids[row]
            new_centroid, new_bbox = detections[col]
            
            # Calculate velocity
            old_centroid = self.objects[obj_id].centroid
            velocity = (
                new_centroid[0] - old_centroid[0],
                new_centroid[1] - old_centroid[1]
            )
            
            self.objects[obj_id].centroid = new_centroid
            self.objects[obj_id].bounding_box = new_bbox
            self.objects[obj_id].trail.append(new_centroid)
            self.objects[obj_id].age = 0
            self.objects[obj_id].total_frames += 1
            self.objects[obj_id].velocity = velocity
        
        # Handle unmatched objects (disappeared)
        for row in range(len(object_ids)):
            if row not in used_rows:
                obj_id = object_ids[row]
                self.objects[obj_id].age += 1
                if self.objects[obj_id].age > self.max_disappeared:
                    del self.objects[obj_id]
        
        # Handle unmatched detections (new objects)
        for col in range(len(detections)):
            if col not in used_cols:
                centroid, bbox = detections[col]
                self._register(centroid, bbox)
        
        return self.objects
    
    def _register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]):
        """Register objek baru."""
        trail = deque(maxlen=self.trail_length)
        trail.append(centroid)
        
        self.objects[self.next_id] = TrackedObject(
            id=self.next_id,
            centroid=centroid,
            bounding_box=bbox,
            trail=trail
        )
        self.next_id += 1
    
    def reset(self):
        """Reset semua tracking."""
        self.objects.clear()
        self.next_id = 0


def draw_tracking(
    frame: np.ndarray,
    objects: Dict[int, TrackedObject],
    show_trails: bool = True,
    show_id: bool = True,
    trail_color_gradient: bool = True
) -> np.ndarray:
    """
    Gambar hasil tracking pada frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame input
    objects : Dict[int, TrackedObject]
        Dictionary objek yang dilacak
    show_trails : bool
        Tampilkan trail gerakan
    show_id : bool
        Tampilkan ID objek
    trail_color_gradient : bool
        Gunakan gradasi warna pada trail
    """
    output = frame.copy()
    
    # Warna untuk setiap ID (cycling through colors)
    colors = [
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (0, 255, 0),      # Green
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
    ]
    
    for obj_id, obj in objects.items():
        color = colors[obj_id % len(colors)]
        x, y, w, h = obj.bounding_box
        
        # Draw bounding box with glow effect
        cv2.rectangle(output, (x-1, y-1), (x+w+1, y+h+1), (0, 0, 0), 3)
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
        
        # Draw trails
        if show_trails and len(obj.trail) > 1:
            points = list(obj.trail)
            for i in range(1, len(points)):
                if trail_color_gradient:
                    # Fade effect: older = more transparent
                    alpha = i / len(points)
                    thickness = max(1, int(3 * alpha))
                else:
                    thickness = 2
                
                cv2.line(output, points[i-1], points[i], color, thickness)
        
        # Draw centroid
        cv2.circle(output, obj.centroid, 5, color, -1)
        cv2.circle(output, obj.centroid, 5, (255, 255, 255), 1)
        
        # Draw ID and velocity
        if show_id:
            text = f"ID:{obj_id}"
            # Velocity indicator
            speed = math.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
            if speed > 2:
                text += f" [{speed:.0f}px/f]"
            
            cv2.putText(
                output, text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )
            cv2.putText(
                output, text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
    
    return output
