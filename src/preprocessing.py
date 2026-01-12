"""
Preprocessing Module
====================

Modul ini berisi fungsi-fungsi untuk praproses frame sebelum
deteksi gerakan.

Tahapan preprocessing:
1. Konversi ke Grayscale - mengurangi kompleksitas
2. Gaussian Blur - mengurangi noise
3. (Opsional) Resize - untuk efisiensi

Konsep Penting:
- Grayscale: 1 channel (intensitas 0-255), lebih cepat diproses
- Gaussian Blur: Konvolusi dengan kernel Gaussian untuk menghaluskan
- Noise: Variasi acak pada piksel yang tidak merepresentasikan informasi sebenarnya
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Mengubah frame BGR ke grayscale.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame input dengan format BGR (H, W, 3)
    
    Returns
    -------
    np.ndarray
        Frame grayscale dengan format (H, W)
    
    Penjelasan:
    -----------
    Rumus konversi (weighted average berdasarkan persepsi mata):
    Y = 0.299*R + 0.587*G + 0.114*B
    
    Mata manusia lebih sensitif terhadap warna hijau,
    sehingga green mendapat bobot tertinggi.
    """
    # Cek apakah sudah grayscale
    if len(frame.shape) == 2:
        return frame
    
    # Konversi BGR ke Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def apply_gaussian_blur(
    frame: np.ndarray,
    kernel_size: int = 21
) -> np.ndarray:
    """
    Menerapkan Gaussian Blur untuk mengurangi noise.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame input (bisa grayscale atau BGR)
    kernel_size : int
        Ukuran kernel Gaussian (harus ganjil: 3, 5, 7, ...)
        Semakin besar = semakin halus, tapi detail hilang
    
    Returns
    -------
    np.ndarray
        Frame yang sudah di-blur
    
    Penjelasan:
    -----------
    Gaussian Blur bekerja dengan:
    1. Membuat kernel 2D dengan distribusi Gaussian (bell curve)
    2. Menggeser kernel di setiap posisi piksel
    3. Menghitung weighted average piksel di sekitarnya
    
    Kernel 5x5 contoh (dinormalisasi):
        1   4   7   4   1
        4  16  26  16   4
        7  26  41  26   7   / 273
        4  16  26  16   4
        1   4   7   4   1
    
    Kenapa kernel harus ganjil?
    - Agar ada piksel tengah sebagai anchor point
    """
    # Pastikan kernel_size ganjil
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    # Parameter sigma 0 = dihitung otomatis dari kernel_size
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    return blurred


def resize_frame(
    frame: np.ndarray,
    scale: float = 0.5,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Mengubah ukuran frame untuk efisiensi pemrosesan.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame input
    scale : float
        Skala resize (0.5 = setengah ukuran asli)
    target_size : Optional[Tuple[int, int]]
        Ukuran target (width, height), jika None gunakan scale
    
    Returns
    -------
    np.ndarray
        Frame yang sudah di-resize
    
    Penjelasan:
    -----------
    Resize berguna untuk:
    - Mengurangi waktu komputasi (lebih sedikit piksel)
    - Mengurangi noise (downsampling bertindak seperti low-pass filter)
    
    Trade-off:
    - Frame lebih kecil = lebih cepat, tapi detail hilang
    - Objek kecil mungkin tidak terdeteksi setelah resize
    """
    if target_size is not None:
        width, height = target_size
    else:
        height, width = frame.shape[:2]
        width = int(width * scale)
        height = int(height * scale)
    
    # INTER_AREA bagus untuk downsampling (shrinking)
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def preprocess_frame(
    frame: np.ndarray,
    blur_size: int = 21,
    resize_scale: Optional[float] = None
) -> np.ndarray:
    """
    Pipeline preprocessing lengkap.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame input BGR
    blur_size : int
        Ukuran kernel Gaussian blur
    resize_scale : Optional[float]
        Skala resize (None = tidak resize)
    
    Returns
    -------
    np.ndarray
        Frame yang sudah diproses (grayscale + blurred)
    
    Pipeline:
    ---------
    [BGR Frame] → [Resize?] → [Grayscale] → [Gaussian Blur] → [Output]
    """
    # Step 1: Resize (opsional)
    if resize_scale is not None and resize_scale != 1.0:
        frame = resize_frame(frame, scale=resize_scale)
    
    # Step 2: Convert to grayscale
    gray = convert_to_grayscale(frame)
    
    # Step 3: Apply Gaussian blur
    blurred = apply_gaussian_blur(gray, kernel_size=blur_size)
    
    return blurred


def compute_absolute_difference(
    frame1: np.ndarray,
    frame2: np.ndarray
) -> np.ndarray:
    """
    Menghitung perbedaan absolut antara dua frame.
    
    Parameters
    ----------
    frame1 : np.ndarray
        Frame pertama (biasanya frame sebelumnya atau background)
    frame2 : np.ndarray
        Frame kedua (biasanya frame saat ini)
    
    Returns
    -------
    np.ndarray
        Difference image dengan nilai 0-255
    
    Penjelasan:
    -----------
    diff(x,y) = |frame1(x,y) - frame2(x,y)|
    
    Piksel yang berubah drastis akan punya nilai tinggi (terang).
    Piksel yang tidak berubah akan punya nilai rendah (gelap).
    
    Ini adalah dasar dari frame differencing untuk deteksi gerakan.
    """
    diff = cv2.absdiff(frame1, frame2)
    return diff


def apply_threshold(
    diff_image: np.ndarray,
    threshold_value: int = 25
) -> np.ndarray:
    """
    Menerapkan thresholding untuk membuat binary mask.
    
    Parameters
    ----------
    diff_image : np.ndarray
        Difference image (hasil absdiff)
    threshold_value : int
        Nilai ambang (0-255)
        - Terlalu rendah = banyak noise
        - Terlalu tinggi = gerakan kecil tidak terdeteksi
    
    Returns
    -------
    np.ndarray
        Binary mask (nilai 0 atau 255)
    
    Penjelasan:
    -----------
    Untuk setiap piksel:
        if diff(x,y) > threshold:
            mask(x,y) = 255 (putih = GERAKAN)
        else:
            mask(x,y) = 0 (hitam = DIAM)
    
    Tips pemilihan threshold:
    - Mulai dari 25, naikan jika banyak false positive
    - Turunkan jika gerakan lambat tidak terdeteksi
    """
    _, binary_mask = cv2.threshold(
        diff_image,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )
    return binary_mask


def apply_morphology(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations_erode: int = 1,
    iterations_dilate: int = 2
) -> np.ndarray:
    """
    Menerapkan operasi morfologi untuk membersihkan mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask input
    kernel_size : int
        Ukuran structuring element
    iterations_erode : int
        Jumlah iterasi erosion
    iterations_dilate : int
        Jumlah iterasi dilation
    
    Returns
    -------
    np.ndarray
        Mask yang sudah dibersihkan
    
    Penjelasan:
    -----------
    EROSION (Erosi):
    - Mengecilkan area putih
    - Menghilangkan noise kecil (titik-titik putih isolated)
    - Piksel putih menjadi hitam jika ada hitam di tetangganya
    
    DILATION (Dilasi):
    - Memperbesar area putih
    - Mengisi lubang kecil dalam objek
    - Menyambungkan area yang terpisah dekat
    - Piksel hitam menjadi putih jika ada putih di tetangganya
    
    Opening (erosi lalu dilasi):
    - Membersihkan noise tanpa mengubah size objek terlalu banyak
    
    Closing (dilasi lalu erosi):
    - Mengisi lubang tanpa memperbesar objek terlalu banyak
    """
    # Buat structuring element (kernel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,  # Bentuk elips lebih natural
        (kernel_size, kernel_size)
    )
    
    # Step 1: Erosion - hapus noise kecil
    eroded = cv2.erode(mask, kernel, iterations=iterations_erode)
    
    # Step 2: Dilation - perbesar kembali + isi lubang
    dilated = cv2.dilate(eroded, kernel, iterations=iterations_dilate)
    
    return dilated


# === Demo preprocessing ===
if __name__ == "__main__":
    import sys
    
    # Buat gambar dummy untuk demo
    print("Demo Preprocessing Functions")
    print("=" * 50)
    
    # Simulasi frame 100x100 dengan noise
    frame1 = np.zeros((100, 100), dtype=np.uint8)
    frame1[40:60, 40:60] = 200  # Kotak di tengah
    
    frame2 = np.zeros((100, 100), dtype=np.uint8)  
    frame2[42:62, 45:65] = 200  # Kotak bergeser
    
    # Compute difference
    diff = compute_absolute_difference(frame1, frame2)
    print(f"Difference range: {diff.min()} - {diff.max()}")
    
    # Apply threshold
    mask = apply_threshold(diff, threshold_value=50)
    print(f"Mask unique values: {np.unique(mask)}")
    
    # Apply morphology
    clean_mask = apply_morphology(mask)
    print(f"Clean mask shape: {clean_mask.shape}")
    
    print("\nSemua fungsi berjalan dengan baik!")
