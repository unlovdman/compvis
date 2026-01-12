# Sistem Deteksi Gerak dengan Computer Vision Klasik

> **Target Pembaca:** Mahasiswa tingkat sarjana  
> **Metode:** Frame Differencing & Background Subtraction  
> **Library:** OpenCV (Python)

---

## 📚 Daftar Isi

1. [Penjelasan Algoritma](#1-penjelasan-algoritma-deteksi-gerak)
2. [Alasan Pemilihan Metode](#2-alasan-pemilihan-metode-dan-asumsi)
3. [Struktur Proyek](#3-struktur-proyek)
4. [Cara Menjalankan](#4-cara-menjalankan)
5. [Kondisi Keberhasilan dan Kegagalan](#5-kondisi-keberhasilan-dan-kegagalan)
6. [Usulan Perbaikan](#6-usulan-perbaikan)

---

## 1. Penjelasan Algoritma Deteksi Gerak

### Pendahuluan

Deteksi gerak (*motion detection*) adalah proses mengidentifikasi perubahan posisi objek 
dalam urutan frame video. Pada kamera statis, perubahan piksel yang signifikan biasanya 
mengindikasikan adanya objek bergerak.

### Tahapan Algoritma

#### Tahap 1: Pembacaan Frame Video

```
┌─────────────────────────────────────────────────────────────┐
│  VIDEO FILE (.mp4, .avi, dll)                               │
│     ↓                                                       │
│  cv2.VideoCapture(path) → Membaca video frame per frame     │
│     ↓                                                       │
│  Frame ke-n (gambar RGB/BGR)                                │
└─────────────────────────────────────────────────────────────┘
```

Video adalah kumpulan gambar (frame) yang ditampilkan secara berurutan. 
OpenCV membaca setiap frame sebagai array numpy dengan format BGR (Blue-Green-Red).

#### Tahap 2: Konversi ke Grayscale

```
┌─────────────────────────────────────────────────────────────┐
│  Frame BGR (3 channel: B, G, R)                             │
│     ↓                                                       │
│  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                    │
│     ↓                                                       │
│  Frame Grayscale (1 channel: intensitas 0-255)              │
└─────────────────────────────────────────────────────────────┘
```

**Alasan konversi:**
- Mengurangi kompleksitas komputasi (dari 3 channel menjadi 1 channel)
- Deteksi gerak tidak memerlukan informasi warna
- Fokus pada perubahan intensitas cahaya

#### Tahap 3: Pengurangan Noise dengan Gaussian Blur

```
┌─────────────────────────────────────────────────────────────┐
│  Frame Grayscale                                            │
│     ↓                                                       │
│  cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)      │
│     ↓                                                       │
│  Frame yang sudah di-blur (noise berkurang)                 │
└─────────────────────────────────────────────────────────────┘
```

**Mengapa Gaussian Blur?**
- Sensor kamera menghasilkan noise acak
- Kompresi video menambah artefak
- Blur menghaluskan variasi piksel kecil yang bukan gerakan nyata

#### Tahap 4: Background Subtraction / Frame Differencing

**Metode A: Frame Differencing (Sederhana)**

```
┌─────────────────────────────────────────────────────────────┐
│  Frame ke-(n-1) ──┐                                         │
│                   ├──→ cv2.absdiff() ──→ Difference Image   │
│  Frame ke-n ──────┘                                         │
└─────────────────────────────────────────────────────────────┘
```

Menghitung perbedaan absolut antara dua frame berturutan:
```
diff(x,y) = |frame_n(x,y) - frame_{n-1}(x,y)|
```

**Metode B: Running Average Background Model**

```
┌─────────────────────────────────────────────────────────────┐
│  Background Model (rata-rata akumulatif)                    │
│     ↓                                                       │
│  bg_new = α × frame_current + (1-α) × bg_old                │
│     ↓                                                       │
│  diff = |frame_current - background|                        │
└─────────────────────────────────────────────────────────────┘
```

Di mana α (alpha) adalah learning rate (biasanya 0.01 - 0.1).
Background model beradaptasi secara perlahan terhadap perubahan gradual.

#### Tahap 5: Thresholding (Binarisasi)

```
┌─────────────────────────────────────────────────────────────┐
│  Difference Image (nilai 0-255)                             │
│     ↓                                                       │
│  cv2.threshold(diff, threshold_value, 255, THRESH_BINARY)   │
│     ↓                                                       │
│  Binary Mask: piksel = 0 (tidak bergerak) atau 255 (gerak)  │
└─────────────────────────────────────────────────────────────┘
```

**Logika thresholding:**
```
if diff(x,y) > threshold:
    mask(x,y) = 255  # Terdeteksi gerakan
else:
    mask(x,y) = 0    # Tidak ada gerakan
```

#### Tahap 6: Operasi Morfologi

```
┌─────────────────────────────────────────────────────────────┐
│  Binary Mask (mungkin ada noise dan lubang)                 │
│     ↓                                                       │
│  Erosion: Menghapus noise kecil                             │
│     ↓                                                       │
│  Dilation: Mengisi lubang, menyambung area                  │
│     ↓                                                       │
│  Clean Mask                                                 │
└─────────────────────────────────────────────────────────────┘
```

**Erosion:** Mengecilkan area putih, menghilangkan titik-titik noise kecil  
**Dilation:** Memperbesar area putih, menyambungkan bagian yang terpisah

#### Tahap 7: Deteksi Kontur dan Bounding Box

```
┌─────────────────────────────────────────────────────────────┐
│  Clean Mask                                                 │
│     ↓                                                       │
│  cv2.findContours() → Menemukan tepi area bergerak          │
│     ↓                                                       │
│  cv2.boundingRect() → Kotak pembatas untuk setiap kontur    │
│     ↓                                                       │
│  Filter berdasarkan minimum_area → Hasil akhir              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Alasan Pemilihan Metode dan Asumsi

### Mengapa Frame Differencing?

| Aspek | Penjelasan |
|-------|------------|
| **Kesederhanaan** | Mudah dipahami dan diimplementasikan |
| **Efisiensi** | Komputasi ringan, cocok untuk real-time |
| **Kamera Statis** | Background relatif konstan, metode ini efektif |

### Mengapa Running Average?

| Aspek | Penjelasan |
|-------|------------|
| **Adaptif** | Dapat menyesuaikan dengan perubahan gradual (cahaya berubah perlahan) |
| **Robust** | Lebih tahan terhadap variasi kecil dibanding frame differencing |
| **Kontrol** | Parameter alpha dapat di-tuning sesuai kebutuhan |

### Asumsi yang Mendasari

1. **Kamera Statis**
   - Background tidak berubah secara tiba-tiba
   - Tidak ada guncangan atau pergeseran kamera

2. **Pencahayaan Stabil**
   - Tidak ada perubahan drastis (misal: lampu menyala/mati)
   - Cahaya alami berubah perlahan (dapat ditangani dengan running average)

3. **Foreground Berbeda dengan Background**
   - Objek bergerak memiliki intensitas berbeda dari background
   - Jika objek dan background sangat mirip, deteksi menjadi sulit

4. **Frame Rate Memadai**
   - Gerakan antar frame cukup signifikan untuk dideteksi
   - Jika terlalu cepat, objek mungkin "melompat" tanpa trail

---

## 3. Struktur Proyek

```
camtask/
├── README.md                    # Dokumentasi (file ini)
├── requirements.txt             # Dependencies Python
├── src/
│   ├── __init__.py
│   ├── video_reader.py          # Modul pembaca video
│   ├── preprocessing.py         # Grayscale, blur, dll
│   ├── background_model.py      # Model background
│   ├── motion_detector.py       # Deteksi gerakan
│   └── visualization.py         # Fungsi visualisasi
├── main.py                      # Program utama
└── sample_videos/               # (Opsional) Video sampel
```

---

## 4. Cara Menjalankan

### Prasyarat

```bash
# Buat virtual environment (opsional tapi disarankan)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# atau: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Menjalankan Program

```bash
# Jalankan dengan video default (jika ada)
python main.py

# Atau jalankan dengan video tertentu
python main.py --video path/to/your/video.mp4

# Opsi lainnya
python main.py --help
```

### Parameter yang Dapat Diatur

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `--video` | - | Path ke file video |
| `--threshold` | 25 | Nilai threshold untuk binarisasi |
| `--min-area` | 500 | Area minimum kontur (piksel²) |
| `--blur-size` | 21 | Ukuran kernel Gaussian blur |
| `--alpha` | 0.01 | Learning rate background model |

---

## 5. Kondisi Keberhasilan dan Kegagalan

### ✅ Kondisi di Mana Metode Bekerja Baik

1. **Kamera Benar-Benar Statis**
   - Dipasang dengan tripod atau mounting tetap
   - Tidak ada getaran dari lingkungan

2. **Pencahayaan Konsisten**
   - Indoor dengan lampu stabil
   - Outdoor pada siang hari yang cerah (bukan mendung berawan)

3. **Kontras Tinggi antara Objek dan Background**
   - Orang berpakaian gelap di depan dinding terang
   - Objek berwarna di depan background netral

4. **Gerakan Moderat**
   - Kecepatan tidak terlalu lambat (sulit dibedakan dari noise)
   - Kecepatan tidak terlalu cepat (ada overlap antar frame)

5. **Background Sederhana**
   - Dinding polos, lantai seragam
   - Sedikit tekstur yang bergerak (tidak ada pohon, air, dll)

### ❌ Kondisi di Mana Metode Cenderung Gagal

1. **Perubahan Pencahayaan Drastis**
   - Awan menutupi matahari → seluruh frame berubah
   - Lampu dinyalakan/dimatikan → false positive masif

2. **Background Dinamis**
   - Pohon bergoyang tertiup angin
   - Air beriak di kolam
   - Layar TV/monitor di background

3. **Kamera Bergerak/Bergetar**
   - Kamera handheld
   - Getaran dari kendaraan lewat

4. **Objek Mirip Background (Camouflage)**
   - Orang berpakaian warna sama dengan dinding
   - Bayangan objek terdeteksi sebagai gerakan

5. **Objek Bergerak Sangat Lambat**
   - Perubahan per frame terlalu kecil
   - Tidak melewati threshold

6. **Refleksi dan Bayangan**
   - Cermin atau permukaan mengkilap
   - Bayangan objek ikut terdeteksi sebagai gerakan terpisah

---

## 6. Usulan Perbaikan (Classical CV)

### Perbaikan 1: Adaptive Threshold

**Masalah:** Threshold global tidak optimal untuk semua bagian frame.

**Solusi:** Gunakan `cv2.adaptiveThreshold()` yang menghitung threshold berbeda 
untuk setiap region berdasarkan neighborhood.

```python
mask = cv2.adaptiveThreshold(
    diff, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)
```

### Perbaikan 2: Mixture of Gaussians (MOG2)

**Masalah:** Single background model tidak dapat menangani background multimodal 
(misal: daun bergerak, air bergelombang).

**Solusi:** Gunakan `cv2.createBackgroundSubtractorMOG2()` yang meng-model 
setiap piksel dengan campuran distribusi Gaussian.

```python
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True  # Dapat mendeteksi bayangan!
)
mask = mog2.apply(frame)
```

### Perbaikan 3: Optical Flow untuk Validasi

**Masalah:** Perubahan intensitas bukan selalu berarti gerakan nyata.

**Solusi:** Gunakan Lucas-Kanade optical flow untuk memverifikasi arah dan 
kecepatan gerakan pada area yang terdeteksi.

```python
# Track points menggunakan optical flow
p1, status, err = cv2.calcOpticalFlowPyrLK(
    prev_gray, curr_gray, p0, None
)
# Hitung displacement untuk validasi
```

### Perbaikan 4: Shadow Detection & Removal

**Masalah:** Bayangan objek terdeteksi sebagai gerakan terpisah.

**Solusi:** Transformasi ke color space HSV, analisis ratio chromaticity 
untuk membedakan bayangan dari objek.

```python
# Bayangan biasanya memiliki:
# - Hue yang sama dengan background
# - Saturation lebih rendah
# - Value lebih rendah
```

---

## 📖 Referensi

1. Learning OpenCV (O'Reilly) - Gary Bradski & Adrian Kaehler
2. OpenCV Documentation: https://docs.opencv.org/
3. "A Survey on Moving Object Detection Methods" - Joshi & Thakore (2012)

---

## 📝 Lisensi

Proyek ini dibuat untuk tujuan edukasi. Bebas digunakan dan dimodifikasi.
