# Sistem Deteksi Gerak dengan Computer Vision Klasik

> **Target Pembaca:** Mahasiswa tingkat sarjana  
> **Metode:** Frame Differencing & Background Subtraction  
> **Library:** OpenCV (Python)

---

## ✨ Fitur Baru v2.0

| Fitur | Deskripsi | Shortcut |
|-------|-----------|----------|
| 🎥 **Webcam Support** | Deteksi langsung dari kamera | `--webcam` |
| 📍 **Motion Tracking** | Melacak objek dengan trails | `T` |
| 🌡️ **Motion Heatmap** | Visualisasi area panas | `H` |
| 📹 **Recording** | Simpan hasil deteksi | `R` |
| 📸 **Snapshot** | Ambil screenshot | `S` |
| 🔔 **Audio Alert** | Bunyi saat gerakan | `A` |
| 🎨 **Modern UI** | Tampilan keren dengan glow | - |

---

## 📚 Daftar Isi

1. [Cara Menjalankan](#cara-menjalankan)
2. [Keyboard Shortcuts](#keyboard-shortcuts)
3. [Penjelasan Algoritma](#penjelasan-algoritma)
4. [Struktur Proyek](#struktur-proyek)
5. [Parameter](#parameter)

---

## Cara Menjalankan

### Prasyarat

```bash
# Buat virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Dari Video File

```bash
python main.py --video path/to/video.mp4
```

### Dari Webcam

```bash
# Webcam default (index 0)
python main.py --webcam

# Webcam tertentu
python main.py --webcam 1
```

### Dengan Fitur Tambahan

```bash
# Aktifkan tracking dan heatmap
python main.py --webcam --tracking --heatmap

# Langsung recording
python main.py --webcam --record

# Dengan audio alert
python main.py --webcam --alert

# Kombinasi semua fitur
python main.py --webcam --tracking --heatmap --alert --record
```

---

## Keyboard Shortcuts

| Key | Fungsi |
|-----|--------|
| `Q` | Keluar program |
| `D` | Toggle debug view (4 panel) |
| `H` | Toggle heatmap overlay |
| `T` | Toggle motion tracking |
| `S` | Ambil snapshot |
| `R` | Start/stop recording |
| `Space` | Pause/resume |
| `+`/`-` | Adjust threshold |
| `A` | Toggle audio alert |
| `L` | Toggle keyboard legend |

---

## Penjelasan Algoritma

### Pipeline Deteksi Gerak

```
┌──────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐
│  Frame   │ →  │ Grayscale │ →  │   Blur     │ →  │  Diff    │
│  Input   │    │ Conversion│    │ (Gaussian) │    │ (BG Sub) │
└──────────┘    └───────────┘    └────────────┘    └──────────┘
                                                        ↓
┌──────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐
│ Display  │ ←  │  Bounding │ ←  │ Morphology │ ←  │Threshold │
│ Result   │    │    Box    │    │(Erode+Dil) │    │ (Binary) │
└──────────┘    └───────────┘    └────────────┘    └──────────┘
```

### Metode Background Subtraction

1. **Frame Differencing** - Bandingkan frame berturutan
2. **Running Average** - Background adaptif (recommended)
3. **MOG2** - Mixture of Gaussians untuk scene kompleks

---

## Struktur Proyek

```
camtask/
├── main.py                 # Program utama
├── requirements.txt
├── src/
│   ├── video_reader.py     # Input video/webcam
│   ├── preprocessing.py    # Grayscale, blur
│   ├── background_model.py # Background subtraction
│   ├── motion_detector.py  # Deteksi kontur
│   ├── visualization.py    # UI modern
│   ├── motion_tracker.py   # Object tracking
│   ├── heatmap.py          # Motion heatmap
│   ├── recorder.py         # Video recording
│   └── alert.py            # Audio alerts
├── output/                 # Recordings & snapshots
└── README.md
```

---

## Parameter

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `--video` | - | Path ke file video |
| `--webcam` | 0 | Index webcam |
| `--method` | running_average | frame_diff / running_average / mog2 |
| `--threshold` | 25 | Nilai threshold (0-255) |
| `--min-area` | 500 | Area minimum kontur (px²) |
| `--blur-size` | 21 | Ukuran kernel blur |
| `--alpha` | 0.01 | Learning rate background |
| `--tracking` | off | Aktifkan tracking |
| `--heatmap` | off | Aktifkan heatmap |
| `--record` | off | Langsung recording |
| `--alert` | off | Aktifkan audio alert |
| `--output-dir` | output | Direktori output |

---

## Tips Penggunaan

### Threshold Terlalu Sensitif?
```bash
python main.py --webcam --threshold 40
```
Atau tekan `+` saat program berjalan.

### Ingin Tracking Lebih Smooth?
Gabungkan dengan heatmap untuk lihat pola:
```bash
python main.py --webcam --tracking --heatmap
```

### Recording Otomatis?
```bash
python main.py --webcam --record
```
File disimpan di folder `output/` dengan timestamp.

---

## 📖 Referensi

1. Learning OpenCV (O'Reilly) - Gary Bradski & Adrian Kaehler
2. OpenCV Documentation: https://docs.opencv.org/

---

## 📝 Lisensi

Proyek ini dibuat untuk tujuan edukasi. Bebas digunakan dan dimodifikasi.
