# SnapQC — Resume & Interview Kit (Phase 1 Trained Model: Welding v1)

**Pemilik Proyek:** Miftahul Ikhsan Tahir
**Produk:** SnapQC — Custom Deep Learning Visual Welding Inspection & Maritime QC Reporting (`v0.3.0`)
**Live App:** `https://snapqc.icxhnk.com` · **GitHub (Public):** `https://github.com/icxhnk/snapqc`
**Model Produksi (Champion):** `YOLO11n @ 640px` → `ONNX` (10,13 MB) + mode inferensi **SAHI 2×2 tiling**; dibandingkan secara otomatis melawan Challenger `YOLO11s @ 960px` (36,34 MB) di **NVIDIA GeForce RTX 5060**
**Dataset:** `rikkarth/welding-defect-object-detection` (CC0 Public Domain · 2.028 foto nyata · 5.686 label kotak)

---

## 1. Poin Siap Tempel untuk CV (Bahasa Indonesia)

**SnapQC — Deteksi Cacat Las Visual Berbasis Model Deep Learning Mandiri** *(Proyek Mandiri · PyTorch CUDA, YOLO11n, ONNX Runtime · 2026)*
- Melatih model *Computer Vision* mandiri (**YOLO11n**, 40 epoch di GPU RTX 5060) pada **2.028 foto las publik CC0 (5.686 label)** untuk membedakan jalur las baik (*Good Weld*), jalur las bermasalah (*Bad Weld*), dan titik cacat lokal (*Defect*).
- Mengevaluasi performa secara objektif pada **126 foto uji terpisah (*held-out test set*)**, mencapai **AP@50 80,2% untuk *Bad Weld***, **71,5% untuk *Good Weld***, dan **mAP@50 total 59,7%** (validasi 64,8%), sekaligus memetakan keterbatasan cacat mikro (*Defect* 27,4%) sebagai target peningkatan v2.
- Mengekspor model ke **ONNX (10,13 MB)** dan membangun server deteksi lokal berbasis **ONNX Runtime** + antarmuka web kamera dengan kendali verifikasi inspector (*Human-in-the-Loop*) dan ekspor label YOLO untuk *Active Learning* tanpa mengirim foto proyek ke API cloud pihak ketiga.

---

## 2. Poin Siap Tempel untuk CV (English — ATS Optimized)

**SnapQC — Custom Deep Learning Visual Welding Inspection Assistant** *(Independent Project · PyTorch CUDA, YOLO11n, ONNX Runtime · 2026)*
- Trained a custom **YOLO11n** object detection model (40 epochs on NVIDIA RTX 5060) on **2,028 CC0 public welding images (5,686 bounding boxes)** covering *Bad Weld*, *Good Weld*, and localized surface *Defect* classes.
- Evaluated strictly on a **126-image held-out test split**, achieving **80.2% AP@50 on Bad Weld**, **71.5% AP@50 on Good Weld**, and **59.7% overall mAP@50** (64.8% validation), while diagnosing small-object defect recall limitations for targeted v2 high-resolution tiling (*SAHI*).
- Exported a lightweight **10.13 MB ONNX** runtime model and built a local **ONNX Runtime** camera/upload inference web interface featuring **Human-in-the-Loop** inspector verification and YOLO label export for continuous *Active Learning* without cloud API dependencies.

---

## 3. Cara Menjelaskan Hasil Jujur Ini Saat Wawancara (Sangat Mengesankan Senior Engineer!)

- **Kenapa angka `Bad Weld` (80,2%) dan `Good Weld` (71,5%) jauh lebih tinggi dari `Defect` (27,4%)?**
  *"Karena saya menguji model pada 126 foto uji yang benar-benar terpisah. Cacat mikro seperti lubang porositas 20–40 piksel jauh lebih kecil dibanding keseluruhan jalur las saat di-resize ke 640×640. Justru dari evaluasi jujur ini saya tahu langkah perbaikan v2 yang tepat: gunakan resolusi lebih tinggi / metode potong petak (SAHI tiling) dan kumpulkan koreksi inspector lewat fitur Active Learning."*
  *(Jawaban seperti ini langsung membedakan Anda dari kandidat yang hanya mengklaim "akurasi 99%" tanpa paham evaluasi ML!)*
