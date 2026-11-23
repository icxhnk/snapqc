# SnapQC — Maritime Deep Learning Blueprint & Dataset Roadmap

**Fokus Domain:** Inspeksi Visual Galangan Kapal (*Hull / Lambung, Visual Welding, Coating & Corrosion, Fit-Up, Outfitting*)
**Pendekatan AI:** Model Deep Learning mandiri (*Custom Object Detection / Segmentation*) yang dijalankan ringan di perangkat (*Edge / Browser ONNX*) maupun server lokal, dengan verifikasi wajib oleh inspector manusia (*Human-in-the-Loop*).

---

## 1. Mengapa Model Deep Learning Mandiri (Bukan Sekadar Prompt API)?

Dalam inspeksi galangan kapal, objek dan pola visual yang diperiksa **berulang terus-menerus** di setiap kapal dan setiap periode docking:

1. **Pola visual berulang & khas**: undercut las, porositas, retak permukaan, blister cat, karat/pitting pelat lambung, hingga celah *fit-up* memiliki tekstur visual yang sangat cocok dipelajari model *Computer Vision*.
2. **Sinyal internet di dry dock & dalam tangki sering lemah**: model ringan (seperti **YOLOv8n / YOLOv11n** yang diekspor ke **ONNX / WebGPU / ONNX Runtime Web**, ukuran ~6–12 MB) dapat berjalan **langsung di ponsel/browser secara offline**, tanpa biaya token per foto.
3. **Konsistensi & privasi foto kapal**: foto lambung atau proyek milik *ship owner* tidak harus dikirim ke penyedia AI pihak ketiga.
4. **Makin pintar lewat *Active Learning***: setiap koreksi kotak dan label oleh inspector di SnapQC dapat diekspor dalam format YOLO untuk melatih ulang versi model berikutnya.

---

## 2. Kurasi Dataset Publik (Sumber Awal Pelatihan)

Berikut dataset publik nyata yang telah diidentifikasi sebagai fondasi pelatihan bertahap (verifikasi lisensi masing-masing sumber sebelum penggunaan komersial):

| Tahap | Nama Dataset Publik | Sumber | Cakupan Visual untuk SnapQC | Catatan Lisensi & Penggunaan |
|---|---|---|---|---|
| **Fase 1A — Lambung & Korosi** | **MaVeCoDD** *(Marine Vessel Hull Corrosion in Dry-Dock Images)* | Mendeley Data | Foto asli lambung kapal saat dry-docking dengan anotasi manual ahli untuk area korosi & karat (disertai konversi COCO JSON) | Sangat relevan langsung dengan kondisi pencahayaan dry dock; cek syarat atribusi akademik Mendeley |
| **Fase 1A — Korosi & Propeller** | **Corrosion Detect Dataset** & **Ship Hull/Propeller Corrosion (~1.819 foto)** | Kaggle & GitHub (`pjsun2012/Phase5_Capstone-Project`) | Korosi struktur, area kabin, lambung kapal, dan propeller berformat YOLO / klasifikasi | Cocok untuk menambah variasi sudut pengambilan gambar korosi maritim |
| **Fase 1B — Cacat Las Visual** | **Welding Defect Dataset** (`welding-defect-tpfif`) & **Welding Defect Object Detection** | Roboflow Universe & Kaggle | Indikasi las permukaan: *crack*, *porosity/air-hole*, *undercut/bite-edge*, *broken-arc*, *bad weld* vs *good weld* (format YOLO siap latih) | Sangat cocok untuk baseline detektor visual welding di SnapQC |
| **Pre-training Permukaan Baja** | **NEU-DET** (1.800 foto) & **Severstal Steel Defect** (~18.000 foto) | Northeastern Univ. / Kaggle / Hugging Face | Tekstur cacat permukaan pelat baja termasuk *pitted surface*, *scratches*, *inclusion*, dan segmentasi mask | Dipakai untuk *pre-training / transfer learning* pengenalan tekstur baja sebelum *fine-tuning* ke foto galangan berwarna |
| **Fase 2 — Fit-Up & Outfitting** | **Dataset Internal / Active Learning SnapQC** | Foto lapangan berizin milik tim galangan | Celah *fit-up* bracket, keselarasan sambungan, kelengkapan handrail, pipa, & penandaan outfitting | Dataset publik minim di kategori ini; SnapQC menyediakan fitur anotasi cepat + ekspor format YOLO dari temuan yang sudah diverifikasi inspector |

---

## 3. Taksonomi Kelas Deteksi (SnapQC Maritime Taxonomy v1)

Model dirancang bertahap agar akurasi tetap tinggi dan mudah dievaluasi:

### Grup A — Model v1 (Langsung Dilatih dari Dataset Publik)
1. `weld_undercut` — Indikasi undercut / bite-edge pada jalur las
2. `weld_porosity` — Indikasi porositas / lubang udara permukaan las
3. `weld_crack` — Indikasi retak permukaan pada sambungan las
4. `hull_corrosion` — Karat aktif / *pitting* pada pelat lambung & struktur
5. `coating_blister` — Cat mengelupas, gelembung (*blistering*), atau kerusakan *coating*

### Grup B — Model v2 (Diperluas via Active Learning Lapangan)
6. `fitup_misalignment` — Indikasi celah atau ketidakselarasan *fit-up*
7. `outfitting_damage` — Kerusakan visual handrail, penetrasi pipa, atau dudukan outfitting

---

## 4. Arsitektur Pipeline Pelatihan & Inferensi

```text
[Dataset Publik: MaVeCoDD + Welding YOLO + Corrosion]
        │
        ▼  (Pembersihan label, penyatuan kelas, augmentasi cahaya dry-dock/basah/pantulan)
[Training Pipeline — PyTorch + Ultralytics YOLOv8n / YOLOv11n]
        │   • Input: 640×640 RGB
        │   • Augmentasi: brightness/contrast jitter, blur gerak tangan, rotasi ringan
        │   • Evaluasi wajib: mAP@50, mAP@50-95, Precision/Recall per kelas, Confusion Matrix
        ▼
[Ekspor Model Ringan — ONNX (FP16 / INT8, ~6–12 MB)]
        │
        ├──► [Inferensi Edge di Browser HP (ONNX Runtime Web / WebGPU)] → Jalan cepat & tetap bisa offline di dry dock
        └──► [Inferensi Server (opsional)] → Untuk pemrosesan batch foto resolusi tinggi
        │
        ▼
[Antarmuka Verifikasi Inspector (Human-in-the-Loop)]
        • Inspector menyetujui / menolak / menggeser kotak deteksi AI
        • Keputusan akhir & standar penerimaan tetap di tangan inspector berwenang
        • Foto terverifikasi dapat diekspor ke format YOLO (.txt) untuk melatih ulang Model v(n+1)
```

---

## 5. Tata Kelola Risiko & Batasan Kejujuran (Safety Guardrails)

- **Asisten Visual, Bukan Pengambil Keputusan Class**: setiap hasil deteksi SnapQC wajib menampilkan skor keyakinan (*confidence score*) dan peringatan jelas bahwa deteksi AI hanya alat bantu skrining awal — bukan pengganti pengukuran fisik, NDT, atau persetujuan surveyor class/flag.
- **Tantangan Lapangan Nyata yang Diantisipasi**: pantulan air pada pelat basah, bayangan dok/staging, dan biofouling (teritip/lumut) yang menyerupai karat akan diuji khusus sebagai *hard negative samples*.
- **Target Evaluasi Sebelum Rilis Model v1**: Precision ≥ 0.80, Recall ≥ 0.85 untuk kelas prioritas tinggi (`weld_crack`, `hull_corrosion`) pada *validation split* independen yang tidak bercampur dengan data latih.
