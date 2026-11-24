/* Single versioned source for SnapQC release history. */
const SnapQCReleases = [
  {
    version: "0.3.1",
    date: "2026-11-24",
    category: "Perbaikan",
    title: "Perbaikan tata letak grid agar panel verifikasi tidak meluber ke samping",
    summary:
      "Memperbaiki batas lebar minimum kolom grid dan kotak pratinjau YOLO agar seluruh teks verifikasi tampil pas tanpa scroll horizontal pada layar laptop 1100–1366px.",
  },
  {
    version: "0.3.0",
    date: "2026-11-24",
    category: "Baru",
    title: "Laporan QC Galangan (SQLite + Cetak PDF), Mode Detail Mikro SAHI 2×2, & Upgrade UI/UX",
    summary:
      "Menambahkan Formulir Laporan QC Galangan (kapal, blok/frame, joint, WPS, welder ID, standar ISO 5817/AWS D1.1/BKI, keputusan ACCEPT/REPAIR/HOLD) yang tersimpan permanen di database SQLite lengkap dengan ekspor CSV, ekspor dataset Active Learning JSON, dan cetak PDF laporan resmi A4. Menambahkan mode Detail Mikro (SAHI 2×2 tiling), drag-and-drop serta tempel Ctrl+V foto las, penempatan label kotak anti-tumpuk, tombol unduh PNG beranotasi dan salin/unduh label YOLO .txt, serta benchmark jujur Champion (YOLO11n 640px, 59,7% mAP@50) vs Challenger eksperimen (YOLO11s 960px, 56,6% mAP@50, +1,1% Defect).",
  },
  {
    version: "0.2.2",
    date: "2026-11-23",
    category: "Perbaikan",
    title: "Perbaikan pemeriksaan kesehatan container produksi (healthcheck)",
    summary:
      "Menambahkan utilitas curl di image container produksi agar healthcheck Coolify dan routing HTTPS otomatis berjalan stabil tanpa gangguan.",
  },
  {
    version: "0.2.1",
    date: "2026-11-23",
    category: "Baru",
    title: "Rilis web produksi SnapQC, container Docker, dan kesiapan OAuth PKCE",
    summary:
      "SnapQC kini dikemas sebagai container produksi Docker dengan pemuatan model ONNX di latar belakang agar halaman terbuka instan, proteksi batas laju unggahan foto, header keamanan HTTPS, dukungan rute langsung /whats-new, serta alur autentikasi OAuth 2.0 + PKCE S256 yang siap diaktifkan.",
  },
  {
    version: "0.2.0-welding-v1",
    date: "2026-11-23",
    category: "Baru",
    title: "Model Deep Learning Cacat Las v1 terlatih (YOLO11n ONNX) & deteksi foto nyata",
    summary:
      "Model mandiri SnapQC Welding v1 selesai dilatih selama 40 epoch di GPU RTX 5060 menggunakan 2.028 foto las publik CC0 (5.686 label: Bad Weld, Good Weld, Defect) dan diekspor ke ONNX 10,13 MB. Evaluasi pada 126 foto uji terpisah mencapai mAP@50 59,7% (Bad Weld 80,2%, Good Weld 71,5%, Defect kecil 27,4%). Server kini menjalankan deteksi ONNX nyata pada foto kamera maupun unggahan, tetap dengan verifikasi wajib inspector manusia.",
  },
  {
    version: "0.1.0-design",
    date: "2026-11-23",
    category: "Baru",
    title: "Prototipe desain antarmuka dan blueprint Deep Learning maritim",
    summary:
      "Rilis awal tahap desain SnapQC khusus inspeksi visual galangan kapal (lambung, pengelasan, coating/karat, fit-up, dan outfitting). Mencakup prototipe kamera interaktif, filter batas keyakinan (confidence), verifikasi inspector (Human-in-the-Loop), pratinjau ekspor label YOLO untuk Active Learning, serta kurasi dataset publik.",
  },
];
if (typeof module !== "undefined") module.exports = SnapQCReleases;
