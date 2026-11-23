# SnapQC — Maritime Welding AI Visual Inspection (YOLO11n ONNX)

**Live Web App:** `https://snapqc.icxhnk.com` (Portfolio view: `/portfolio`, Release notes: `/whats-new`)  
**Repository:** `https://github.com/icxhnk/snapqc`  
**Owner:** Miftahul Ikhsan Tahir · **Release:** `v0.2.1`

**SnapQC** is a standalone camera-assisted visual inspection web application for **shipyard and maritime quality control** powered by a custom-trained **YOLO11n ONNX** deep learning model (Phase 1: **Welding Defect Detection**).

Unlike project-control ledgers, SnapQC focuses on **fast, consistent field screening**: an inspector points a phone camera or uploads a weld photo, the ONNX Runtime engine highlights potential **Bad Weld**, **Good Weld**, and **Defect** regions in ~32 ms on CPU, and the inspector remains in control via **Human-in-the-Loop** approve/reject verification and YOLO-format Active Learning label export.

> **QC Safety Principle:** SnapQC is a visual screening assistant. Final acceptance, weld-gauge verification, NDT testing, and class/flag approval always remain with certified human inspectors.

---

## Real Trained Model Performance (Phase 1 — Welding v1)

Trained locally for 40 epochs on an **NVIDIA GeForce RTX 5060** using the CC0 public dataset [`rikkarth/welding-defect-object-detection`](https://huggingface.co/datasets/rikkarth/welding-defect-object-detection) (**2,028 images**, **5,686 bounding boxes**: train 1,619 / val 283 / held-out test 126).

| Split | Images | mAP@50 | Precision | Recall | Bad Weld AP@50 | Good Weld AP@50 | Defect AP@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 283 | 64.8% | 67.9% | 59.5% | 74.6% | 77.0% | 42.8% |
| **Held-Out Test** | **126** | **59.7%** | **48.0%** | **63.8%** | **80.2%** | **71.5%** | **27.4%** |

- **Exported artifact:** [`models/snapqc-welding-v1.onnx`](models/snapqc-welding-v1.onnx) (`10.13 MB`, runs offline on CPU via ONNX Runtime at **~32 ms/image**).
- **Honest engineering note:** Bead-level classes (`Bad Weld` 80.2%, `Good Weld` 71.5%) perform strongly on held-out images, while micro-defects (`Defect`, 20–40 px local porosity/slag spots) drop to 27.4% at 640px resize. Planned v2 upgrades include high-resolution 1024px training / tiled inference (SAHI) plus Phase 2 hull corrosion & coating detection.

---

## Project Structure

- [`server.py`](server.py): Production & local Python HTTP server with instant startup, background ONNX loading, `/api/detect`, `/healthz`, security headers, per-IP rate limiting, and OAuth 2.0 Authorization Code + PKCE S256 (`IDENTITY_*`) integration.
- [`prototype/index.html`](prototype/index.html): Responsive desktop & mobile inspection UI with live camera capture, photo upload, interactive confidence threshold slider, Human-in-the-Loop approve/reject review, YOLO label export, model evaluation charts, and **What's New** (`/whats-new`).
- [`portfolio/index.html`](portfolio/index.html) & [`portfolio/RESUME-KIT.md`](portfolio/RESUME-KIT.md): Printable case-study sheet and bilingual resume bullets.
- [`ml/download_dataset.py`](ml/download_dataset.py) & [`ml/train_welding.py`](ml/train_welding.py): Reproducible CC0 dataset downloader and YOLO11n training + ONNX export scripts.
- [`models/`](models/): Exported ONNX model, [`metrics-v1.json`](models/metrics-v1.json), training curves, and normalized confusion matrix.
- [`Dockerfile`](Dockerfile) & [`requirements-runtime.txt`](requirements-runtime.txt): Production container build (non-root user, `/healthz` healthcheck, port `8000`).
- [`tests/test_server.py`](tests/test_server.py): Automated unit & integration tests for ONNX inference, `/whats-new`, security headers, rate limiting, and OAuth 2.0 PKCE + RS256 JWKS verification.

---

## Quick Start (Local)

```powershell
# Windows (uses pre-built .venv)
.\Start-SnapQC.ps1 -Port 57840
# Open http://127.0.0.1:57840
```

Or with Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-runtime.txt
python -u server.py
```

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Docker Build & Run

```bash
docker build -t snapqc:0.2.1 .
docker run --rm -p 8000:8000 -e PUBLIC_DEMO_ENABLED=true snapqc:0.2.1
# Healthcheck: http://127.0.0.1:8000/healthz
```

## Configuration (`.env.example`)

See [`.env.example`](.env.example). Canonical `IDENTITY_*` variables support OAuth 2.0 Authorization Code + PKCE S256 (`RS256` JWKS verification, `sub` + `roles` enforcement, and fail-closed behavior when `IDENTITY_PLATFORM_ENABLED=false` and `PUBLIC_DEMO_ENABLED=false`). Never commit secrets into Git or frontend bundles.
