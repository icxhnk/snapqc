"""Train SnapQC Welding Defect v1 (YOLO11n), evaluate on held-out test set, and export ONNX."""
import json
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "datasets" / "welding-defect" / "snapqc-welding.yaml"
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "prototype" / "samples"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training SnapQC Welding v1 on device={device} ({gpu_name})")

    t0 = time.time()
    model = YOLO("yolo11n.pt")
    train_res = model.train(
        data=str(DATA_YAML),
        epochs=40,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,
        patience=12,
        project=str(RUNS_DIR),
        name="welding-v1",
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        fliplr=0.5,
        mosaic=1.0,
        verbose=True,
    )
    train_seconds = round(time.time() - t0, 1)

    best_pt = Path(train_res.save_dir) / "weights" / "best.pt"
    shutil.copy2(best_pt, MODELS_DIR / "snapqc-welding-v1.pt")

    best_model = YOLO(str(best_pt))

    # Evaluate on validation split AND completely held-out test split
    val_metrics = best_model.val(data=str(DATA_YAML), split="val", imgsz=640, device=device, verbose=False)
    test_metrics = best_model.val(data=str(DATA_YAML), split="test", imgsz=640, device=device, verbose=False)

    # Export ONNX for fast local / edge inference
    exported_onnx = Path(best_model.export(format="onnx", imgsz=640, opset=12, simplify=False))
    target_onnx = MODELS_DIR / "snapqc-welding-v1.onnx"
    shutil.copy2(exported_onnx, target_onnx)

    # Verify ONNX Runtime inference on a real test image
    test_imgs = sorted((ROOT / "datasets" / "welding-defect" / "test" / "images").glob("*.jpg"))
    sess = ort.InferenceSession(str(target_onnx), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    img0 = cv2.imread(str(test_imgs[0]))
    inp = cv2.resize(img0, (640, 640))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    t_inf = time.time()
    ort_out = sess.run(None, {in_name: inp[None, ...]})
    cpu_ms = round((time.time() - t_inf) * 1000, 1)

    # Save 6 real held-out test sample images + predictions for the interactive SnapQC demo & portfolio
    names = best_model.names
    sample_manifest = []
    # Pick diverse test images that actually produce detections
    chosen = []
    for p in test_imgs:
        r = best_model.predict(str(p), imgsz=640, conf=0.35, device=device, verbose=False)[0]
        if len(r.boxes) >= 1:
            classes_seen = {names[int(c)] for c in r.boxes.cls.tolist()}
            chosen.append((p, r, classes_seen))
        if len(chosen) >= 18:
            break
    # Pick 6 covering Defect, Bad Weld, and Good Weld
    selected = []
    used = set()
    for want in ("Defect", "Bad Weld", "Good Weld"):
        for p, r, cs in chosen:
            if p.name in used:
                continue
            if want in cs:
                selected.append((p, r))
                used.add(p.name)
                if sum(1 for _, rr in selected if want in {names[int(c)] for c in rr.boxes.cls.tolist()}) >= 2:
                    break
    for p, r, _ in chosen:
        if len(selected) >= 6:
            break
        if p.name not in used:
            selected.append((p, r))
            used.add(p.name)

    for idx, (p, r) in enumerate(selected[:6], start=1):
        out_raw = SAMPLES_DIR / f"test-{idx:02d}.jpg"
        out_pred = SAMPLES_DIR / f"test-{idx:02d}-pred.jpg"
        shutil.copy2(p, out_raw)
        cv2.imwrite(str(out_pred), r.plot())
        h, w = r.orig_shape
        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item())
            dets.append({
                "clsId": cls_id,
                "label": names[cls_id],
                "conf": round(float(b.conf[0].item()), 4),
                "box": [round(x1 / w, 4), round(y1 / h, 4), round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)],
            })
        sample_manifest.append({
            "id": f"test-{idx:02d}",
            "source": p.name,
            "image": f"samples/test-{idx:02d}.jpg",
            "predictedImage": f"samples/test-{idx:02d}-pred.jpg",
            "detections": dets,
        })

    def extract_metrics(m):
        per_class = {}
        for i, cls_name in names.items():
            per_class[cls_name] = {
                "ap50": round(float(m.box.ap50[i]), 4) if i < len(m.box.ap50) else None,
                "ap50_95": round(float(m.box.ap[i]), 4) if i < len(m.box.ap) else None,
            }
        return {
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "mAP50": round(float(m.box.map50), 4),
            "mAP50_95": round(float(m.box.map), 4),
            "per_class": per_class,
        }

    report = {
        "model": "SnapQC Welding v1 (YOLO11n)",
        "dataset": "rikkarth/welding-defect-object-detection (CC0 Public Domain)",
        "hardware": gpu_name,
        "epochs_requested": 40,
        "train_seconds": train_seconds,
        "onnx_path": "models/snapqc-welding-v1.onnx",
        "onnx_size_mb": round(target_onnx.stat().st_size / (1024 * 1024), 2),
        "onnx_cpu_latency_ms": cpu_ms,
        "onnx_output_shape": list(ort_out[0].shape),
        "classes": names,
        "validation": extract_metrics(val_metrics),
        "test_held_out": extract_metrics(test_metrics),
        "samples": sample_manifest,
    }
    (MODELS_DIR / "metrics-v1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Also copy confusion matrix & results plot if generated
    for plot_name in ("confusion_matrix_normalized.png", "results.png"):
        src_plot = Path(train_res.save_dir) / plot_name
        if src_plot.exists():
            shutil.copy2(src_plot, MODELS_DIR / plot_name)

    print("TRAINING_COMPLETE_JSON:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
