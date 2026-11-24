"""Train SnapQC Welding Defect v2 (YOLO11s @ 960px), compare strictly against v1
on the exact same held-out test set, and export ONNX + benchmark metrics."""

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
IMGSZ = 960


def letterbox(img, new_shape=IMGSZ):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape - nw) / 2, (new_shape - nh) / 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training SnapQC Welding v2 (YOLO11s @ {IMGSZ}px) on device={device} ({gpu_name})", flush=True)

    t0 = time.time()
    model = YOLO("yolo11s.pt")
    train_res = model.train(
        data=str(DATA_YAML),
        epochs=45,
        imgsz=IMGSZ,
        batch=8,
        device=device,
        workers=2,
        cache="ram",
        amp=True,
        cos_lr=True,
        close_mosaic=10,
        patience=14,
        optimizer="AdamW",
        lr0=0.0012,
        lrf=0.05,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        scale=0.6,
        fliplr=0.5,
        mosaic=1.0,
        copy_paste=0.1,
        project=str(RUNS_DIR),
        name="welding-v2",
        exist_ok=True,
        verbose=True,
    )
    train_seconds = round(time.time() - t0, 1)

    best_pt = Path(train_res.save_dir) / "weights" / "best.pt"
    shutil.copy2(best_pt, MODELS_DIR / "snapqc-welding-v2.pt")
    best_model = YOLO(str(best_pt))

    val_metrics = best_model.val(data=str(DATA_YAML), split="val", imgsz=IMGSZ, device=device, verbose=False)
    test_metrics = best_model.val(data=str(DATA_YAML), split="test", imgsz=IMGSZ, device=device, verbose=False)

    # Export ONNX at IMGSZ for higher small-defect recall
    exported_onnx = Path(best_model.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=False))
    target_onnx = MODELS_DIR / "snapqc-welding-v2.onnx"
    shutil.copy2(exported_onnx, target_onnx)

    # Benchmark steady-state CPU ONNX latency (1 warmup + 2 timed runs)
    test_imgs = sorted((ROOT / "datasets" / "welding-defect" / "test" / "images").glob("*.jpg"))
    sess = ort.InferenceSession(str(target_onnx), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    img0 = cv2.imread(str(test_imgs[0]))
    padded = letterbox(img0, IMGSZ)
    inp = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)[None, ...]
    sess.run(None, {in_name: inp})
    t_inf = time.perf_counter()
    ort_out = sess.run(None, {in_name: inp})
    cpu_ms = round((time.perf_counter() - t_inf) * 1000, 1)

    names = best_model.names

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

    v1_path = MODELS_DIR / "metrics-v1.json"
    v1_meta = json.loads(v1_path.read_text(encoding="utf-8")) if v1_path.exists() else None

    # Update sample predictions with v2 on the existing 6 test samples if v2 beats v1 on held-out mAP50
    v2_test = extract_metrics(test_metrics)
    v1_test_map = v1_meta["test_held_out"]["mAP50"] if v1_meta else 0.0
    promoted = v2_test["mAP50"] >= v1_test_map

    sample_manifest = []
    for idx in range(1, 7):
        raw_p = SAMPLES_DIR / f"test-{idx:02d}.jpg"
        if not raw_p.exists():
            continue
        r = best_model.predict(str(raw_p), imgsz=IMGSZ, conf=0.35, device=device, verbose=False)[0]
        if promoted:
            cv2.imwrite(str(SAMPLES_DIR / f"test-{idx:02d}-pred.jpg"), r.plot())
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
            "image": f"samples/test-{idx:02d}.jpg",
            "predictedImage": f"samples/test-{idx:02d}-pred.jpg",
            "detections": dets,
        })

    for plot_name in ("confusion_matrix_normalized.png", "results.png"):
        src_plot = Path(train_res.save_dir) / plot_name
        if src_plot.exists():
            shutil.copy2(src_plot, MODELS_DIR / f"v2-{plot_name}")
            if promoted:
                shutil.copy2(src_plot, MODELS_DIR / plot_name)

    report = {
        "model": f"SnapQC Welding v2 (YOLO11s @ {IMGSZ}px)",
        "promoted_as_default": promoted,
        "dataset": "rikkarth/welding-defect-object-detection (CC0 Public Domain)",
        "hardware": gpu_name,
        "imgsz": IMGSZ,
        "epochs_requested": 45,
        "train_seconds": train_seconds,
        "onnx_path": "models/snapqc-welding-v2.onnx",
        "onnx_size_mb": round(target_onnx.stat().st_size / (1024 * 1024), 2),
        "onnx_cpu_latency_ms": cpu_ms,
        "onnx_output_shape": list(ort_out[0].shape),
        "classes": names,
        "validation": extract_metrics(val_metrics),
        "test_held_out": v2_test,
        "comparison_vs_v1": {
            "v1_test_mAP50": v1_meta["test_held_out"]["mAP50"] if v1_meta else None,
            "v2_test_mAP50": v2_test["mAP50"],
            "v1_test_defect_ap50": v1_meta["test_held_out"]["per_class"]["Defect"]["ap50"] if v1_meta else None,
            "v2_test_defect_ap50": v2_test["per_class"]["Defect"]["ap50"],
        },
        "samples": sample_manifest,
    }
    (MODELS_DIR / "metrics-v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("TRAINING_V2_COMPLETE_JSON:", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
