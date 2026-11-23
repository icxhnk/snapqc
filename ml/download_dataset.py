"""Download and validate the CC0 public welding defect dataset from Hugging Face."""
import json
from collections import Counter
from pathlib import Path
from huggingface_hub import snapshot_download
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "welding-defect"
SUMMARY_PATH = ROOT / "ml" / "dataset-summary.json"
REPO_ID = "rikkarth/welding-defect-object-detection"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} -> {DATA_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        ignore_patterns=["*.DS_Store", "coco/*"],
    )

    orig_yaml = DATA_DIR / "data.yaml"
    cfg = yaml.safe_load(orig_yaml.read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, list):
        names_dict = {i: n for i, n in enumerate(names)}
    else:
        names_dict = {int(k): v for k, v in names.items()}

    # Write clean local YAML with explicit absolute paths
    local_yaml = DATA_DIR / "snapqc-welding.yaml"
    clean_cfg = {
        "path": str(DATA_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names_dict),
        "names": names_dict,
    }
    local_yaml.write_text(yaml.safe_dump(clean_cfg, sort_keys=False), encoding="utf-8")

    summary = {"repo": REPO_ID, "license": "CC0-1.0 (Public Domain)", "classes": names_dict, "splits": {}}
    for split in ("train", "valid", "test"):
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"
        imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        counts = Counter()
        empty_labels = 0
        for img in imgs:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                empty_labels += 1
                continue
            lines = [ln.strip() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                empty_labels += 1
            for ln in lines:
                cls_id = int(float(ln.split()[0]))
                counts[names_dict.get(cls_id, str(cls_id))] += 1
        summary["splits"][split] = {
            "images": len(imgs),
            "annotations": sum(counts.values()),
            "background_images": empty_labels,
            "per_class": dict(counts),
        }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
