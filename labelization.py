import os
import csv
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# labelization.py - script pour faire les prédictions sur les nouvelles images, 
# appliquer les règles de filtrage, et générer un CSV + JSON pour Label Studio

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "best_model_resnet18_flat_final1.pth"

# Dossier contenant les nouvelles images à traiter
INPUT_DIR = "data/image_test2"

# Dossier de sortie
OUTPUT_DIR = "data/predictions"

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seuil global
DEFAULT_CONFIDENCE_THRESHOLD = 0.98
MIN_MARGIN = 0.20

CLASS_NAMES = [
    "Clairs sans aurores",
    "Nuageux sans aurores",
    "Clairs avec aurores légères",
    "Nuageux avec aurores légères",
    "Clairs avec aurores fortes",
    "Nuageux avec aurores fortes",
    "Obstrués",
]

# Classes à toujours envoyer en review
ALWAYS_REVIEW_CLASSES = {
    "Clairs avec aurores légères",
    "Nuageux avec aurores légères",
}

# Seuils spécifiques par classe
CLASS_THRESHOLDS = {
    "Clairs sans aurores": 0.98,
    "Nuageux sans aurores": 0.97,
    "Clairs avec aurores légères": 0.999,
    "Nuageux avec aurores légères": 0.999,
    "Clairs avec aurores fortes": 0.98,
    "Nuageux avec aurores fortes": 0.98,
    "Obstrués": 0.95,
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Label Studio
LABEL_STUDIO_IMAGE_FIELD = "image"
LABEL_STUDIO_LABEL_FROM_NAME = "label"
LS_IMAGE_BASE_URL = "/data/upload/9/"

# =========================================================
# TRANSFORM
# =========================================================
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================================================
# MODEL
# =========================================================
def build_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(in_features, num_classes)
    )
    return model


def load_model(model_path, device):
    model = build_model(len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# =========================================================
# HELPERS
# =========================================================
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def list_images(input_dir: str):
    root = Path(input_dir)
    return sorted([p for p in root.rglob("*") if is_image_file(p)])


def ensure_output_dirs():
    for split_name in ["high_confidence", "review"]:
        for class_name in CLASS_NAMES:
            out_dir = Path(OUTPUT_DIR) / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if dst.exists():
        stem = src.stem
        suffix = src.suffix
        i = 1
        while True:
            candidate = dst_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            i += 1

    shutil.copy2(src, dst)
    return dst


def path_for_label_studio(path: Path) -> str:
    rel_path = path.as_posix()
    return LS_IMAGE_BASE_URL + rel_path


# =========================================================
# PREDICTION
# =========================================================
def predict_image(model, image_path: Path, device):
    image = Image.open(image_path).convert("RGB")
    x = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_indices = probs.argsort(descending=True)
    top1_idx = int(top_indices[0].item())
    top2_idx = int(top_indices[1].item())

    top1_conf = float(probs[top1_idx].item())
    top2_conf = float(probs[top2_idx].item())
    margin = top1_conf - top2_conf

    return {
        "pred_idx": top1_idx,
        "pred_label": CLASS_NAMES[top1_idx],
        "top1_conf": top1_conf,
        "top2_idx": top2_idx,
        "top2_label": CLASS_NAMES[top2_idx],
        "top2_conf": top2_conf,
        "margin": margin,
        "all_probs": probs.cpu().numpy(),
    }


def decide_status(pred_label: str, confidence: float, margin: float):
    if pred_label in ALWAYS_REVIEW_CLASSES:
        return "REVIEW_REQUIRED", "always_review_class"

    threshold = CLASS_THRESHOLDS.get(pred_label, DEFAULT_CONFIDENCE_THRESHOLD)

    if confidence >= threshold and margin >= MIN_MARGIN:
        return "AUTO_HIGH_CONFIDENCE", "threshold_ok"

    return "REVIEW_REQUIRED", "low_confidence_or_margin"


# =========================================================
# LABEL STUDIO JSON
# =========================================================
def build_label_studio_task(
    image_ref: str,
    predicted_label: str,
    confidence: float,
    second_label: str,
    second_confidence: float,
    margin: float,
    status: str,
    decision_reason: str,
    model_version: str = "resnet18_flat_final"
):
    task = {
        "data": {
            LABEL_STUDIO_IMAGE_FIELD: image_ref,
            "predicted_label": predicted_label,
            "top1_confidence": confidence,
            "second_label": second_label,
            "top2_confidence": second_confidence,
            "margin": margin,
            "status": status,
            "decision_reason": decision_reason,
        },
        "predictions": [
            {
                "model_version": model_version,
                "score": confidence,
                "result": [
                    {
                        "from_name": LABEL_STUDIO_LABEL_FROM_NAME,
                        "to_name": LABEL_STUDIO_IMAGE_FIELD,
                        "type": "choices",
                        "value": {
                            "choices": [predicted_label]
                        }
                    }
                ]
            }
        ]
    }
    return task


# =========================================================
# MAIN
# =========================================================
def main():
    input_images = list_images(INPUT_DIR)
    if not input_images:
        raise ValueError(f"Aucune image trouvée dans {INPUT_DIR}")

    ensure_output_dirs()

    model = load_model(MODEL_PATH, DEVICE)

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "predictions.csv"
    ls_json_path = output_root / "label_studio_predictions.json"

    rows = []
    ls_tasks = []

    count_high = 0
    count_review = 0

    for img_path in input_images:
        pred = predict_image(model, img_path, DEVICE)

        status, reason = decide_status(
            pred_label=pred["pred_label"],
            confidence=pred["top1_conf"],
            margin=pred["margin"]
        )

        if status == "AUTO_HIGH_CONFIDENCE":
            target_dir = output_root / "high_confidence" / pred["pred_label"]
            count_high += 1
        else:
            target_dir = output_root / "review" / pred["pred_label"]
            count_review += 1

        copied_path = safe_copy(img_path, target_dir)

        row = {
            "filename": img_path.name,
            "source_path": str(img_path),
            "copied_path": str(copied_path),
            "predicted_label": pred["pred_label"],
            "top1_confidence": round(pred["top1_conf"], 6),
            "second_label": pred["top2_label"],
            "top2_confidence": round(pred["top2_conf"], 6),
            "margin": round(pred["margin"], 6),
            "status": status,
            "decision_reason": reason,
        }

        for class_name, p in zip(CLASS_NAMES, pred["all_probs"]):
            row[f"prob_{class_name}"] = round(float(p), 6)

        rows.append(row)

        ls_image_ref = path_for_label_studio(copied_path)

        ls_task = build_label_studio_task(
            image_ref=ls_image_ref,
            predicted_label=pred["pred_label"],
            confidence=round(pred["top1_conf"], 6),
            second_label=pred["top2_label"],
            second_confidence=round(pred["top2_conf"], 6),
            margin=round(pred["margin"], 6),
            status=status,
            decision_reason=reason,
        )
        ls_tasks.append(ls_task)

    base_fields = [
        "filename",
        "source_path",
        "copied_path",
        "predicted_label",
        "top1_confidence",
        "second_label",
        "top2_confidence",
        "margin",
        "status",
        "decision_reason",
    ]
    prob_fields = [f"prob_{name}" for name in CLASS_NAMES]
    fieldnames = base_fields + prob_fields

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(ls_json_path, "w", encoding="utf-8") as f:
        json.dump(ls_tasks, f, ensure_ascii=False, indent=2)

    print(f"✅ Images traitées : {len(rows)}")
    print(f"✅ High confidence : {count_high}")
    print(f"✅ Review required : {count_review}")
    print(f"✅ CSV exporté : {csv_path}")
    print(f"✅ JSON Label Studio exporté : {ls_json_path}")
    print(f"✅ Dossier sortie : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()