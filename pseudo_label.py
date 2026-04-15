import os
import csv
import json
import shutil
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm


# Script pour faire les prédictions sur les nouvelles images,
# appliquer les règles de filtrage, et générer un CSV + JSON pour Label Studio
# Le script charge un modèle pré-entraîné, parcourt les images d'un dossier donné, 
# fait des prédictions, applique des règles de filtrage basées sur la confiance et la marge 
# entre les classes, et copie les images retenues dans des dossiers organisés par classe. 

# Un CSV détaillé est généré pour toutes les images, et un JSON spécifique est créé pour les 
# images retenues, prêt à être importé dans Label Studio.
# Les règles de filtrage sont configurables : on peut définir des seuils de confiance spécifiques 
# par classe, une marge minimale entre la première et la deuxième prédiction, un quota maximum d'images 
# retenues par classe, et même des classes à exclure complètement du pseudo-labeling automatique. 
# Le script est conçu pour être facilement adaptable à différents scénarios de pseudo-labeling en 
# ajustant simplement les paramètres de configuration.

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "best_model_resnet18_flat_final.pth"

# Dossier des images non labellisées
INPUT_DIR = "unlabeled_data"

# Dossier de sortie pour les pseudo-labels retenus
OUTPUT_DIR = "pseudo_labels_dataset"

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Clairs sans aurores",
    "Nuageux sans aurores",
    "Clairs avec aurores légères",
    "Nuageux avec aurores légères",
    "Clairs avec aurores fortes",
    "Nuageux avec aurores fortes",
    "Obstrués",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# =========================================================
# REGLES DE FILTRAGE
# =========================================================

# Seuil unique pour toutes les classes
DEFAULT_CONFIDENCE_THRESHOLD = 0.98

# Marge minimale entre top1 et top2
MIN_MARGIN = 0.20

# Quota max de pseudo-labels retenus par classe
MAX_PER_CLASS = {
    "Clairs sans aurores": 10000,
    "Nuageux sans aurores": 10000,
    "Clairs avec aurores légères": 5000,
    "Nuageux avec aurores légères": 5000,
    "Clairs avec aurores fortes": 10000,
    "Nuageux avec aurores fortes": 10000,
    "Obstrués": 10000,
}

# Plus de seuils spécifiques par classe
CLASS_THRESHOLDS = {}

# Classes qu'on refuse de pseudo-labelliser automatiquement
ALWAYS_REJECT_CLASSES = set()


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
# IO HELPERS
# =========================================================
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def list_images(input_dir: str):
    root = Path(input_dir)
    return sorted([p for p in root.rglob("*") if is_image_file(p)])


def ensure_output_dirs():
    root = Path(OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        (root / class_name).mkdir(parents=True, exist_ok=True)


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


def accept_prediction(pred_label: str, confidence: float, margin: float, current_count: int):
    if pred_label in ALWAYS_REJECT_CLASSES:
        return False, "class_rejected"

    threshold = CLASS_THRESHOLDS.get(pred_label, DEFAULT_CONFIDENCE_THRESHOLD)
    quota = MAX_PER_CLASS.get(pred_label, 0)

    if current_count >= quota:
        return False, "class_quota_reached"

    if confidence < threshold:
        return False, "low_confidence"

    if margin < MIN_MARGIN:
        return False, "low_margin"

    return True, "accepted"


# =========================================================
# MAIN
# =========================================================
def main():
    input_images = list_images(INPUT_DIR)
    if not input_images:
        raise ValueError(f"Aucune image trouvée dans {INPUT_DIR}")

    ensure_output_dirs()
    model = load_model(MODEL_PATH, DEVICE)

    rows = []
    accepted_rows = []
    class_counts = defaultdict(int)

    progress_bar = tqdm(input_images, desc="Pseudo-labelisation", unit="image")

    for img_path in progress_bar:
        pred = predict_image(model, img_path, DEVICE)

        current_count = class_counts[pred["pred_label"]]
        accepted, reason = accept_prediction(
            pred_label=pred["pred_label"],
            confidence=pred["top1_conf"],
            margin=pred["margin"],
            current_count=current_count
        )

        row = {
            "filename": img_path.name,
            "source_path": str(img_path),
            "predicted_label": pred["pred_label"],
            "top1_confidence": round(pred["top1_conf"], 6),
            "second_label": pred["top2_label"],
            "top2_confidence": round(pred["top2_conf"], 6),
            "margin": round(pred["margin"], 6),
            "accepted": accepted,
            "decision_reason": reason,
            "copied_path": "",
        }

        for class_name, p in zip(CLASS_NAMES, pred["all_probs"]):
            row[f"prob_{class_name}"] = round(float(p), 6)

        if accepted:
            target_dir = Path(OUTPUT_DIR) / pred["pred_label"]
            copied_path = safe_copy(img_path, target_dir)
            row["copied_path"] = str(copied_path)
            class_counts[pred["pred_label"]] += 1
            accepted_rows.append({
                "image_path": str(copied_path),
                "label_name": pred["pred_label"],
                "label_idx": CLASS_NAMES.index(pred["pred_label"]),
                "filename": copied_path.name,
                "confidence": round(pred["top1_conf"], 6),
                "margin": round(pred["margin"], 6),
            })

        rows.append(row)

        progress_bar.set_postfix({
            "accepted": len(accepted_rows),
            "last_class": pred["pred_label"],
            "conf": f"{pred['top1_conf']:.3f}"
        })

    # CSV complet
    csv_path = Path(OUTPUT_DIR) / "pseudo_labels.csv"
    base_fields = [
        "filename",
        "source_path",
        "predicted_label",
        "top1_confidence",
        "second_label",
        "top2_confidence",
        "margin",
        "accepted",
        "decision_reason",
        "copied_path",
    ]
    prob_fields = [f"prob_{name}" for name in CLASS_NAMES]
    fieldnames = base_fields + prob_fields

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # JSON des pseudo-labels retenus
    json_path = Path(OUTPUT_DIR) / "pseudo_labels.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(accepted_rows, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Images analysées : {len(rows)}")
    print(f"✅ Images retenues : {len(accepted_rows)}")
    print(f"✅ CSV exporté : {csv_path}")
    print(f"✅ JSON exporté : {json_path}")
    print("\nRépartition des pseudo-labels retenus :")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {class_counts[class_name]}")


if __name__ == "__main__":
    main()