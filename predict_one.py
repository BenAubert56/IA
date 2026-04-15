import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# Script pour faire les prédictions sur une image donnée, et afficher les résultats

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "best_model_resnet18_flat_final.pth"
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
# PREDICTION
# =========================================================
def predict_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    x = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=3)

    result = []
    for p, idx in zip(top_probs, top_indices):
        result.append({
            "label": CLASS_NAMES[int(idx.item())],
            "confidence": float(p.item())
        })

    return result


# =========================================================
# MAIN
# =========================================================
def main():
    if len(sys.argv) < 2:
        print("Usage : python predict_one.py <chemin_image>")
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Image introuvable : {image_path}")
        return

    model = load_model(MODEL_PATH, DEVICE)
    predictions = predict_image(model, image_path, DEVICE)

    print(f"\nImage : {image_path}")
    print(f"Device : {DEVICE}")
    print("\nTop 3 prédictions :")
    for i, pred in enumerate(predictions, start=1):
        print(f"{i}. {pred['label']} - {pred['confidence']:.4f}")

    print(f"\nClasse prédite : {predictions[0]['label']}")


if __name__ == "__main__":
    main()