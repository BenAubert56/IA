import json
import shutil
from pathlib import Path

# Trier les images dans des dossiers selon les labels du JSON

# 📁 Chemins
JSON_FILE = "javoue.json"   # 👉 ton fichier JSON
SOURCE_DIR = Path("data/image_test")
DEST_BASE_DIR = Path("data/pseudo_labels_dataset")     # dossiers de labels créés ici

def clean_label(label):
    """Nettoie le nom du dossier (optionnel mais recommandé)"""
    return label.replace("/", "-").strip()

def main():
    # Charger le JSON
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Si c'est une liste de tâches
    if isinstance(data, list):
        tasks = data
    else:
        tasks = [data]

    for task in tasks:
        file_name = task.get("file_upload")

        if not file_name:
            continue

        annotations = task.get("annotations", [])
        if not annotations:
            continue

        for ann in annotations:
            results = ann.get("result", [])

            for res in results:
                if res.get("type") != "choices":
                    continue

                choices = res.get("value", {}).get("choices", [])
                if not choices:
                    continue

                label = clean_label(choices[0])

                # 📁 Dossier cible = label
                dest_dir = DEST_BASE_DIR / label
                dest_dir.mkdir(parents=True, exist_ok=True)

                src_path = SOURCE_DIR / file_name
                dest_path = dest_dir / file_name

                if not src_path.exists():
                    print(f"❌ Image introuvable: {src_path}")
                    continue

                # Éviter écrasement
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{Path(file_name).stem}_{counter}{Path(file_name).suffix}"
                    counter += 1

                shutil.move(str(src_path), str(dest_path))
                print(f"✅ Déplacé: {file_name} → {label}")

if __name__ == "__main__":
    main()