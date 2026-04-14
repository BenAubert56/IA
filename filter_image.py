import os
import random
import shutil
from pathlib import Path

# Dossier source (racine)
SOURCE_DIR = Path("images_kiruna/allsky-sweden-kiruna-2024-2")

# Dossier destination
DEST_DIR = Path("data/image_test")

# Nombre d'images à garder
NB_IMAGES = 500

# Extensions d'images autorisées
EXTENSIONS = [".jpg", ".jpeg", ".png"]

def get_all_images(source_dir):
    images = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if Path(file).suffix.lower() in EXTENSIONS:
                images.append(Path(root) / file)
    return images

def main():
    # Créer le dossier destination si besoin
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print("Recherche des images...")
    all_images = get_all_images(SOURCE_DIR)
    print(f"➡️ {len(all_images)} images trouvées")

    if len(all_images) == 0:
        print("Aucune image trouvée")
        return

    # Sélection aléatoire
    selected_images = random.sample(all_images, min(NB_IMAGES, len(all_images)))

    print(f"{len(selected_images)} images sélectionnées")

    # Déplacement
    for img_path in selected_images:
        dest_path = DEST_DIR / img_path.name

        # Éviter les conflits de noms
        counter = 1
        while dest_path.exists():
            dest_path = DEST_DIR / f"{img_path.stem}_{counter}{img_path.suffix}"
            counter += 1

        shutil.move(str(img_path), str(dest_path))

    print("Terminé !")

if __name__ == "__main__":
    main()