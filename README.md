# Setup rapide Python (venv + requirements)

## 1. Créer un environnement virtuel
```bash
python3.12 -m venv env
```

## 2. Activer l'environnement
Windows
```bash
env\Scripts\activate
```

## 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 4. Structure de projet recommandée
```bash
│
├── env/                  # Environnement virtuel
├── src/                  # Code source
│   └── main.py
├── requirements.txt      # Dépendances
├── README.md             # Documentation
└── .gitignore
```

## 5. Désactiver l'environnement
```bash
deactivate
```
