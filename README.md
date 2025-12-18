# Deep Learning Project

Ce projet compare trois approches (Scikit-Learn, PyTorch et TensorFlow) pour prédire des indicateurs de santé (taux de cholestérol, calories consommées et risque de maladie) à partir de données sur le mode de vie.

## Structure du Projet
```
DeepLearning-Project/
├── data/
│   ├── health_lifestyle_dataset.csv          # Dataset original
│   └── health_lifestyle_dataset_cleaned.csv  # Dataset nettoyé et standardisé
├── src/
│   ├── EDA.py              # Prétraitement et nettoyage des données
│   ├── metrics.py          # Fonctions de calcul des scores (R2, MAE, Accuracy)
│   ├── scikit/
│   │   └── main.py         # Implémentation avec Scikit-Learn
│   ├── torch/              
│   │   ├── torch_model.py  
│   │   ├── torch_wrapper.py
│   │   └── main.py         # Implémentation avec PyTorch
│   └── tf/                 
│       ├── tf_wrapper.py   
│       └── main.py         # Implémentation avec TensorFlow
├── pyproject.toml          # Configuration du projet et dépendances pour uv
├── requirements.txt        # Liste des dépendances Python pour pip
└── README.md
```

## Installation
### Option 1 : Via Pip
Créer un environnement virtuel : 
```
python -m venv .venv
```

Installer les dépendances :
```
pip install -r requirements.txt
```

### Option 2 : Via uv
```
uv sync
```
## Utilisation

### Préparer les données (génère le fichier nettoyé) :
```
python src/EDA.py
```
### Exécuter les modèles :

Scikit-Learn (par Ronyl):
```
python src/scikit/main.py
```

PyTorch (par Corentin):
```
python src/torch/main.py
```

TensorFlow (par Lubin):
```
python src/tf/main.py
```
