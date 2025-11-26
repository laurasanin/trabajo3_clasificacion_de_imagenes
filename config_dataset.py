"""
config_dataset.py

Descarga y localiza el dataset de rayos X de neumonía desde Kaggle.

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Requiere: pip install kagglehub
"""

from pathlib import Path
import kagglehub

DATASET_ID = "paultimothymooney/chest-xray-pneumonia"


def download_and_get_data_dir() -> Path:
    """
    Descarga el dataset (si no está descargado) y retorna la ruta a la carpeta 'chest_xray'.

    Returns
    -------
    Path
        Ruta a la carpeta principal del dataset (que contiene train/val/test).
    """
    print(f"Descargando (o reutilizando) dataset de Kaggle: {DATASET_ID}")
    raw_path = kagglehub.dataset_download(DATASET_ID)
    base = Path(raw_path)

    # En Kaggle normalmente viene como 'chest_xray/train', 'chest_xray/test', 'chest_xray/val'
    data_dir = base / "chest_xray"
    if not data_dir.exists():
        # Por si la estructura cambia
        print("Advertencia: no se encontró carpeta 'chest_xray', usando carpeta base.")
        data_dir = base

    print("DATA_DIR:", data_dir.resolve())
    return data_dir


if __name__ == "__main__":
    # Prueba rápida
    download_and_get_data_dir()