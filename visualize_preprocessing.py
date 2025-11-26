"""
visualize_preprocessing.py

Ejemplo visual del pipeline de preprocesamiento usado en el proyecto:
- carga en gris
- resize a tamaño fijo
- CLAHE

Genera una figura comparando:
- imagen original
- imagen redimensionada/normalizada
- imagen tras CLAHE
y muestra los histogramas de intensidad correspondientes.

Este script sirve como evidencia visual para la justificación del preprocesamiento
descrita en el README.
"""

from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform, exposure

from config_dataset import download_and_get_data_dir

# Tamaño objetivo (ajusta si en tu código usas otro)
IMG_SIZE = (256, 256)
OUTPUT_FIG = "examples_preprocessing.png"


def load_random_image_from_class(base_dir: Path, subset: str = "train", cls: str = "NORMAL"):
    """
    Selecciona y carga una imagen aleatoria de la clase indicada.
    base_dir apunta a la carpeta 'chest_xray' que devuelve config_dataset.
    """
    class_dir = base_dir / subset / cls
    files = [f for f in class_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not files:
        raise RuntimeError(f"No se encontraron imágenes en {class_dir}")
    img_path = random.choice(files)
    img = io.imread(img_path.as_posix())
    return img, img_path.name


def preprocess_steps(img: np.ndarray):
    """
    Aplica los mismos pasos conceptuales que el pipeline principal:
    - asegurar escala de grises
    - resize a tamaño fijo
    - CLAHE
    """
    # Escala de grises
    if img.ndim == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img.astype(np.float32)
        img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)

    # Resize
    img_resized = transform.resize(img_gray, IMG_SIZE, anti_aliasing=True)

    # CLAHE (ecualización de histograma adaptativa con límite de contraste)
    img_clahe = exposure.equalize_adapthist(img_resized, clip_limit=0.03)

    return img_gray, img_resized, img_clahe


def plot_preprocessing_example():
    # 1. Localizar dataset usando el mismo helper que el pipeline principal
    data_dir = download_and_get_data_dir()  # carpeta 'chest_xray'
    print("Usando DATA_DIR:", data_dir)

    # 2. Cargar una imagen de ejemplo (p.ej. NORMAL/train)
    img, name = load_random_image_from_class(data_dir, subset="train", cls="NORMAL")
    img_gray, img_resized, img_clahe = preprocess_steps(img)

    # 3. Crear figura con imágenes + histogramas
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Fila 1: imágenes
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title(f"Original\n{name}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_resized, cmap="gray")
    axes[0, 1].set_title("Redimensionada/normalizada")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_clahe, cmap="gray")
    axes[0, 2].set_title("Tras CLAHE")
    axes[0, 2].axis("off")

    # Fila 2: histogramas de intensidades
    axes[1, 0].hist(img_gray.ravel(), bins=50)
    axes[1, 0].set_title("Histograma original")

    axes[1, 1].hist(img_resized.ravel(), bins=50)
    axes[1, 1].set_title("Histograma redimensionada")

    axes[1, 2].hist(img_clahe.ravel(), bins=50)
    axes[1, 2].set_title("Histograma tras CLAHE")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150)
    plt.close()
    print(f"Figura de ejemplo guardada en: {OUTPUT_FIG}")


if __name__ == "__main__":
    plot_preprocessing_example()