"""
preprocessing.py

Funciones de:
- Exploración (conteo de clases, tamaños).
- Carga de imágenes en gris.
- Pipeline de preprocesamiento: resize + CLAHE.
- (Hooks opcionales para segmentación de ROI).

Requiere: pip install numpy matplotlib scikit-image
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform, exposure


# ---------------------------
# Rutas y listado de imágenes
# ---------------------------

def get_image_paths(
    base_dir: Path,
    subset: str = "train",
    extensions=(".jpeg", ".jpg", ".png")
) -> List[Tuple[Path, str, str]]:
    """
    Recorre base_dir/subset y retorna lista de (ruta_imagen, etiqueta, subset).

    Se espera estructura tipo:
      base_dir/
        train/
          NORMAL/*.jpeg
          PNEUMONIA/*.jpeg
        test/
          ...
        val/
          ...
    """
    subset_dir = base_dir / subset
    images_info = []

    if not subset_dir.exists():
        print(f"[get_image_paths] Subset '{subset}' no encontrado en {base_dir}")
        return images_info

    for label_dir in subset_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name  # p.ej. NORMAL o PNEUMONIA
        for ext in extensions:
            for img_path in label_dir.glob(f"*{ext}"):
                images_info.append((img_path, label, subset))

    return images_info


# ---------------------------
# Carga y preprocesamiento
# ---------------------------

def load_image_gray(path_img: Path) -> np.ndarray:
    """
    Carga una imagen y la devuelve en escala de grises normalizada a [0, 1].

    - Si la imagen tiene 3 canales (RGB), se convierte a gris.
    """
    img = io.imread(path_img)

    # Convertir a escala de grises si es necesario
    if img.ndim == 3:
        img = color.rgb2gray(img)

    img = img.astype("float32")
    max_val = np.max(img)
    if max_val > 0:
        img = img / max_val

    return img


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 0.01,
    kernel_size=None
) -> np.ndarray:
    """
    Aplica CLAHE (ecualización de histograma adaptativa) a una imagen [0, 1].

    Parámetros típicos para radiografías, recomendado en el enunciado del trabajo.
    """
    img_eq = exposure.equalize_adapthist(
        img,
        clip_limit=clip_limit,
        kernel_size=kernel_size
    )
    return img_eq


def preprocess_image(
    img: np.ndarray,
    img_size=(256, 256),
    clip_limit: float = 0.01,
    kernel_size=None
) -> np.ndarray:
    """
    Pipeline de preprocesamiento:
    1) Redimensiona a img_size.
    2) Aplica CLAHE.

    Retorna imagen procesada en [0, 1].
    """
    img_resized = transform.resize(img, img_size, anti_aliasing=True)
    img_clahe = apply_clahe(img_resized, clip_limit=clip_limit, kernel_size=kernel_size)
    return img_clahe


# ---------------------------
# Exploración básica
# ---------------------------

def exploratory_analysis(image_list: List[Tuple[Path, str, str]], max_samples: int = 500):
    """
    Análisis exploratorio rápido:
    - Conteo de imágenes por clase.
    - Histograma de tamaños (alto/ancho) de una muestra.

    Parameters
    ----------
    image_list : list
        Lista de (ruta, label, subset).
    max_samples : int
        Número máximo de imágenes para muestreo de tamaños.
    """
    print("\n=== Análisis exploratorio ===")

    if not image_list:
        print("No se encontraron imágenes para explorar.")
        return

    # Conteo por clase
    labels = [label for _, label, _ in image_list]
    unique, counts = np.unique(labels, return_counts=True)
    print("\nNúmero de imágenes por clase:")
    for u, c in zip(unique, counts):
        print(f"{u}: {c}")

    # Distribución de tamaños (muestra)
    sample = image_list[:max_samples]
    sizes = []
    for img_path, _, _ in sample:
        img = io.imread(img_path)
        h, w = img.shape[:2]
        sizes.append((h, w))

    sizes = np.array(sizes)
    heights = sizes[:, 0]
    widths = sizes[:, 1]

    print("\nEjemplos de tamaños (primeros 10):")
    print(sizes[:10])

    plt.figure()
    plt.hist(heights, bins=20, alpha=0.5, label="Alto")
    plt.hist(widths, bins=20, alpha=0.5, label="Ancho")
    plt.xlabel("Pixeles")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de tamaños de imagen (muestra)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------
# (Opcional) Segmentación ROI
# ---------------------------

def segment_roi_placeholder(img: np.ndarray) -> np.ndarray:
    """
    Placeholder para una segmentación de región de interés (ROI).

    Por ahora simplemente retorna la imagen completa. Si quieres experimentar
    más adelante, puedes aplicar umbralización, detección de bordes, etc.
    """
    return img


if __name__ == "__main__":
    print("Este módulo se usa importándolo desde otros scripts (no hace nada solo).")