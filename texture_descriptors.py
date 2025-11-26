"""
texture_descriptors.py

Implementación de descriptores de textura clásicos para radiografías:

- LBP (Local Binary Patterns)
- GLCM (Gray Level Co-occurrence Matrix)
- Filtros de Gabor
- (Opcional) Estadísticas de primer orden

Requiere: pip install numpy scikit-image
"""

from typing import Dict

import numpy as np
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor


# Parámetros LBP
LBP_P = 8               # número de vecinos
LBP_R = 1               # radio
LBP_METHOD = "uniform"  # recomendado para texturas

# Parámetros GLCM
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS = 256       # cuantización (0-255)

# Parámetros Gabor
GABOR_THETAS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GABOR_FREQUENCIES = [0.1, 0.2, 0.3]


# ---------------------------
# Descriptores individuales
# ---------------------------

def extract_lbp_features(img: np.ndarray) -> np.ndarray:
    """
    Calcula LBP y devuelve el histograma normalizado de patrones como vector.

    img : np.ndarray
        Imagen en escala de grises [0, 1].
    """
    lbp = local_binary_pattern(img, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype("float32")


def extract_glcm_features(img: np.ndarray) -> np.ndarray:
    """
    Calcula matriz de co-ocurrencia (GLCM) y extrae propiedades:

    - contraste
    - disimilaridad
    - homogeneidad
    - energía
    - correlación
    - ASM

    Se promedia sobre distancias y ángulos.
    """
    # Cuantizar a niveles 0..GLCM_LEVELS-1
    img_scaled = (img * (GLCM_LEVELS - 1)).astype("uint8")

    glcm = greycomatrix(
        img_scaled,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = []
    for prop in properties:
        vals = greycoprops(glcm, prop)  # shape (len(distances), len(angles))
        feats.append(vals.mean())

    return np.array(feats, dtype="float32")


def extract_gabor_features(img: np.ndarray) -> np.ndarray:
    """
    Calcula un banco de filtros de Gabor con diferentes orientaciones y frecuencias.

    Para cada combinación (theta, frequency) se extraen:
    - media de la magnitud de la respuesta
    - desviación estándar de la magnitud

    Devuelve vector concatenado.
    """
    feats = []

    for theta in GABOR_THETAS:
        for freq in GABOR_FREQUENCIES:
            real, imag = gabor(img, frequency=freq, theta=theta)
            magnitude = np.sqrt(real ** 2 + imag ** 2)
            feats.append(magnitude.mean())
            feats.append(magnitude.std())

    return np.array(feats, dtype="float32")


def extract_first_order_stats(img: np.ndarray) -> np.ndarray:
    """
    Estadísticas de primer orden sobre la imagen (intensidad):

    - media
    - varianza
    - skewness
    - kurtosis
    - entropía

    (Este descriptor es opcional en el trabajo, pero útil y barato de calcular).
    """
    x = img.ravel().astype("float32")
    mean = np.mean(x)
    var = np.var(x) + 1e-8  # evitar división por cero
    std = np.sqrt(var)

    # skewness y kurtosis "a mano" para no depender de scipy
    skew = np.mean(((x - mean) / std) ** 3)
    kurt = np.mean(((x - mean) / std) ** 4)

    # histograma para entropía
    hist, _ = np.histogram(x, bins=256, range=(0.0, 1.0), density=True)
    hist = hist + 1e-12
    entropy = -np.sum(hist * np.log2(hist))

    return np.array([mean, var, skew, kurt, entropy], dtype="float32")


# ---------------------------
# Vector de características completo
# ---------------------------

def extract_all_texture_features(
    img: np.ndarray,
    include_first_order: bool = True
) -> Dict[str, np.ndarray]:
    """
    Calcula y devuelve todos los vectores de textura definidos:

    Retorna un dict con:
      - 'lbp'   : histograma LBP
      - 'glcm'  : características GLCM
      - 'gabor' : características Gabor
      - 'f1'    : estadísticas de primer orden (opcional)

    """
    feats_lbp = extract_lbp_features(img)
    feats_glcm = extract_glcm_features(img)
    feats_gabor = extract_gabor_features(img)

    features = {
        "lbp": feats_lbp,
        "glcm": feats_glcm,
        "gabor": feats_gabor,
    }

    if include_first_order:
        features["f1"] = extract_first_order_stats(img)

    return features


def concatenate_features(feat_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Toma el diccionario de vectores de textura y los concatena en un solo vector 1D.

    El orden por defecto es: LBP, GLCM, Gabor, F1 (si está).
    """
    order = ["lbp", "glcm", "gabor", "f1"]
    vecs = []
    for key in order:
        if key in feat_dict:
            vecs.append(feat_dict[key].ravel())
    if not vecs:
        raise ValueError("No hay vectores de características en feat_dict.")
    return np.concatenate(vecs).astype("float32")


if __name__ == "__main__":
    print("Este módulo define descriptores de textura (LBP, GLCM, Gabor, F1).")