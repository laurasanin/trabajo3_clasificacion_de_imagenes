"""
build_feature_dataset.py

Orquesta el pipeline de Rol 1:

1. Descarga el dataset (config_dataset.download_and_get_data_dir).
2. Hace análisis exploratorio en 'train' (conteo de clases, tamaños).
3. Recorre train/val/test, aplica:
   - carga en gris
   - (opcional) segmentación ROI
   - preprocesamiento (resize + CLAHE)
   - extracción de descriptores de textura (LBP, GLCM, Gabor, F1)
4. Construye un DataFrame con:
   - filepath
   - label
   - subset
   - f_0, f_1, ..., f_n  (features)
5. Guarda un CSV con los vectores de características.

Requiere: pip install numpy pandas matplotlib scikit-image kagglehub
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config_dataset import download_and_get_data_dir
from preprocessing import (
    get_image_paths,
    load_image_gray,
    preprocess_image,
    segment_roi_placeholder,
    exploratory_analysis,
)
from texture_descriptors import extract_all_texture_features, concatenate_features


def process_subset(base_dir: Path, subset: str = "train") -> pd.DataFrame:
    """
    Procesa un subset (train/val/test):

    - Obtiene lista de imágenes.
    - Aplica ROI (placeholder), preprocesamiento y extracción de descriptores.
    - Retorna DataFrame con filepath, label, subset y columnas de features.
    """
    image_list = get_image_paths(base_dir, subset=subset)

    if not image_list:
        print(f"[process_subset] No se encontraron imágenes en subset '{subset}'.")
        return pd.DataFrame()

    rows = []
    for idx, (img_path, label, subset_name) in enumerate(image_list, start=1):
        try:
            img = load_image_gray(img_path)
            img_roi = segment_roi_placeholder(img)
            img_prep = preprocess_image(img_roi)

            feat_dict = extract_all_texture_features(img_prep, include_first_order=True)
            feat_vec = concatenate_features(feat_dict)

            row = {
                "filepath": str(img_path),
                "label": label,
                "subset": subset_name,
            }
            # Expandir feat_vec en f_0, f_1, ..., f_n
            for j, val in enumerate(feat_vec):
                row[f"f_{j}"] = float(val)

            rows.append(row)

            if idx % 100 == 0:
                print(f"[{subset}] Procesadas {idx} imágenes...")

        except Exception as exc:
            print(f"Error procesando imagen {img_path}: {exc}")

    df = pd.DataFrame(rows)
    print(f"[process_subset] Subset '{subset}' -> {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df


def main():
    # 1. Descargar dataset y obtener carpeta base (con train/val/test)
    data_dir = download_and_get_data_dir()

    # 2. Análisis exploratorio (solo en train)
    train_list = get_image_paths(data_dir, subset="train")
    print(f"Total imágenes en 'train': {len(train_list)}")
    if train_list:
        exploratory_analysis(train_list, max_samples=500)

    # 3. Procesar subsets
    dfs = []
    for subset in ["train", "val", "test"]:
        subset_dir = data_dir / subset
        if subset_dir.exists():
            print(f"\n=== Procesando subset: {subset} ===")
            df_subset = process_subset(data_dir, subset=subset)
            if not df_subset.empty:
                dfs.append(df_subset)
        else:
            print(f"[main] Subset '{subset}' no existe en {data_dir}, se omite.")

    if not dfs:
        print("No se generaron DataFrames. Revisa la estructura del dataset.")
        return

    # 4. Unir y guardar
    df_all = pd.concat(dfs, ignore_index=True)
    print("\nShape final del dataset de características:", df_all.shape)

    out_path = Path("features_texture_pneumonia.csv")
    df_all.to_csv(out_path, index=False)
    print(f"Archivo guardado en: {out_path.resolve()}")


if __name__ == "__main__":
    main()