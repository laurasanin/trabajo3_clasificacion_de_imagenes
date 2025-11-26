
proyecto-clasificación de Imágenes Médicas con Descriptores Clásicos vs. Deep Learning/
├── README.md                   # Este archivo
├── requirements.txt            # Dependencias del proyecto
├── data/
│   ├── original/              # Imágenes originales del comedor
│   └── synthetic/             # Imágenes sintéticas para validación
├── src/
│   ├── feature_detection.py   # Detectores de características (SIFT, ORB, AKAZE)
│   ├── matching.py            # Emparejamiento robusto de características
│   ├── registration.py        # Estimación de homografías y fusión
│   ├── measurement.py         # Calibración y medición
│   └── utils.py               # Funciones auxiliares y visualización
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb    # Análisis exploratorio
│   ├── 02_synthetic_validation.ipynb    # Validación con imágenes sintéticas
│   └── 03_main_pipeline.ipynb           # Pipeline principal de fusión
├── results/
│   ├── figures/               # Gráficas y visualizaciones
│   └── measurements/          # Resultados de mediciones
└── tests/                     # Pruebas unitarias (opcional)


├── config_dataset.py           # Descarga el dataset desde Kaggle
├── preprocessing.py            # Normalización de tamaño, CLAHE y análisis exploratorio
├── texture_descriptors.py      # Implementación de LBP, GLCM, Gabor y stats de intensidad
├── build_feature_dataset.py    # Orquesta todo el pipeline y genera el CSV final
├── features_texture_pneumonia.csv (generado)
├── README.md
└── requirements.txt



pip install kagglehub numpy pandas matplotlib scikit-image

Con columnas:

- `filepath`
- `label` (NORMAL / PNEUMONIA)
- `subset` (train / val / test)
- `f_0`, `f_1`, ..., `f_n` (características numéricas)

Este archivo lo consumirán los roles encargados del modelado.

---

## 

Cómo Ejecutar el Proyecto

### 1. Instalar dependencias
Crear entorno virtual (opcional):

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt