
# Clasificación de imágenes médicas (radiografías de tórax): descriptores clásicos de textura vs. Deep Learning

Este proyecto compara dos enfoques para la clasificación de neumonía en radiografías de tórax:
	•	Baseline clásico: descriptores de textura (LBP, GLCM, Gabor, estadísticas de intensidad).
	•	Deep Learning: preparación del dataset para modelos más avanzados.

## Estructura del Proyecto

proyecto-clasificacion-imagenes-medicas/
│
├── config_dataset.py           # Descarga automática del dataset desde Kaggle (kagglehub)
├── preprocessing.py            # Normalización, CLAHE y análisis exploratorio
├── texture_descriptors.py      # LBP, GLCM, Gabor y estadísticas de intensidad
├── build_feature_dataset.py    # Ejecuta todo el pipeline y genera el CSV final
│
├── features_texture_pneumonia.csv   # Dataset final de características
├── README.md
└── requirements.txt

## Cómo ejecutar el proyecto

1. Crear entorno virtual 

python3 -m venv venv
source venv/bin/activate

2. Instalar dependencias

pip install -r requirements.txt

 3. Descargar el Dataset

El proyecto usa el dataset Chest X-Ray Pneumonia de Kaggle.

Para su descarga se debe ejecuta:

python config_dataset.py

Esto descargará automáticamente las carpetas:

chest_xray/
├── train/
├── val/
└── test/

4. Preprocesamiento de imágenes

Incluye:
	•	Normalización de tamaño
	•	Conversión a escala de grises
	•	CLAHE para mejorar contraste
	•	Histograma y estadísticas por clase
	•	Visualizaciones exploratorias

Ejecuta:

python preprocessing.py

5. Extracción de descriptores clásicos de textura

El archivo texture_descriptors.py implementa:
	•	LBP (Local Binary Patterns)
	•	GLCM (Gray Level Co-occurrence Matrix)
	•	Filtros de Gabor
	•	Estadísticas de intensidad (mean, std, skew, kurtosis…)

     6. Construir el Dataset final de features

     python build_feature_dataset.py

Esto generará:

     features_texture_pneumonia.csv

     con columnas como:
	•	lbp_hist_*
	•	glcm_contrast, glcm_homogeneity, etc.
	•	gabor_energy_*
	•	intensity_mean, intensity_std, …

   ##  Decisiones de preprocesamiento: qué se hace y por qué

El pipeline de preprocesamiento está implementado principalmente en `preprocessing.py` y se orquesta desde
`build_feature_dataset.py`. Antes de calcular los descriptores de textura (definidos en
`texture_descriptors.py`), todas las imágenes pasan por la misma secuencia de transformaciones para
garantizar comparabilidad entre ejemplos y resaltar estructuras relevantes.

Las decisiones de preprocesamiento son:

1. **Exploración inicial del dataset (build_feature_dataset.py + preprocessing.py)**  
   Antes de extraer características, se realiza un análisis exploratorio sobre el subconjunto `train`:
   - Conteo de imágenes por clase (NORMAL vs PNEUMONIA).
   - Distribución de tamaños de imagen.  
   Esto permite detectar posibles desequilibrios de clases y variaciones fuertes en la resolución de las
   radiografías que puedan afectar al cálculo de los descriptores.

2. **Carga de imágenes en escala de grises (`preprocessing.py`)**  
   Aunque las radiografías de tórax suelen estar ya en un solo canal, se fuerza una representación
   consistente en escala de grises. Esto simplifica el cálculo de descriptores de textura clásicos
   (LBP, GLCM, Gabor), que están definidos sobre niveles de intensidad, y evita diferencias debidas a
   formatos de archivo con varios canales.

3. **Redimensionamiento a tamaño fijo + normalización (`preprocessing.py`)**  
   Las imágenes originales tienen resoluciones variables. En el pipeline se aplica un *resize* a un
   tamaño fijo (por ejemplo, 256×256 píxeles), con normalización de intensidades:
   - Hace que todas las imágenes sean comparables entre sí.
   - Permite construir vectores de características de dimensión fija.
   - Facilita la futura integración con modelos de *deep learning*.  

   La normalización de intensidades (reescala de valores) ayuda a que los descriptores (estadísticos y
   de textura) no estén dominados por diferencias de brillo global entre imágenes obtenidas con equipos
   distintos.

4. **CLAHE – Contrast Limited Adaptive Histogram Equalization (`preprocessing.py`)**  
   Sobre la imagen redimensionada se aplica **CLAHE**, una ecualización de histograma adaptativa con
   límite de contraste. En radiografías médicas esto es especialmente útil porque:
   - Mejora el contraste local en el parénquima pulmonar.
   - Permite resaltar opacidades e infiltrados sutiles que pueden estar asociados a neumonía.
   - Reduce el riesgo de amplificar ruido en exceso frente a una ecualización global estándar, gracias
     al parámetro de límite de contraste (*clip limit*).

   En el repositorio se incluye un script de visualización (`visualize_preprocessing.py`) que muestra
   ejemplos de imágenes **antes y después** de aplicar CLAHE, junto con sus histogramas de intensidad.

5. ** Segmentación de región de interés – ROI (`preprocessing.py`)**  
   El módulo define un *placeholder* (`segment_roi_placeholder`) para una posible segmentación de región
   de interés (ROI). Actualmente, la ROI coincide con la imagen completa, pero la estructura del código
   permite, en trabajos futuros, introducir recortes más específicos (por ejemplo, focalizados en la
   región pulmonar) sin romper el pipeline de extracción de características.

6. **Integración con la extracción de descriptores (`build_feature_dataset.py` + `texture_descriptors.py`)**  
   Una vez preprocesadas (gris → resize → CLAHE → ROI), las imágenes pasan al módulo
   `texture_descriptors.py`, donde se calculan descriptores clásicos:
   - **LBP (Local Binary Patterns)**  
   - **GLCM (Gray Level Co-occurrence Matrix)**  
   - **Filtros de Gabor**  
   - **Estadísticos de primer orden (F1), opcionales**  

   `build_feature_dataset.py` recorre `train/`, `val/` y `test/`, aplica el mismo preprocesamiento a
   todas las imágenes y construye un `features_texture_pneumonia.csv` con:
   - `filepath`
   - `label` (NORMAL / PNEUMONIA)
   - `subset` (train / val / test)
   - todas las columnas de características de textura concatenadas.

   Las figuras de ejemplo generadas por `visualize_preprocessing.py` (ver `examples_preprocessing.png`)
muestran visualmente el efecto de cada etapa del pipeline (gris → resize → CLAHE) y apoyan la
justificación de las decisiones de preprocesamiento descritas en esta sección.
