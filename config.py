"""Configuración central del pipeline. Todo lo ajustable vive aquí."""

from pathlib import Path

# ---------- Rutas ----------
ROOT = Path(__file__).parent
DATA = ROOT / "data"

RAW_DIR         = DATA / "raw"
LABELS_AUTO_DIR = DATA / "labels_auto"
LABELS_CHECK_DIR= DATA / "labels_check"
TRAIN_DIR       = DATA / "train"
TEST_DIR        = DATA / "test"
VALIDATION_DIR        = DATA / "validation"
# 40 imágenes nunca vistas por el entrenamiento (ni siquiera como val de
# Ultralytics durante train_yolo.py, a diferencia de TEST_DIR -- ver nota en
# CLAUDE.md sobre contaminación del holdout). Mismo layout images/+labels/
# que TEST_DIR, para que hybrid_inference.py pueda apuntar a cualquiera de
# los dos sin cambiar cómo lee las rutas. LABELS_VALIDATION_DIR aún no tiene
# contenido real -- se llena corriendo auto_labeling.py apuntado aquí (ver
# su toggle INPUT_DIR/LABELS_AUTO_OUT/LABELS_CHECK_OUT).
VALIDATION_IMAGES_DIR      = VALIDATION_DIR / "images"
VALIDATION_LABELS_DIR      = VALIDATION_DIR / "labels"
VALIDATION_LABELS_CHECK_DIR = VALIDATION_DIR / "labels_check"

RESULTS   = ROOT / "results"
METRICS_DIR = RESULTS / "metrics"
FIGURES_DIR = RESULTS / "figures"
GRAPHS_DIR  = RESULTS / "graphs"   # curvas Precision-Recall por corrida (metrics.py)
MODELS_DIR  = RESULTS / "models"

DATASET_YAML = ROOT / "dataset.yaml"

# Carpetas y dataset.yaml "fixture": datos falsos generados por
# src/generate_fixture_dataset.py, usados por src/generate_fixture_dataset.py y
# src/split_dataset.py para probar el pipeline sin tocar los datos reales.
RAW_FIXTURE_DIR          = DATA / "raw_fixture"
LABELS_AUTO_FIXTURE_DIR  = DATA / "labels_auto_fixture"
LABELS_CHECK_FIXTURE_DIR = DATA / "labels_check_fixture"
TRAIN_FIXTURE_DIR        = DATA / "train_fixture"
TEST_FIXTURE_DIR         = DATA / "test_fixture"
DATASET_YAML_FIXTURE     = ROOT / "dataset_fixture.yaml"

# ---------- Modelos ----------
CLASS_ID   = 0
CLASS_NAME = "husky"

# ---------- Imágenes ----------
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMG_PREFIX     = CLASS_NAME   # husky_001.jpg, husky_002.jpg, ...

# rename_images.py también redimensiona las imágenes que excedan esto (en su
# lado más grande, manteniendo aspecto, nunca agranda). Algunas imágenes de
# data/raw/ son enormes y hacían que Qwen intentara reservar >10GB de golpe en
# una sola llamada de atención. Ver CLAUDE.md.
MAX_IMAGE_DIM = 1280

# Tamaños de Qwen3.5 disponibles (mismos repos para etiquetador y validadores).
# Nota: Qwen3.5 es nativamente multimodal, los repos de HF NO llevan sufijo "-VL"
# (a diferencia de Qwen2.5-VL). Verificado en huggingface.co/Qwen/Qwen3.5-4B.
QWEN_MODELS = {
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "2b":   "Qwen/Qwen3.5-2B",
    "4b":   "Qwen/Qwen3.5-4B",
    "9b":  "Qwen/Qwen3.5-9B",
}

# Fase 1 (auto_labeling.py): qué tamaño usar como etiquetador. Cambiar aquí a mano
# según la máquina: en esta laptop (sin GPU, RAM limitada) solo "0.8b" corre sin
# quedarse sin memoria; la tarea pide 2b o 4b en una máquina con GPU real.
QWEN_LABELER = QWEN_MODELS["0.8b"]
# QWEN_LABELER = QWEN_MODELS["2b"]
# QWEN_LABELER = QWEN_MODELS["4b"]
# QWEN_LABELER = QWEN_MODELS["9b"]  # usar en Colab/GPU (junto con DEVICE="auto" abajo), no en esta laptop

# Fase 4 (hybrid_inference.py): qué tamaño usar como validador de la cascada.
# La tarea pide correr y comparar los dos, así que cambiar aquí a mano entre
# corridas de hybrid_inference.py (una por tamaño).
QWEN_VALIDATOR = QWEN_MODELS["0.8b"]
# QWEN_VALIDATOR = QWEN_MODELS["2b"]

# Fase 3/4 (hybrid_inference.py): "yolo_only" corre solo el detector (línea
# base); "cascade" además valida cada caja con QWEN_VALIDATOR. Cambiar aquí a
# mano y volver a correr para cada una de las 3 configuraciones a comparar.
#HYBRID_MODE = "cascade"
HYBRID_MODE = "yolo_only"

# Yes/No es una palabra -- no vale la pena esperar a que QwenVLM.ask() genere
# hasta su default (512 tokens) en un loop de cientos de recortes.
VALIDATION_MAX_NEW_TOKENS = 8

# Nombre opcional para una corrida de hybrid_inference.py. Si no es None,
# REEMPLAZA por completo el run_name auto-generado (yolo_only/cascade_08b/
# cascade_2b) -- útil para nombrar la corrida con los modelos usados en vez
# del nombre genérico, ej. "armando_yolov8s_qwen08b". No afecta qué modelo se
# usa (eso lo deciden HYBRID_MODE/QWEN_VALIDATOR arriba) ni lo que se guarda
# en el .json (mode/qwen_model se calculan aparte, no se leen del nombre) --
# solo cambia cómo se llaman results/metrics/<esto>.json,
# results/figures/<esto>/ y results/graphs/<esto>_pr_curve.png.
# RUN_LABEL = None
RUN_LABEL = "Yolov8s"
#RUN_LABEL = "Yolov8s_Qwen0_8b"  # ejemplo de cómo nombrar la corrida con los modelos usados
#RUN_LABEL = "Yolov8s_Qwen2_0b"  # ejemplo de cómo nombrar la corrida con los modelos usados

# Fase 1 (auto_labeling.py): límite de imágenes de data/raw/ a procesar por corrida.
# None = todas. Cambiar aquí a un entero para probar rápido con pocas imágenes.

AUTO_LABELING_LIMIT = None
#AUTO_LABELING_LIMIT = 5

YOLO_BASE    = "yolov8s.pt"          # pesos preentrenados de Ultralytics

# Carpeta de salida "de trabajo" de Ultralytics durante el entrenamiento (se
# regenera cada corrida, gitignored -- ver train_yolo.py). Separado de
# YOLO_TRAINED porque ahora ese vive en models/, que no sigue la estructura
# <project>/<name>/weights/ que arma Ultralytics.
YOLO_RUNS_DIR = ROOT / "runs" / "detect"
YOLO_RUN_NAME = "train"

# Modelo "activo" que usa el resto del pipeline (YOLODetector, hybrid_inference.py).
# train_yolo.py copia aquí el best.pt de la corrida al terminar. Vive en models/,
# que sí se comitea (excepción en .gitignore) a diferencia de runs/.
YOLO_TRAINED = ROOT / "models" / "yolov8_finetuned_armando.pt"

# Sin GPU NVIDIA disponible en esta laptop (solo AMD integrada) -> CPU.
# float16 no está bien soportado para generación en CPU, por eso float32.
# Cambiar a "cuda"/"float16" en una máquina con GPU NVIDIA.
DEVICE = "cpu"
#DEVICE = "auto"
#DEVICE = "cuda"
DTYPE  = "float32"


# ---------- Prompts (versionados: evidencia para la pregunta 4) ----------
# Nota: se probó pedir [ymin, xmin, ymax, xmax] explícitamente y Qwen3.5 lo ignoraba,
# respondiendo siempre con su formato nativo "bbox_2d": [x1, y1, x2, y2]. Orden
# confirmado visualmente en data/labels_check/test_orden_*.jpg. Por eso el prompt
# pide directamente ese formato nativo en vez de pelear contra él.
#
# Prompt corto (primera versión, usado para las pruebas iniciales de Fase 1):
# PROMPT_LABELING = (
#     "Detect all husky dogs in this image. "
#     "Return a JSON list of objects, each with a \"bbox_2d\" key: "
#     "[x1, y1, x2, y2] (top-left and bottom-right corners) in a 0-1000 scale. "
#     "Return only the JSON, no explanation."
# )
#
# Prompt activo: más descriptivo/estricto (pide TODAS las instancias, cajas
# ajustadas sin adivinar partes ocultas, sin fences de markdown).
PROMPT_LABELING = (
    "Locate every single husky dog visible in this image, even if there "
    "are several of them. Do not stop after the first one -- include ALL "
    "husky dogs you can see as separate entries in the array. "
    "Output ONLY a JSON array, no explanation, no markdown fences, "
    "no bold/asterisks, in this exact format: "
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "husky dog"}, ...] '
    "where coordinates are normalized to a 0-1000 scale relative to the "
    "image width and height, (x1, y1) is the top-left corner and "
    "(x2, y2) is the bottom-right corner. "
    "Each box must be a TIGHT bounding box around only the VISIBLE part of "
    "each dog's body. If a dog is partially occluded by another dog, an "
    "object, or the edge of the image, draw the box only around the "
    "visible pixels of that dog -- do NOT guess or extend the box to cover "
    "body parts that are hidden or not visible in the image. "
    "If no husky dog is visible, output []."
)

PROMPT_VALIDATION = (
    "Is this a husky dog inside this image crop? Answer only Yes or No."
)

# ---------- Hiperparámetros ----------
# Estos parámetros se pueden obtener de la página oficial de ultralytics.
# https://docs.ultralytics.com/modes/train

# Entrenamiento
EPOCHS     = 100
IMG_SIZE   = 640
BATCH      = 8
PATIENCE   = 20
SEED       = 42          # también controla el split 70/30

# Con optimizer="auto" (default de Ultralytics), cls_loss explotó (3.5 -> 18 -> 36)
# a partir de la época 3 y el entrenamiento nunca se recuperó (70 imágenes, solo
# 9 batches/época, LR elegido automáticamente demasiado agresivo para eso). Se fija
# el optimizador explícito para que LR0 realmente se use (con "auto", Ultralytics
# ignora el LR0 que le pases). FREEZE congela las primeras 10 capas (el backbone,
# capas 0-9 del resumen del modelo) -- transfer learning más estable con pocos datos.
OPTIMIZER  = "AdamW"
LR0        = 0.001       # la mitad del que "auto" eligió (0.002) y explotó
FREEZE     = 10

# Augmentación (bajo volumen de datos → agresiva)
AUGMENT = {
    "mosaic": 1.0,
    "mixup": 0.15,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "degrees": 10.0,
    "scale": 0.5,
}

# Inferencia
CONF_THRESHOLD = 0.15    # bajo a propósito: más detecciones para que la cascada filtre
IOU_THRESHOLD  = 0.45
CROP_PADDING   = 10      # px extra alrededor de la caja antes de mandarla al VLM

# Fase 4/5 (hybrid_inference.py + metrics.py)
MAP_IOU_THRESHOLD  = 0.5   # IoU mínimo para considerar una detección TP vs. ground truth (mAP@0.5)
DEDUP_IOU_THRESHOLD = 0.4  # IoU para descartar cajas redundantes tras la cascada (2da pasada tipo NMS)

# ---------- Split ----------
TRAIN_RATIO = 0.7        # 70 train / 30 test