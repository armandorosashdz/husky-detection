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

RESULTS   = ROOT / "results"
METRICS_DIR = RESULTS / "metrics"
FIGURES_DIR = RESULTS / "figures"

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
# QWEN_LABELER = QWEN_MODELS["0.8b"]
QWEN_LABELER = QWEN_MODELS["2b"]
# QWEN_LABELER = QWEN_MODELS["4b"]
# QWEN_LABELER = QWEN_MODELS["9b"]  # usar en Colab/GPU (junto con DEVICE="cuda" abajo), no en esta laptop

# Fase 4 (hybrid_inference.py): tamaños de validador a comparar en la cascada.
QWEN_VALIDATORS = QWEN_MODELS

# Fase 1 (auto_labeling.py): límite de imágenes de data/raw/ a procesar por corrida.
# None = todas. Cambiar aquí a un entero para probar rápido con pocas imágenes.

AUTO_LABELING_LIMIT = None
#AUTO_LABELING_LIMIT = 5

YOLO_BASE    = "yolov8s.pt"          # pesos preentrenados de Ultralytics
YOLO_TRAINED = ROOT / "runs/detect/train/weights/best.pt"

# Sin GPU NVIDIA disponible en esta laptop (solo AMD integrada) -> CPU.
# float16 no está bien soportado para generación en CPU, por eso float32.
# Cambiar a "cuda"/"float16" en una máquina con GPU NVIDIA.
#DEVICE = "cpu"
#DTYPE  = "float32"
DEVICE = "cuda"
DTYPE  = "float16"


# ---------- Prompts (versionados: evidencia para la pregunta 4) ----------
# Nota: se probó pedir [ymin, xmin, ymax, xmax] explícitamente y Qwen3.5 lo ignoraba,
# respondiendo siempre con su formato nativo "bbox_2d": [x1, y1, x2, y2]. Orden
# confirmado visualmente en data/labels_check/test_orden_*.jpg. Por eso el prompt
# ahora pide directamente ese formato nativo en vez de pelear contra él.
PROMPT_LABELING = (
    "Detect all husky dogs in this image. "
    "Return a JSON list of objects, each with a \"bbox_2d\" key: "
    "[x1, y1, x2, y2] (top-left and bottom-right corners) in a 0-1000 scale. "
    "Return only the JSON, no explanation."
)
"""
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
    "If no husky dog is visible, output [].
)
"""

PROMPT_VALIDATION = (
    "Is this a husky dog inside this image crop? Answer only Yes or No."
)

# ---------- Hiperparámetros ----------
# Aún no se usan, pero se dejan aquí para referencia futura (entrenamiento YOLOv8).

# Entrenamiento
EPOCHS     = 100
IMG_SIZE   = 640
BATCH      = 8
PATIENCE   = 20
SEED       = 42          # también controla el split 70/30

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

# ---------- Split ----------
TRAIN_RATIO = 0.7        # 70 train / 30 test