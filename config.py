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

# ---------- Modelos ----------
CLASS_ID   = 0
CLASS_NAME = "husky"

# Fase 1: auto-etiquetado (modelo grande, calidad sobre velocidad)
QWEN_LABELER = "Qwen/Qwen3.5-4B-VL"

# Fase 4: validadores en cascada, se eligen por flag --validator
QWEN_VALIDATORS = {
    "0.8b": "Qwen/Qwen3.5-0.8B-VL",
    "2b":   "Qwen/Qwen3.5-2B-VL",
}

YOLO_BASE    = "yolov8s.pt"          # pesos preentrenados de Ultralytics
YOLO_TRAINED = ROOT / "runs/detect/train/weights/best.pt"

DEVICE = "cuda"
DTYPE  = "float16"

# ---------- Prompts (versionados: evidencia para la pregunta 4) ----------
PROMPT_LABELING = (
    "Detect all husky dogs in this image. Return one bounding box per dog "
    "as a JSON list of [ymin, xmin, ymax, xmax] in a 0-1000 scale. "
    "Return only the JSON, no explanation."
)

PROMPT_VALIDATION = (
    "Is this a husky dog inside this image crop? Answer only Yes or No."
)

# ---------- Hiperparámetros ----------
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