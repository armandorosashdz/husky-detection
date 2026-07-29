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

# ---------- Imágenes ----------
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMG_PREFIX     = CLASS_NAME   # husky_001.jpg, husky_002.jpg, ...

# Fase 1: auto-etiquetado (modelo grande, calidad sobre velocidad)
# Nota: Qwen3.5 es nativamente multimodal, los repos de HF NO llevan sufijo "-VL"
# (a diferencia de Qwen2.5-VL). Verificado en huggingface.co/Qwen/Qwen3.5-4B.
QWEN_LABELER = "Qwen/Qwen3.5-4B"

# Fase 4: validadores en cascada, se eligen por flag --validator
QWEN_VALIDATORS = {
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "2b":   "Qwen/Qwen3.5-2B",
    "4b":  "Qwen/Qwen3.5-4B"
}

YOLO_BASE    = "yolov8s.pt"          # pesos preentrenados de Ultralytics
YOLO_TRAINED = ROOT / "runs/detect/train/weights/best.pt"

# Sin GPU NVIDIA disponible en esta laptop (solo AMD integrada) -> CPU.
# float16 no está bien soportado para generación en CPU, por eso float32.
# Cambiar a "cuda"/"float16" en una máquina con GPU NVIDIA.
DEVICE = "cpu"
DTYPE  = "float32"

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