"""Configuración central del pipeline. Todo lo ajustable vive aquí, agrupado
por fase (ver README para el orden de los pasos y CLAUDE.md para el diseño)."""

from pathlib import Path

# ============================================================
# RUTAS
# ============================================================
ROOT = Path(__file__).parent
DATA = ROOT / "data"

# ---------- Fase 1 (rename_and_resize_images.py + auto_labeling.py): datos reales ----------
RAW_DIR          = DATA / "raw"           # imágenes crudas sin anotar, entrada de Fase 1
LABELS_AUTO_DIR  = DATA / "labels_auto"   # .txt YOLO generados por Qwen
LABELS_CHECK_DIR = DATA / "labels_check"  # visualizaciones con BB dibujadas, para QA manual

# ---------- Fase 2 (split_dataset.py -> train_yolo.py): split train/test ----------
TRAIN_DIR = DATA / "train"   # 70% del split, images/+labels/
TEST_DIR  = DATA / "test"    # 30% del split, images/+labels/ -- también usado como val
                              # de Ultralytics durante el entrenamiento (ver README)

DATASET_YAML = ROOT / "dataset.yaml"   # generado por split_dataset.py, lo lee train_yolo.py

# ---------- Fase 3/4/5 (hybrid_inference.py): holdout de validación limpio ----------
# imágenes nunca vistas por el entrenamiento. Mismo layout images/+labels/ que
# TEST_DIR, para que hybrid_inference.py pueda apuntar a cualquiera de los
# dos sin cambiar cómo lee las rutas. labels/ se llena corriendo
# auto_labeling.py apuntado aquí.
VALIDATION_DIR              = DATA / "validation"
VALIDATION_IMAGES_DIR       = VALIDATION_DIR / "images"
VALIDATION_LABELS_DIR       = VALIDATION_DIR / "labels"
VALIDATION_LABELS_CHECK_DIR = VALIDATION_DIR / "labels_check"

# ---------- Fase 5 (metrics.py) / reporte: resultados ----------
RESULTS     = ROOT / "results"
METRICS_DIR = RESULTS / "metrics"   # un JSON por corrida de hybrid_inference.py
FIGURES_DIR = RESULTS / "figures"   # imágenes anotadas por corrida (verde=TP/naranja=FP)
GRAPHS_DIR  = RESULTS / "graphs"    # curvas Precision-Recall por corrida (metrics.py)
MODELS_DIR  = RESULTS / "models"    # no referenciado por ningún script aún -- distinto de
                                     # models/ en la raíz, donde vive YOLO_TRAINED (ver abajo)

# ---------- Fixtures (datos falsos, solo para probar scripts sin arriesgar los reales) ----------
# Generados por src/generate_fixture_dataset.py, usados por split_dataset.py
# cuando su bloque de rutas real está comentado.
RAW_FIXTURE_DIR          = DATA / "raw_fixture"
LABELS_AUTO_FIXTURE_DIR  = DATA / "labels_auto_fixture"
LABELS_CHECK_FIXTURE_DIR = DATA / "labels_check_fixture"
TRAIN_FIXTURE_DIR        = DATA / "train_fixture"
TEST_FIXTURE_DIR         = DATA / "test_fixture"
DATASET_YAML_FIXTURE     = ROOT / "dataset_fixture.yaml"


# ============================================================
# CLASE E IMÁGENES -- usado en Fase 1 (etiquetado) y Fase 2 (dataset.yaml)
# ============================================================
CLASS_ID   = 0
CLASS_NAME = "husky"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMG_PREFIX     = CLASS_NAME   # rename_and_resize_images.py: husky_000.jpg, husky_001.jpg, ...

# rename_and_resize_images.py también redimensiona las imágenes que excedan
# esto (en su lado más grande, manteniendo aspecto, nunca agranda). Algunas
# imágenes de data/raw/ son enormes y hacían que Qwen intentara reservar
# >10GB de golpe en una sola llamada de atención.
MAX_IMAGE_DIM = 1280


# ============================================================
# MODELOS QWEN -- tamaños disponibles (compartido entre fases)
# ============================================================
# Nota: Qwen3.5 es nativamente multimodal, los repos de HF NO llevan sufijo "-VL"
# (a diferencia de Qwen2.5-VL). Verificado en huggingface.co/Qwen/Qwen3.5-4B.
QWEN_MODELS = {
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "2b":   "Qwen/Qwen3.5-2B",
    "4b":   "Qwen/Qwen3.5-4B",
    "9b":   "Qwen/Qwen3.5-9B",
}


# ============================================================
# FASE 1 -- auto_labeling.py (auto-etiquetado con Qwen)
# ============================================================

# Qué tamaño usar como etiquetador. Cambiar aquí a mano según la máquina.
QWEN_LABELER = QWEN_MODELS["0.8b"]
# QWEN_LABELER = QWEN_MODELS["2b"]
# QWEN_LABELER = QWEN_MODELS["4b"]
# QWEN_LABELER = QWEN_MODELS["9b"]  # usar en Colab/kaggle (junto con DEVICE="auto" abajo)

# Límite de imágenes de INPUT_DIR a procesar por corrida. None = todas.
# Cambiar a un entero para probar rápido con pocas imágenes.
AUTO_LABELING_LIMIT = None
# AUTO_LABELING_LIMIT = 5

# Si está activo, cada caja que detecta Qwen se recorta y se le manda una
# segunda consulta binaria aparte (PROMPT_VERIFY_BREED) preguntando si esa
# región es específicamente un Husky Siberiano -- las cajas que no se
# confirman se descartan antes de escribirse. Reutiliza el mismo modelo ya
# cargado como QWEN_LABELER, así que funciona igual con cualquier tamaño
# (0.8b/2b/4b) -- no depende de ninguno en particular.
VERIFY_BREED = True
# VERIFY_BREED = False

PROMPT_VERIFY_BREED = (
    "Look only at this cropped image. Is the dog shown specifically a "
    "Siberian Husky (not any other breed, and not a wolf)? "
    "Answer with exactly one word: Yes or No."
)

# Aviso nada más (no descarta nada): si una imagen termina con más cajas
# confirmadas que esto, probablemente valga la pena revisarla a mano.
SUSPICIOUS_BOX_COUNT = 12

# Prompt corto (primera versión, usado para las pruebas iniciales de Fase 1):
# PROMPT_LABELING = (
#     "Detect all husky dogs in this image. "
#     "Return a JSON list of objects, each with a \"bbox_2d\" key: "
#     "[x1, y1, x2, y2] (top-left and bottom-right corners) in a 0-1000 scale. "
#     "Return only the JSON, no explanation."
# )
#
# Prompt activo: más descriptivo/estricto (pide TODAS las instancias, cajas
# ajustadas sin adivinar partes ocultas, sin fences de markdown, y excluye
# explícitamente otras razas -- complementa a VERIFY_BREED en vez de
# reemplazarlo, ya que Qwen no siempre respeta esta instrucción al 100%).
PROMPT_LABELING = (
    "Locate every single husky dog visible in this image, even if there "
    "are several of them. Do not stop after the first one -- include ALL "
    "husky dogs you can see as separate entries in the array. "
    "IMPORTANT: only include Siberian Husky breed dogs. If the image also "
    "contains dogs of OTHER breeds (e.g. Jack Russell, Labrador, German "
    "Shepherd, etc.) or wolves, do NOT include them, even if they appear "
    "right next to a husky. "
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


# ============================================================
# FASE 2 -- split_dataset.py + train_yolo.py (fine-tuning YOLOv8s)
# ============================================================

TRAIN_RATIO = 0.7   # split_dataset.py: 70 train / 30 test

YOLO_BASE = "yolov8s.pt"   # pesos preentrenados de Ultralytics

# Carpeta de salida "de trabajo" de Ultralytics durante el entrenamiento (se
# regenera cada corrida, gitignored salvo runs/detect/train/ -- ver
# train_yolo.py). Separado de YOLO_TRAINED porque ese vive en models/, que
# no sigue la estructura <project>/<name>/weights/ que arma Ultralytics.
YOLO_RUNS_DIR = ROOT / "runs" / "detect"
YOLO_RUN_NAME = "train"

# Modelo "activo" que usa el resto del pipeline (YOLODetector,
# hybrid_inference.py). train_yolo.py copia aquí el best.pt de la corrida al
# terminar -- así el pipeline completo (main.py) puede correr de punta a
# punta sin pasos manuales. models/ vive en la raíz y sí se comitea
# (excepción en .gitignore) a diferencia de runs/.
#
# YOLO_TRAINED_NAME es lo único que hay que cambiar para que tu propia
# corrida de train_yolo.py se guarde con otro nombre (sin pisar el de otra
# persona) o para usar un modelo ya entrenado por alguien más sin
# reentrenar -- el resto del pipeline sigue automáticamente a YOLO_TRAINED.

MODELS_ROOT = ROOT / "models"
YOLO_TRAINED_NAME = "yolov8_finetuned_armando.pt"
# YOLO_TRAINED_NAME = "yolov8_finetuned_pedro.pt"

YOLO_TRAINED = MODELS_ROOT / YOLO_TRAINED_NAME

# Hiperparámetros de entrenamiento -- ver docs.ultralytics.com/modes/train
EPOCHS   = 100
IMG_SIZE = 640
BATCH    = 8
PATIENCE = 20
SEED     = 42   # también controla el split 70/30 de arriba

# Con optimizer="auto" (default de Ultralytics), cls_loss explotó (3.5 -> 18 -> 36)
# a partir de la época 3 y el entrenamiento nunca se recuperó (70 imágenes, solo
# 9 batches/época, LR elegido automáticamente demasiado agresivo para eso). Se fija
# el optimizador explícito para que LR0 realmente se use (con "auto", Ultralytics
# ignora el LR0 que le pases). FREEZE congela las primeras 10 capas (el backbone,
# capas 0-9 del resumen del modelo) -- transfer learning más estable con pocos datos.
OPTIMIZER = "AdamW"
LR0       = 0.001   # la mitad del que "auto" eligió (0.002) y explotó
FREEZE    = 10

# Augmentación (bajo volumen de datos -> agresiva)
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


# ============================================================
# FASE 3/4 -- hybrid_inference.py (detección híbrida YOLO + cascada Qwen)
# ============================================================

# "yolo_only" corre solo el detector (línea base); "cascade" además valida
# cada caja con QWEN_VALIDATOR. Cambiar aquí a mano y volver a correr para
# cada una de las 3 configuraciones a comparar.
HYBRID_MODE = "yolo_only"
# HYBRID_MODE = "cascade"

# Qué tamaño usar como validador de la cascada. La tarea pide correr y
# comparar los dos, así que cambiar aquí a mano entre corridas.
QWEN_VALIDATOR = QWEN_MODELS["0.8b"]
# QWEN_VALIDATOR = QWEN_MODELS["2b"]

PROMPT_VALIDATION = (
    "Is this a husky dog inside this image crop? Answer only Yes or No."
)

# Yes/No es una palabra -- no vale la pena esperar a que QwenVLM.ask() genere
# hasta su default (512 tokens) en un loop de cientos de recortes.
VALIDATION_MAX_NEW_TOKENS = 8

# Inferencia YOLO
CONF_THRESHOLD = 0.15   # bajo a propósito: más detecciones para que la cascada filtre
IOU_THRESHOLD  = 0.45
CROP_PADDING   = 10     # px extra alrededor de la caja antes de mandarla al VLM

# Nombre opcional para una corrida de hybrid_inference.py. Si no es None,
# REEMPLAZA por completo el run_name auto-generado (yolo_only/cascade_08b/
# cascade_2b) -- útil para nombrar la corrida con los modelos usados en vez
# del nombre genérico. Cambiar aquí a mano entre corridas.

# RUN_LABEL = None
RUN_LABEL = "Test"
# RUN_LABEL = "Yolov8s_Qwen0_8b"  # ejemplo de cómo nombrar la corrida con los modelos usados
# RUN_LABEL = "Yolov8s_Qwen2_0b"  # ejemplo de cómo nombrar la corrida con los modelos usados


# ============================================================
# FASE 5 -- metrics.py (evaluación: mAP, matching contra ground truth)
# ============================================================
MAP_IOU_THRESHOLD   = 0.5   # IoU mínimo para considerar una detección TP vs. ground truth (mAP@0.5)
DEDUP_IOU_THRESHOLD = 0.4   # IoU para descartar cajas redundantes tras la cascada (2da pasada tipo NMS)


# ============================================================
# RUNTIME -- usado por utils.py (QwenVLM.load()) en Fase 1 y Fase 4
# ============================================================

# Cambiar a "cuda"/"float16" en una máquina con GPU NVIDIA.
# De lo contrario usar "cpu"/"float32" (o "auto", que detecta automáticamente si hay GPU).

# DEVICE = "cpu"
# DEVICE = "auto"
DEVICE = "cuda"
# DTYPE = "float32"
DTYPE = "float16"
# DTYPE = "bfloat16"
