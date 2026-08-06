"""
Fase 1: auto-etiquetado con Qwen VL. data/raw/ -> data/labels_auto/ (+ data/labels_check/)

Por cada imagen en data/raw/: le pide a Qwen las cajas (PROMPT_LABELING), las
convierte a formato YOLO y escribe un .txt en data/labels_auto/, y dibuja las
cajas sobre la imagen para revisión manual en data/labels_check/.

Uso:
    python src/auto_labeling.py
"""

from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import QwenVLM, convert_to_yolo, parse_boxes

# Rutas de entrada/salida: por defecto las reales de Fase 1 (data/raw/ ->
# labels_auto/ + labels_check/). Para generar pseudo-ground-truth sobre las
# 40 imágenes de validación (data/validation/images/ -> data/validation/labels/
# + labels_check/, un holdout que el entrenamiento nunca vio -- ver nota en
# CLAUDE.md), comentar el bloque de abajo y descomentar el de validación. No
# toca las rutas reales de Fase 1 en ningún caso.
#INPUT_DIR = config.RAW_DIR
#LABELS_AUTO_OUT = config.LABELS_AUTO_DIR
#LABELS_CHECK_OUT = config.LABELS_CHECK_DIR

INPUT_DIR = config.VALIDATION_IMAGES_DIR
LABELS_AUTO_OUT = config.VALIDATION_LABELS_DIR
LABELS_CHECK_OUT = config.VALIDATION_LABELS_CHECK_DIR


def es_imagen(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in config.IMG_EXTENSIONS


def listar_imagenes() -> list[Path]:
    imagenes = []
    for p in INPUT_DIR.iterdir():
        if es_imagen(p):
            imagenes.append(p)
    return sorted(imagenes)


def dibujar_cajas(image: Image.Image, cajas: list[list[float]]) -> Image.Image:
    """Dibuja, sobre una copia de la imagen, las cajas en escala 0-1000 [x1,y1,x2,y2]."""
    ancho, alto = image.size
    salida = image.copy()
    draw = ImageDraw.Draw(salida)
    for x1, y1, x2, y2 in cajas:
        rect = (x1 / 1000 * ancho, y1 / 1000 * alto, x2 / 1000 * ancho, y2 / 1000 * alto)
        draw.rectangle(rect, outline="lime", width=4)
    return salida


def procesar_imagen(vlm: QwenVLM, imagen_path: Path) -> int:
    """Procesa una imagen: pide las cajas a Qwen, escribe el .txt YOLO y guarda la
    visualización con los BB dibujados. Regresa el número de cajas detectadas."""
    image = Image.open(imagen_path).convert("RGB")

    respuesta = vlm.ask(image, config.PROMPT_LABELING)
    cajas = parse_boxes(respuesta)

    lineas_yolo = [convert_to_yolo(caja) for caja in cajas]
    txt_path = LABELS_AUTO_OUT / f"{imagen_path.stem}.txt"
    txt_path.write_text("\n".join(lineas_yolo) + ("\n" if lineas_yolo else ""))

    visualizacion = dibujar_cajas(image, cajas)
    visualizacion.save(LABELS_CHECK_OUT / imagen_path.name)

    return len(cajas)


def main():
    imagenes = listar_imagenes()
    if not imagenes:
        sys.exit(f"No se encontraron imágenes en {INPUT_DIR}")

    if config.AUTO_LABELING_LIMIT is not None:
        imagenes = imagenes[:config.AUTO_LABELING_LIMIT]

    LABELS_AUTO_OUT.mkdir(parents=True, exist_ok=True)
    LABELS_CHECK_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Cargando modelo {config.QWEN_LABELER} en {config.DEVICE} (puede tardar)...")
    
    modelo = config.QWEN_LABELER
    vlm = QwenVLM(modelo).load()

    print(f"Procesando {len(imagenes)} imagen(es)...")

    for i, imagen_path in enumerate(imagenes, start=1):
        n_cajas = procesar_imagen(vlm, imagen_path)
        print(f"[{i}/{len(imagenes)}] {imagen_path.name}: {n_cajas} caja(s)")

    print("\nListo.")


if __name__ == "__main__":
    main()
