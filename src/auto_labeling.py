"""
Fase 1: auto-etiquetado con Qwen VL. data/raw/ -> data/labels_auto/

Por ahora solo una prueba de humo (smoke test) para validar que QwenVLM.load()
y QwenVLM.ask() funcionan de punta a punta con una sola imagen antes de escribir
el loop completo sobre las 100 imágenes.

Uso:
    python src/auto_labeling.py
"""

from pathlib import Path
import sys

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
import config
from vlm_utils import QwenVLM, parse_boxes


def es_imagen(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in config.IMG_EXTENSIONS


def main():

    # Se crea la lista de imágenes a procesar (solo las que tengan extensión válida).
    imagenes = []

    for p in config.RAW_DIR.iterdir():
        if es_imagen(p):
            imagenes.append(p)
    imagenes = sorted(imagenes)

    # Si no hay imágenes, abortar.
    if not imagenes:
        sys.exit(f"No se encontraron imágenes en {config.RAW_DIR}")


    # ---------- TEST ----------
    # Se toma la primera imagen para el test.
    imagen_prueba = imagenes[0]
    print(f"Probando con: {imagen_prueba.name}")

    # Se configura el modelo Qwen a usar (0.8b, 2b o 4b).
    modelo_prueba = config.QWEN_VALIDATORS["0.8b"]
    print(f"Cargando modelo {modelo_prueba} en {config.DEVICE} (puede tardar)...")

    # Se crea el objeto QwenVLM y se carga el modelo en memoria.
    vlm = QwenVLM(modelo_prueba).load()

    # se abre imagen y se corre el modelo con el prompt de detección (Fase 1).
    image = Image.open(imagen_prueba).convert("RGB")
    respuesta = vlm.ask(image, config.PROMPT_LABELING)

    print("\nRespuesta cruda de Qwen:")
    print(respuesta)

    # Se parsea la respuesta para extraer las cajas y se imprimen.
    boxes = parse_boxes(respuesta)
    print(f"\nCajas parseadas ({len(boxes)}):")
    for box in boxes:
        print(f"  {box}")


if __name__ == "__main__":
    main()
