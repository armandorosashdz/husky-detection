"""
Fase 1 (variante Pedro) - Auto-etiquetado de Huskies con verificacion de raza
--------------------------------------------------------------------------
Extiende auto_labeling.py: por cada caja que detecta Qwen, se recorta esa
region de la imagen ORIGINAL y se le pregunta al modelo -en una consulta
binaria aparte- si eso es realmente un husky (y no otra raza / lobo).
Las cajas no confirmadas se descartan antes de escribirse a
data/labels_auto/ y antes de dibujarse en data/labels_check/.

Convencion del proyecto: sin argumentos de CLI. Todo lo que varia entre
corridas viene de config.py; lo que es especifico de esta variante (y aun
no vive en config.py) esta como constante arriba de este archivo.

Uso:
    python src/auto_labeling_pedro.py
"""
from pathlib import Path
import sys

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import QwenVLM, parse_boxes, convert_to_yolo
from auto_labeling import listar_imagenes, dibujar_cajas


# ---------------- Config propia de esta variante ----------------
# TODO: si esto se vuelve permanente, mover ENABLE_BREED_VERIFICATION y
# VERIFY_PROMPT a config.py, junto con PROMPT_LABELING, para no romper la
# convencion de "toda la config vive en config.py".
ENABLE_BREED_VERIFICATION = True

VERIFY_PROMPT = (
    "Look only at this cropped image. Is the dog shown specifically a "
    "Siberian Husky (not any other breed, and not a wolf)? "
    "Answer with exactly one word: Yes or No."
)

SUSPICIOUS_BOX_COUNT = 12
# ------------------------------------------------------------------


def crop_from_box_0_1000(image: Image.Image, box: list[float]) -> Image.Image:
    """Recorta la region de la imagen ORIGINAL a partir de una caja en escala
    0-1000 [x1,y1,x2,y2] (formato nativo de parse_boxes)."""
    ancho, alto = image.size
    x1, y1, x2, y2 = box
    x1_px = max(0, int(x1 / 1000 * ancho))
    y1_px = max(0, int(y1 / 1000 * alto))
    x2_px = min(ancho, int(x2 / 1000 * ancho))
    y2_px = min(alto, int(y2 / 1000 * alto))
    return image.crop((x1_px, y1_px, x2_px, y2_px))


def verify_is_husky(vlm: QwenVLM, crop_image: Image.Image) -> bool:
    """Conservador: ante cualquier duda o error devuelve False -- mejor
    perder una caja valida que meter una mala."""
    if crop_image.width < 10 or crop_image.height < 10:
        return False
    try:
        respuesta = vlm.ask(crop_image, VERIFY_PROMPT)
    except Exception as e:
        print(f"   ! error en verificacion, se descarta la caja por seguridad: {e}")
        return False
    return respuesta.strip().lower().startswith("yes")


def procesar_imagen_pedro(vlm: QwenVLM, imagen_path: Path) -> tuple[int, int]:
    """Como procesar_imagen() de auto_labeling.py, pero filtrando cada caja con
    una segunda consulta binaria de raza antes de escribirla. Regresa
    (cajas escritas, cajas descartadas)."""
    image = Image.open(imagen_path).convert("RGB")
    respuesta = vlm.ask(image, config.PROMPT_LABELING)
    cajas_crudas = parse_boxes(respuesta)

    cajas_confirmadas = []
    descartadas = 0
    for caja in cajas_crudas:
        if ENABLE_BREED_VERIFICATION:
            crop = crop_from_box_0_1000(image, caja)
            if not verify_is_husky(vlm, crop):
                descartadas += 1
                continue
        cajas_confirmadas.append(caja)

    lineas_yolo = [convert_to_yolo(caja, class_id=config.CLASS_ID) for caja in cajas_confirmadas]
    txt_path = config.LABELS_AUTO_DIR / f"{imagen_path.stem}.txt"
    txt_path.write_text("\n".join(lineas_yolo) + ("\n" if lineas_yolo else ""))

    if len(cajas_confirmadas) > SUSPICIOUS_BOX_COUNT:
        print(f"   !! sospechoso: {len(cajas_confirmadas)} cajas es mucho, revisa a mano")

    visualizacion = dibujar_cajas(image, cajas_confirmadas)
    visualizacion.save(config.LABELS_CHECK_DIR / imagen_path.name)

    return len(cajas_confirmadas), descartadas


def main():
    imagenes = listar_imagenes()
    if not imagenes:
        sys.exit(f"No se encontraron imágenes en {config.RAW_DIR}")
    if config.AUTO_LABELING_LIMIT is not None:
        imagenes = imagenes[:config.AUTO_LABELING_LIMIT]

    config.LABELS_AUTO_DIR.mkdir(parents=True, exist_ok=True)
    config.LABELS_CHECK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cargando modelo {config.QWEN_LABELER} en {config.DEVICE} (puede tardar)...")
    vlm = QwenVLM(config.QWEN_LABELER).load()

    print(f"Procesando {len(imagenes)} imagen(es)...")
    total_cajas = 0
    total_descartadas = 0
    for i, imagen_path in enumerate(imagenes, start=1):
        n_cajas, n_descartadas = procesar_imagen_pedro(vlm, imagen_path)
        total_cajas += n_cajas
        total_descartadas += n_descartadas
        print(f"[{i}/{len(imagenes)}] {imagen_path.name}: {n_cajas} caja(s) "
              f"({n_descartadas} descartada(s) por raza)")

    print("\n==== Resumen ====")
    print(f"Total de cajas escritas: {total_cajas}")
    print(f"Total descartadas por el validador de raza: {total_descartadas}")
    print("Listo.")


if __name__ == "__main__":
    main()
