"""
Fase 1: auto-etiquetado con Qwen VL, con verificación de raza opcional por
caja (config.VERIFY_BREED).

Por cada imagen en INPUT_DIR: le pide a Qwen las cajas (config.PROMPT_LABELING).
Si config.VERIFY_BREED está activo, cada caja se recorta y se le manda a
Qwen una segunda consulta binaria (config.PROMPT_VERIFY_BREED) preguntando
si es un Husky Siberiano -- las que no se confirman se descartan.

Uso:
    python src/auto_labeling.py
"""

from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import QwenVLM, convert_to_yolo, parse_boxes

# Por defecto las rutas reales de Fase 1. Para etiquetar data/validation/
# en vez de data/raw/, comentar este bloque y descomentar el de abajo.
INPUT_DIR = config.RAW_DIR
LABELS_AUTO_OUT = config.LABELS_AUTO_DIR
LABELS_CHECK_OUT = config.LABELS_CHECK_DIR

#INPUT_DIR = config.VALIDATION_IMAGES_DIR
#LABELS_AUTO_OUT = config.VALIDATION_LABELS_DIR
#LABELS_CHECK_OUT = config.VALIDATION_LABELS_CHECK_DIR


def es_imagen(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in config.IMG_EXTENSIONS


def listar_imagenes() -> list[Path]:
    imagenes = []
    for p in INPUT_DIR.iterdir():
        if es_imagen(p):
            imagenes.append(p)
    return sorted(imagenes)


def dibujar_cajas(image: Image.Image, cajas: list[list[float]]) -> Image.Image:
    """Dibuja cajas en escala 0-1000 [x1,y1,x2,y2] sobre una copia de la imagen."""
    ancho, alto = image.size
    salida = image.copy()
    draw = ImageDraw.Draw(salida)
    for x1, y1, x2, y2 in cajas:
        rect = (x1 / 1000 * ancho, y1 / 1000 * alto, x2 / 1000 * ancho, y2 / 1000 * alto)
        draw.rectangle(rect, outline="lime", width=4)
    return salida


def crop_from_box_0_1000(image: Image.Image, box: list[float]) -> Image.Image:
    """Recorta la imagen original a partir de una caja en escala 0-1000."""
    ancho, alto = image.size
    x1, y1, x2, y2 = box
    x1_px = max(0, int(x1 / 1000 * ancho))
    y1_px = max(0, int(y1 / 1000 * alto))
    x2_px = min(ancho, int(x2 / 1000 * ancho))
    y2_px = min(alto, int(y2 / 1000 * alto))
    return image.crop((x1_px, y1_px, x2_px, y2_px))


def verify_is_husky(vlm: QwenVLM, crop_image: Image.Image) -> bool:
    """Segunda consulta a Qwen sobre el recorte de una caja. Ante cualquier
    duda o error devuelve False -- mejor perder una caja válida que meter
    una mala."""
    if crop_image.width < 10 or crop_image.height < 10:
        return False
    try:
        respuesta = vlm.ask(crop_image, config.PROMPT_VERIFY_BREED, max_new_tokens=8)
    except Exception as e:
        print(f"   ! error en verificación, se descarta la caja por seguridad: {e}")
        return False
    return respuesta.strip().lower().startswith("yes")


def procesar_imagen(vlm: QwenVLM, imagen_path: Path) -> tuple[int, int]:
    """Detecta cajas, opcionalmente las filtra con verify_is_husky(), escribe
    el .txt YOLO y guarda la visualización. Regresa (cajas escritas, descartadas)."""
    image = Image.open(imagen_path).convert("RGB")

    respuesta = vlm.ask(image, config.PROMPT_LABELING)
    cajas_crudas = parse_boxes(respuesta)

    cajas = []
    descartadas = 0
    for caja in cajas_crudas:
        if config.VERIFY_BREED:
            crop = crop_from_box_0_1000(image, caja)
            if not verify_is_husky(vlm, crop):
                descartadas += 1
                continue
        cajas.append(caja)

    lineas_yolo = [convert_to_yolo(caja) for caja in cajas]
    txt_path = LABELS_AUTO_OUT / f"{imagen_path.stem}.txt"
    txt_path.write_text("\n".join(lineas_yolo) + ("\n" if lineas_yolo else ""))

    if len(cajas) > config.SUSPICIOUS_BOX_COUNT:
        print(f"   !! sospechoso: {len(cajas)} cajas es mucho, revisa a mano")

    visualizacion = dibujar_cajas(image, cajas)
    visualizacion.save(LABELS_CHECK_OUT / imagen_path.name)

    return len(cajas), descartadas


def main():
    imagenes = listar_imagenes()
    if not imagenes:
        sys.exit(f"No se encontraron imágenes en {INPUT_DIR}")

    if config.AUTO_LABELING_LIMIT is not None:
        imagenes = imagenes[:config.AUTO_LABELING_LIMIT]

    LABELS_AUTO_OUT.mkdir(parents=True, exist_ok=True)
    LABELS_CHECK_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Cargando modelo {config.QWEN_LABELER} en {config.DEVICE} (puede tardar)...")
    vlm = QwenVLM(config.QWEN_LABELER).load()

    print(f"Procesando {len(imagenes)} imagen(es) (VERIFY_BREED={config.VERIFY_BREED})...")

    total_cajas = 0
    total_descartadas = 0
    for i, imagen_path in enumerate(imagenes, start=1):
        n_cajas, n_descartadas = procesar_imagen(vlm, imagen_path)
        total_cajas += n_cajas
        total_descartadas += n_descartadas
        sufijo = f" ({n_descartadas} descartada(s) por raza)" if config.VERIFY_BREED else ""
        print(f"[{i}/{len(imagenes)}] {imagen_path.name}: {n_cajas} caja(s){sufijo}")

    print("\n==== Resumen ====")
    print(f"Total de cajas escritas: {total_cajas}")
    if config.VERIFY_BREED:
        print(f"Total descartadas por el verificador de raza: {total_descartadas}")
    print("Listo.")


if __name__ == "__main__":
    main()
