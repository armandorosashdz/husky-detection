"""
Fase 1 (versión de prueba) - auto-etiquetado con Qwen VL + verificación de
raza opcional por caja.

Combina auto_labeling.py con la idea de auto_labeling_pedro.py: además de
pedirle a Qwen las cajas de la imagen completa (PROMPT_LABELING), si
VERIFY_BREED está activo, cada caja se recorta y se le manda a Qwen una
segunda consulta binaria aparte (VERIFY_PROMPT) preguntando si esa región
es específicamente un Husky Siberiano -- las cajas que no se confirman se
descartan antes de escribirse. La verificación reutiliza el mismo modelo
ya cargado como QWEN_LABELER, así que funciona igual con "0.8b", "2b" o
"4b" -- no depende de ningún tamaño en particular.

Es un script de prueba (por eso el nombre): mientras se decide si esto
reemplaza a auto_labeling.py, corre por separado y no toca sus rutas de
salida a menos que se le indique lo mismo en INPUT_DIR/LABELS_AUTO_OUT/
LABELS_CHECK_OUT de abajo.

Uso:
    python src/auto_labeling_test.py
"""

from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import QwenVLM, convert_to_yolo, parse_boxes

# Rutas de entrada/salida: mismo toggle que auto_labeling.py -- por defecto
# las reales de Fase 1 (data/raw/ -> labels_auto/ + labels_check/). Para
# probar sobre las 40 imágenes de validación, comentar el bloque de abajo y
# descomentar el de validación. No toca las rutas reales de Fase 1 en
# ningún caso.
INPUT_DIR = config.RAW_DIR
LABELS_AUTO_OUT = config.LABELS_AUTO_DIR
LABELS_CHECK_OUT = config.LABELS_CHECK_DIR

#INPUT_DIR = config.VALIDATION_IMAGES_DIR
#LABELS_AUTO_OUT = config.VALIDATION_LABELS_DIR
#LABELS_CHECK_OUT = config.VALIDATION_LABELS_CHECK_DIR

# ---------------- Verificación de raza (de auto_labeling_pedro.py) ----------------
# TODO: si esto se vuelve permanente, mover VERIFY_BREED/VERIFY_PROMPT a
# config.py, junto con PROMPT_LABELING/PROMPT_VALIDATION, para no romper la
# convención de "toda la config vive en config.py".
VERIFY_BREED = False
# VERIFY_BREED = True

VERIFY_PROMPT = (
    "Look only at this cropped image. Is the dog shown specifically a "
    "Siberian Husky (not any other breed, and not a wolf)? "
    "Answer with exactly one word: Yes or No."
)

# Aviso nada más (no descarta nada): si una imagen termina con más cajas
# confirmadas que esto, probablemente valga la pena revisarla a mano.
SUSPICIOUS_BOX_COUNT = 12
# ------------------------------------------------------------------------------


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


def crop_from_box_0_1000(image: Image.Image, box: list[float]) -> Image.Image:
    """Recorta la región de la imagen ORIGINAL a partir de una caja en escala
    0-1000 [x1,y1,x2,y2] (formato nativo de parse_boxes)."""
    ancho, alto = image.size
    x1, y1, x2, y2 = box
    x1_px = max(0, int(x1 / 1000 * ancho))
    y1_px = max(0, int(y1 / 1000 * alto))
    x2_px = min(ancho, int(x2 / 1000 * ancho))
    y2_px = min(alto, int(y2 / 1000 * alto))
    return image.crop((x1_px, y1_px, x2_px, y2_px))


def verify_is_husky(vlm: QwenVLM, crop_image: Image.Image) -> bool:
    """Segunda consulta a Qwen (mismo modelo ya cargado como QWEN_LABELER,
    sin importar su tamaño) sobre solo el recorte de una caja. Conservador:
    ante cualquier duda o error devuelve False -- mejor perder una caja
    válida que meter una mala a las etiquetas."""
    if crop_image.width < 10 or crop_image.height < 10:
        return False
    try:
        respuesta = vlm.ask(crop_image, VERIFY_PROMPT)
    except Exception as e:
        print(f"   ! error en verificación, se descarta la caja por seguridad: {e}")
        return False
    return respuesta.strip().lower().startswith("yes")


def procesar_imagen(vlm: QwenVLM, imagen_path: Path) -> tuple[int, int]:
    """Procesa una imagen: pide las cajas a Qwen, opcionalmente las filtra
    con verify_is_husky() (si VERIFY_BREED), escribe el .txt YOLO y guarda
    la visualización con los BB dibujados. Regresa (cajas escritas,
    cajas descartadas por el verificador de raza)."""
    image = Image.open(imagen_path).convert("RGB")

    respuesta = vlm.ask(image, config.PROMPT_LABELING)
    cajas_crudas = parse_boxes(respuesta)

    cajas = []
    descartadas = 0
    for caja in cajas_crudas:
        if VERIFY_BREED:
            crop = crop_from_box_0_1000(image, caja)
            if not verify_is_husky(vlm, crop):
                descartadas += 1
                continue
        cajas.append(caja)

    lineas_yolo = [convert_to_yolo(caja) for caja in cajas]
    txt_path = LABELS_AUTO_OUT / f"{imagen_path.stem}.txt"
    txt_path.write_text("\n".join(lineas_yolo) + ("\n" if lineas_yolo else ""))

    if len(cajas) > SUSPICIOUS_BOX_COUNT:
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

    print(f"Procesando {len(imagenes)} imagen(es) (VERIFY_BREED={VERIFY_BREED})...")

    total_cajas = 0
    total_descartadas = 0
    for i, imagen_path in enumerate(imagenes, start=1):
        n_cajas, n_descartadas = procesar_imagen(vlm, imagen_path)
        total_cajas += n_cajas
        total_descartadas += n_descartadas
        sufijo = f" ({n_descartadas} descartada(s) por raza)" if VERIFY_BREED else ""
        print(f"[{i}/{len(imagenes)}] {imagen_path.name}: {n_cajas} caja(s){sufijo}")

    print("\n==== Resumen ====")
    print(f"Total de cajas escritas: {total_cajas}")
    if VERIFY_BREED:
        print(f"Total descartadas por el verificador de raza: {total_descartadas}")
    print("Listo.")


if __name__ == "__main__":
    main()
