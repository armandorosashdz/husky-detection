"""
Fase 1 - Auto-etiquetado de Huskies con Qwen3.5-2B (todo en un script)
--------------------------------------------------------------------------
1) Recorre ./images, le pide al VLM que detecte los huskies usando el
   formato NATIVO de Qwen (bbox_2d, escala 0-1000, esquinas x1y1x2y2),
   pidiendo explicitamente cajas AJUSTADAS solo a la parte visible de
   cada perro (para minimizar cajas infladas por oclusion) y excluyendo
   otras razas de perro.
2) VALIDACION AUTOMATICA (anticipo de la cascada de Fase 4): por cada caja
   detectada, se recorta esa region de la imagen original y se le pregunta
   al modelo, en una consulta binaria aparte, si eso es realmente un husky.
   Las cajas que el modelo no confirma se descartan automaticamente.
3) Convierte las cajas confirmadas a formato YOLOv8 y escribe un .txt por
   imagen en ./labels (class_id x_center y_center width height, 0-1)
4) Dibuja las cajas finales sobre copias de las imagenes en ./verify.

Uso:
    python auto_label_and_verify.py
"""

import gc
import re
import json
from pathlib import Path

import torch
import cv2
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# ---------------- CONFIG ----------------
MODEL_ID = "Qwen/Qwen3.5-2B"   # version pedida en la consigna para Fase 1 (2B o 4B)
IMAGES_DIR = Path("images")
LABELS_DIR = Path("labels")
VERIFY_DIR = Path("verify")
CLASS_ID = 0
CLASS_NAMES = {0: "husky"}

# Limita el lado mas largo de la imagen antes de pasarla al modelo.
# Sin esto, una sola imagen de resolucion muy alta puede generar tantos
# "tokens visuales" que provoca un pico de VRAM y tumba el proceso,
# incluso si las demas imagenes funcionan bien.
# Punto intermedio entre 768 (mas seguro en VRAM) y 1024 (mas detalle):
# suficiente resolucion extra para distinguir mejor perros muy juntos,
# sin arriesgar tanto la memoria de una GPU de 6GB con el modelo 2B +
# el paso de verificacion (que tambien consume VRAM).
MAX_IMAGE_SIDE = 896

# Si el modelo devuelve mas cajas que esto en una sola imagen, probablemente
# entro en un patron repetitivo/degenerado en vez de detectar huskies reales.
# No se descartan, pero se marca la imagen para que la revises a mano.
SUSPICIOUS_BOX_COUNT = 12

# Usamos el formato NATIVO de Qwen (bbox_2d, escala 0-1000, x1y1x2y2)
# en vez de pedirle un formato inventado que el modelo tiende a ignorar.
PROMPT = (
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

# Segundo paso: por cada caja detectada, se recorta esa region de la imagen
# ORIGINAL y se le pregunta al modelo, en una consulta binaria separada,
# si lo que hay ahi es realmente un husky. Esto es una version adelantada
# de la validacion en cascada de la Fase 4, aplicada ya en el auto-etiquetado
# para reducir falsos positivos de raza (ej. otro perro etiquetado como husky).
ENABLE_BREED_VERIFICATION = True

VERIFY_PROMPT = (
    "Look only at this cropped image. Is the dog shown specifically a "
    "Siberian Husky (not any other breed, and not a wolf)? "
    "Answer with exactly one word: Yes or No."
)
# -----------------------------------------


def load_model():
    print(f"Cargando {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    print("Modelo cargado. Usando:", "cuda" if torch.cuda.is_available() else "cpu")
    return model, processor


def extract_json(text: str):
    """Extrae las cajas detectadas, incluso si la respuesta del modelo
    se corto a la mitad (arreglo JSON sin cerrar) por el limite de tokens,
    o si el modelo metio asteriscos de markdown (**) dentro del JSON."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    text = text.replace("**", "")  # el modelo a veces envuelve numeros en negrita markdown

    # intento 1: el arreglo JSON completo y bien formado
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # intento 2: la respuesta quedo cortada antes de cerrar el arreglo.
    # Rescatamos cada objeto {"bbox_2d": [...], ...} que si se alcanzo a
    # generar completo, descartando solo el ultimo si quedo a medias.
    boxes = []
    for obj_match in re.finditer(r"\{[^{}]*\"bbox_2d\"\s*:\s*\[[^\]]*\][^{}]*\}", text):
        try:
            boxes.append(json.loads(obj_match.group(0)))
        except json.JSONDecodeError:
            continue
    return boxes


def load_and_resize(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    longest = max(w, h)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        image = image.resize(new_size, Image.LANCZOS)
    return image


def run_inference(model, processor, image_path: Path) -> str:
    image = load_and_resize(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]

    # Liberar memoria GPU explicitamente: sin esto, la VRAM se va acumulando
    # imagen tras imagen y termina en "CUDA error: out of memory".
    del inputs, output_ids, generated
    safe_cuda_cleanup()

    return text


def safe_cuda_cleanup():
    """empty_cache() puede fallar tambien si la GPU quedo sin memoria
    justo antes; no dejamos que eso tumbe el script."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"   (aviso: no se pudo limpiar cache CUDA: {e})")
    gc.collect()


def clamp01(v):
    return max(0.0, min(1.0, float(v)))


def bbox_2d_to_yolo(bbox_2d):
    """Convierte [x1,y1,x2,y2] normalizado 0-1000 (formato nativo Qwen)
    a x_center,y_center,width,height normalizado 0-1 (formato YOLO)."""
    x1, y1, x2, y2 = bbox_2d
    x1n, y1n, x2n, y2n = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
    # por si el modelo invierte esquinas
    x1n, x2n = min(x1n, x2n), max(x1n, x2n)
    y1n, y2n = min(y1n, y2n), max(y1n, y2n)
    xc = (x1n + x2n) / 2
    yc = (y1n + y2n) / 2
    w = x2n - x1n
    h = y2n - y1n
    return clamp01(xc), clamp01(yc), clamp01(w), clamp01(h)


def verify_is_husky(model, processor, crop_image: Image.Image) -> bool:
    """Le pregunta al modelo, en una consulta binaria aparte, si el recorte
    es realmente un husky. Devuelve False ante cualquier duda o error, para
    ser conservadores: mejor perder una caja valida que meter una mala."""
    if crop_image.width < 10 or crop_image.height < 10:
        return False

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": crop_image},
                {"type": "text", "text": VERIFY_PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]

    del inputs, output_ids, generated
    safe_cuda_cleanup()

    return text.strip().lower().startswith("yes")


def crop_from_normalized_box(image: Image.Image, xc, yc, w, h) -> Image.Image:
    """Recorta la region de la imagen ORIGINAL (sin reescalar) a partir de
    una caja normalizada 0-1, para que la verificacion vea el mejor detalle
    posible en vez de la version reducida usada para deteccion."""
    img_w, img_h = image.size
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return image.crop((x1, y1, x2, y2))


def draw_boxes(img_path: Path, label_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"   ! no se pudo leer {img_path} para verificacion")
        return
    h, w = img.shape[:2]

    lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, xc, yc, bw, bh = parts
        cls_id = int(cls_id)
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = CLASS_NAMES.get(cls_id, str(cls_id))
        cv2.putText(img, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(str(out_path), img)


def main():
    LABELS_DIR.mkdir(exist_ok=True)
    VERIFY_DIR.mkdir(exist_ok=True)

    if not IMAGES_DIR.exists():
        raise SystemExit(f"No se encontro la carpeta '{IMAGES_DIR}/'. Creala y pon tus 100 imagenes ahi.")

    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Encontradas {len(image_paths)} imagenes en {IMAGES_DIR}/")

    model, processor = load_model()

    total_boxes = 0
    discarded_boxes = 0
    empty_images = 0

    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {img_path.name}")

        try:
            raw_text = run_inference(model, processor, img_path)
        except Exception as e:
            print(f"   ! error de inferencia: {e}")
            safe_cuda_cleanup()
            continue

        raw_boxes = extract_json(raw_text)
        if not raw_boxes:
            print(f"   (sin detecciones parseables) respuesta cruda: {raw_text[:200]!r}")
            empty_images += 1

        # imagen original sin reescalar, para recortes de verificacion nitidos
        original_image = Image.open(img_path).convert("RGB")

        label_path = LABELS_DIR / (img_path.stem + ".txt")
        lines = []
        for box in raw_boxes:
            bbox_2d = box.get("bbox_2d") if isinstance(box, dict) else None
            if not bbox_2d or len(bbox_2d) != 4:
                continue
            xc, yc, w, h = bbox_2d_to_yolo(bbox_2d)

            if ENABLE_BREED_VERIFICATION:
                crop = crop_from_normalized_box(original_image, xc, yc, w, h)
                try:
                    is_husky = verify_is_husky(model, processor, crop)
                except Exception as e:
                    print(f"   ! error en verificacion, se descarta la caja por seguridad: {e}")
                    safe_cuda_cleanup()
                    is_husky = False
                if not is_husky:
                    discarded_boxes += 1
                    print(f"   (caja descartada por el validador: no parece husky)")
                    continue

            lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(lines))
        total_boxes += len(lines)
        print(f"   -> {len(lines)} caja(s) escritas en {label_path}")
        if len(lines) > SUSPICIOUS_BOX_COUNT:
            print(f"   !! sospechoso: {len(lines)} cajas es mucho, revisa {img_path.name} a mano en verify/")

        draw_boxes(img_path, label_path, VERIFY_DIR / img_path.name)

    print("\n==== Resumen ====")
    print(f"Total de cajas detectadas: {total_boxes}")
    print(f"Cajas descartadas por el validador de raza: {discarded_boxes}")
    print(f"Imagenes sin ninguna deteccion: {empty_images}")
    print(f"Labels YOLO en:      {LABELS_DIR}/")
    print(f"Imagenes con cajas dibujadas en: {VERIFY_DIR}/  <- revisalas a ojo")


if __name__ == "__main__":
    main()
