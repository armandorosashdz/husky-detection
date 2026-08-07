"""Utilidades de Qwen (VLM) y YOLOv8, compartidas por auto_labeling.py y
hybrid_inference.py."""

import gc
import json
import re

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from ultralytics import YOLO

import config


def liberar_memoria_cuda() -> None:
    """Libera VRAM tras cada QwenVLM.ask() -- sin esto, la memoria se
    fragmenta en loops largos y termina en OutOfMemoryError. empty_cache()
    va en try/except porque puede fallar si la GPU ya está sin memoria."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"   (aviso: no se pudo limpiar cache CUDA: {e})")
    gc.collect()


# ---------- Qwen (VLM) ----------

class QwenVLM:
    """Envoltorio sobre un modelo Qwen VL: carga + inferencia genérica de imagen+texto."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = None
        self.model = None

    def load(self):
        """Carga el processor y el modelo. Debe llamarse antes de ask()."""
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # "auto" reparte el modelo entre GPUs (Kaggle T4x2); en CPU hace que
        # accelerate intente offload a disco, así que solo se usa con CUDA.
        device_map = "auto" if torch.cuda.is_available() else config.DEVICE

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            dtype=getattr(torch, config.DTYPE),
            device_map=device_map,
        )
        return self

    def ask(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
        """Corre el modelo sobre una imagen + prompt y regresa el texto de salida.

        enable_thinking=False: evita que Qwen3.5 (4B/9B) gaste max_new_tokens
        en un bloque <think> antes de responder.
        do_sample=False: greedy, para resultados reproducibles.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Modelo no cargado. Llama a load() antes de ask().")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        response = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

        del inputs, output_ids
        liberar_memoria_cuda()

        return response.strip()


def parse_boxes(response: str) -> list[list[float]]:
    """Parsea la respuesta cruda de Qwen a una lista de cajas [x1, y1, x2, y2]
    en escala 0-1000. Acepta el esquema nativo de Qwen3.5
    ([{"bbox_2d": [...]}, ...]) y, por robustez, listas planas o anidadas.
    Quita cualquier bloque <think>...</think> antes de buscar el JSON."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match is None:
        return []

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    if len(data) == 4 and all(isinstance(v, (int, float)) for v in data):
        data = [data]

    boxes = []
    for item in data:
        box = item
        if isinstance(item, dict):
            box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            boxes.append([float(v) for v in box])

    return boxes


def convert_to_yolo(box: list[float], class_id: int = config.CLASS_ID) -> str:
    """Convierte una caja [x1, y1, x2, y2] en escala 0-1000 a una línea de
    anotación YOLO: "class_id x_center y_center width height" normalizado 0-1."""
    x1, y1, x2, y2 = box

    x1_n, y1_n = x1 / 1000, y1 / 1000
    x2_n, y2_n = x2 / 1000, y2 / 1000

    width = x2_n - x1_n
    height = y2_n - y1_n
    x_center = x1_n + width / 2
    y_center = y1_n + height / 2

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# ---------- YOLOv8 (detector) ----------

class YOLODetector:
    """Envoltorio sobre un modelo YOLOv8 de Ultralytics: carga + inferencia + recorte."""

    def __init__(self, model_path=config.YOLO_TRAINED):
        self.model_path = model_path
        self.model = None

    def load(self):
        """Carga los pesos del modelo. Debe llamarse antes de detect()."""
        self.model = YOLO(str(self.model_path))
        return self

    def detect(self, image: Image.Image) -> list[dict]:
        """Corre inferencia y regresa detecciones como
        {"box": (x1,y1,x2,y2) en píxeles, "conf": float, "class_id": int}."""
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama a load() antes de detect().")

        resultados = self.model.predict(
            image,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False,
        )

        detecciones = []
        for resultado in resultados:
            for caja in resultado.boxes:
                x1, y1, x2, y2 = caja.xyxy[0].tolist()
                detecciones.append({
                    "box": (x1, y1, x2, y2),
                    "conf": float(caja.conf[0]),
                    "class_id": int(caja.cls[0]),
                })

        return detecciones

    def crop(self, image: Image.Image, box, padding: int = config.CROP_PADDING) -> Image.Image:
        """Recorta una detección (box en píxeles) con margen extra."""
        x1, y1, x2, y2 = box
        ancho, alto = image.size

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(ancho, x2 + padding)
        y2 = min(alto, y2 + padding)

        return image.crop((x1, y1, x2, y2))
