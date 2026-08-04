"""
Utilidades generales del proyecto: todo lo relacionado a Qwen (VLM) y a YOLOv8
(detector) vive junto aquí, a propósito, en vez de repartido en varios archivos.

Sección Qwen (Fase 1: auto_labeling.py, Fase 4: hybrid_inference.py):
- QwenVLM: carga + inferencia genérica de imagen+texto.
- parse_boxes: parsea la respuesta cruda de Qwen a cajas [x1,y1,x2,y2].
- convert_to_yolo: convierte una caja a formato de anotación YOLO (no es
  específica de Qwen, es pura geometría, pero se dejó aquí para no repartir).

Sección YOLOv8 (Fase 3/4: hybrid_inference.py):
- YOLODetector: carga el detector ya entrenado + inferencia + recorte de detecciones.
"""

import json
import re

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from ultralytics import YOLO

import config


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

        # "auto" deja que accelerate reparta las capas del modelo entre todas las
        # GPUs visibles (útil en máquinas multi-GPU como Kaggle T4x2, para modelos
        # que no caben en float32 en una sola tarjeta). PERO solo se usa si hay
        # CUDA disponible: en esta laptop (sin GPU) "auto" hizo que accelerate
        # ofreciera "offload a disco" incluso para el modelo 0.8B (mensaje
        # "parameters are on the meta device... offloaded to the cpu and disk"),
        # generando E/S de disco pesada y riesgo real de tumbar la máquina (ver
        # CLAUDE.md). Sin GPU, se fuerza config.DEVICE ("cpu") explícito, tal
        # como funcionaba antes de agregar el soporte multi-GPU.
        device_map = "auto" if torch.cuda.is_available() else config.DEVICE

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            dtype=getattr(torch, config.DTYPE),
            device_map=device_map,
        )
        return self

    def ask(self, image: Image.Image, prompt: str) -> str:
        """Corre el modelo sobre una imagen + prompt y regresa el texto crudo de salida.

        Genérico a propósito: no asume el formato de la respuesta, ya que el mismo
        método sirve tanto para pedir boxes (Fase 1) como para pedir Yes/No (Fase 4).

        Sigue el patrón de uso oficial de la tarjeta del modelo en Hugging Face
        (huggingface.co/Qwen/Qwen3.5-0.8B).

        enable_thinking=False: Qwen3.5 (sobre todo el 4B y 9B) genera por defecto
        un bloque de razonamiento <think>...</think> antes de la respuesta final.
        No lo necesitamos para detección de cajas ni para el Yes/No de validación,
        y consumía max_new_tokens sin llegar a la respuesta real. Ver CLAUDE.md.
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
            output_ids = self.model.generate(**inputs, max_new_tokens=512)

        response = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

        # Cada imagen trae una resolución distinta -> cada llamada reserva tensores
        # de tamaño distinto en CUDA (input, KV-cache). Sin liberar explícitamente,
        # la memoria se fragmenta con cada llamada nueva y en un loop de muchas
        # imágenes (ej. auto_labeling.py sobre las 100) termina en OutOfMemoryError
        # aunque cada llamada individual quepa de sobra. No afecta a CPU.
        del inputs, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()


def parse_boxes(response: str) -> list[list[float]]:
    """Parsea la respuesta cruda de Qwen (prompt de detección) a una lista de cajas
    [x1, y1, x2, y2] en escala 0-1000 (esquina superior-izq. y inferior-der.).

    Específico de Qwen: PROMPT_LABELING (config.py) ya pide directamente el esquema
    nativo de "grounding" de Qwen3.5, que es como responde de todos modos aunque se
    le pida otra cosa:
        [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}, ...]
    Por eso esta función acepta esa forma (lista de objetos con clave "bbox_2d") y,
    por robustez, también lista plana de 4 números o lista de listas.

    Orden [x1, y1, x2, y2] confirmado visualmente dibujando las cajas sobre 3
    imágenes de prueba: ver data/labels_check/test_orden_*.jpg (generadas por
    src/test_box_order.py).

    Quita primero cualquier bloque <think>...</think>: aunque QwenVLM.ask() ya
    pide enable_thinking=False, algunos tamaños (ej. 0.8B) pueden entrar en modo
    thinking de todos modos; sin esto, corchetes dentro del razonamiento podrían
    confundir el regex de abajo.
    """
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match is None:
        return []

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    # Acepta tanto una sola caja [a, b, c, d] como una lista de cajas [[a,b,c,d], ...]
    if len(data) == 4 and all(isinstance(v, (int, float)) for v in data):
        data = [data]

    # Acepta tanto lista de listas como lista de dicts con clave "bbox_2d" (o "bbox"/"box")
    boxes = []
    for item in data:
        box = item
        if isinstance(item, dict):
            box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            boxes.append([float(v) for v in box])

    return boxes


def convert_to_yolo(box: list[float], class_id: int = config.CLASS_ID) -> str:
    """Convierte una caja [x1, y1, x2, y2] en escala 0-1000 (formato de parse_boxes)
    al formato de anotación YOLOv8: "class_id x_center y_center width height",
    normalizado 0-1. Una línea de este tipo por caja va en el .txt de cada imagen.

    NO es específica de Qwen (pura geometría/formato) — se dejó en este archivo
    para no repartir utilidades chicas en varios módulos.
    """
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
        """Corre inferencia sobre una imagen y regresa las detecciones como una
        lista de dicts: {"box": (x1,y1,x2,y2) en píxeles, "conf": float, "class_id": int}.

        Usa los thresholds de config.py (CONF_THRESHOLD bajo a propósito, para que
        la cascada de Qwen filtre después los falsos positivos).
        """
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
        """Recorta la región de una detección (box en píxeles, formato x1,y1,x2,y2),
        con un margen extra (padding, en píxeles) para no cortar al perro justo en el
        borde. Antes de mandar el recorte al validador Qwen (Fase 4).
        """
        x1, y1, x2, y2 = box
        ancho, alto = image.size

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(ancho, x2 + padding)
        y2 = min(alto, y2 + padding)

        return image.crop((x1, y1, x2, y2))
