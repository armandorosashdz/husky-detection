"""
Utilidades del VLM (Qwen). El nombre del archivo es genérico a propósito: todo lo
de aquí es específico de Qwen EXCEPTO `convert_to_yolo`, que es una conversión de
formato de cajas puramente geométrica.

Se reutiliza tanto en Fase 1 (auto_labeling.py, prompt de detección) como en
Fase 4 (hybrid_inference.py, prompt de validación binaria) para no repetir la
carga del modelo en dos lugares.
"""

import json
import re

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

import config


class QwenVLM:
    """Envoltorio sobre un modelo Qwen VL: carga + inferencia genérica de imagen+texto."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = None
        self.model = None

    def load(self):
        """Carga el processor y el modelo. Debe llamarse antes de ask()."""
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            dtype=getattr(torch, config.DTYPE),
            device_map=config.DEVICE,
        )
        return self

    def ask(self, image: Image.Image, prompt: str) -> str:
        """Corre el modelo sobre una imagen + prompt y regresa el texto crudo de salida.

        Genérico a propósito: no asume el formato de la respuesta, ya que el mismo
        método sirve tanto para pedir boxes (Fase 1) como para pedir Yes/No (Fase 4).

        Sigue el patrón de uso oficial de la tarjeta del modelo en Hugging Face
        (huggingface.co/Qwen/Qwen3.5-0.8B).
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
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=512)

        response = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

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
    """
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
    para no crear uno extra.
    """
    x1, y1, x2, y2 = box

    x1_n, y1_n = x1 / 1000, y1 / 1000
    x2_n, y2_n = x2 / 1000, y2 / 1000

    width = x2_n - x1_n
    height = y2_n - y1_n
    x_center = x1_n + width / 2
    y_center = y1_n + height / 2

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
