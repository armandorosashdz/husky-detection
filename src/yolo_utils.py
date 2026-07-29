"""
Utilidades del detector YOLOv8 (Ultralytics). Separado de vlm_utils.py a propósito:
ese archivo es sobre etiquetado con Qwen (Fase 1), este es sobre correr el detector
ya entrenado — Fase 3 (deployment) y Fase 4 (validación en cascada, junto con
QwenVLM en hybrid_inference.py).
"""

from PIL import Image
from ultralytics import YOLO

import config


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
