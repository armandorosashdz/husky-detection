"""Funciones compartidas del pipeline autónomo YOLOv8s + Qwen3.5.

Este módulo concentra la carga e inferencia de modelos, manejo de imágenes y
flujos, persistencia de resultados y evaluación de las Fases 1, 3 y 4.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from ultralytics import YOLO

import config


IMAGE_EXTENSIONS = set(config.IMG_EXTENSIONS)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
VALIDATOR_FIELDS = [
    "candidate_id",
    "image_name",
    "confidence",
    "model_id",
    "decision",
    "raw_response",
    "latency_ms",
]


@dataclass(frozen=True)
class Detection:
    """Detección individual expresada en píxeles."""

    frame_index: int
    source: str
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class FrameMetric:
    """Métricas de inferencia correspondientes a un frame."""

    frame_index: int
    source: str
    detections: int
    latency_ms: float
    inference_fps: float


# ---------- Utilidades generales ----------


def resolve_device(requested: str | int) -> str | int:
    """Resuelve auto, cpu, cuda y los índices numéricos de GPU."""
    if isinstance(requested, int):
        if not torch.cuda.is_available():
            raise RuntimeError("Se solicitó una GPU, pero CUDA no está disponible.")
        return requested

    normalized = requested.lower()
    if normalized == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Se solicitó CUDA, pero no está disponible.")
        return 0
    if requested.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("Se solicitó una GPU, pero CUDA no está disponible.")
        return int(requested)
    raise ValueError("El dispositivo debe ser auto, cpu, cuda o un índice de GPU.")


def uses_cuda(device: str | int) -> bool:
    """Indica si el dispositivo seleccionado usa CUDA."""
    return device != "cpu" and torch.cuda.is_available()


def synchronize(device: str | int) -> None:
    """Sincroniza CUDA para obtener mediciones de tiempo correctas."""
    if uses_cuda(device):
        torch.cuda.synchronize(int(device))


def release_accelerator_memory() -> None:
    """Libera referencias recolectables y la caché no utilizada de CUDA."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def automatic_gpu_memory_limit(device: str | int) -> str | None:
    """Reserva margen automáticamente en GPUs con menos de 6 GiB."""
    if not uses_cuda(device):
        return None
    total_gib = torch.cuda.get_device_properties(int(device)).total_memory / 1024**3
    if total_gib < 6.0:
        return f"{max(1, int(total_gib - 0.75))}GiB"
    return None


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 sin cargar el archivo completo en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump_atomic(path: Path, payload: Any) -> None:
    """Escribe JSON sin dejar archivos parciales."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_csv_atomic(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Escribe CSV de forma atómica para permitir reanudaciones seguras."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_dataclass_csv(path: Path, rows: list[Any], fieldnames: list[str]) -> None:
    """Escribe una colección de dataclasses en CSV."""
    save_csv_atomic(path, (asdict(row) for row in rows), fieldnames)


def discover_weights(models_dir: Path) -> Path:
    """Prioriza los pesos ajustados del proyecto y evita ambigüedad."""
    if not models_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de modelos: {models_dir}")

    preferred = sorted(models_dir.rglob("yolov8_finetuned_armando.pt"))
    best = sorted(models_dir.rglob("best.pt"))
    candidates = preferred or best or sorted(models_dir.rglob("*.pt"))

    if not candidates:
        raise FileNotFoundError(f"No se encontraron pesos .pt en {models_dir}")
    if len(candidates) > 1:
        paths = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Se encontraron varios modelos. Seleccione uno en la configuración:\n"
            f"  {paths}"
        )
    return candidates[0]


# ---------- Qwen (Fases 1 y 4) ----------


def torch_dtype_from_name(name: str, device: str | int) -> torch.dtype:
    """Convierte el nombre del dtype a su objeto de PyTorch."""
    if name == "auto":
        return torch.float16 if uses_cuda(device) else torch.float32
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    except KeyError as error:
        raise ValueError(f"Dtype no compatible: {name}") from error


def model_input_device(model: Any) -> torch.device:
    """Obtiene el dispositivo de entrada de un modelo normal u offloaded."""
    device = getattr(model, "device", None)
    if device is not None:
        device = torch.device(device)
        if device.type != "meta":
            return device
    return next(
        parameter.device
        for parameter in model.parameters()
        if not parameter.is_meta
    )


class QwenVLM:
    """Carga y ejecuta Qwen para etiquetado o validación binaria."""

    def __init__(
        self,
        model_id: str,
        device: str | int | None = None,
        dtype: str | None = None,
        gpu_memory: str | None = None,
        cpu_memory: str = "24GiB",
        max_new_tokens: int = 512,
        attn_implementation: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device if device is not None else config.DEVICE)
        self.dtype_name = dtype or config.DTYPE
        self.gpu_memory = gpu_memory
        self.cpu_memory = cpu_memory
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.processor = None
        self.model = None

    def load(self) -> "QwenVLM":
        """Carga el procesador y distribuye el modelo según la memoria disponible."""
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        kwargs: dict[str, Any] = {
            "dtype": torch_dtype_from_name(self.dtype_name, self.device),
            "low_cpu_mem_usage": True,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        if uses_cuda(self.device):
            kwargs["device_map"] = "auto"
            if self.gpu_memory:
                kwargs["max_memory"] = {
                    int(self.device): self.gpu_memory,
                    "cpu": self.cpu_memory,
                }

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            **kwargs,
        )
        if not uses_cuda(self.device):
            self.model.to("cpu")
        self.model.eval()
        return self

    def ask_timed(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> tuple[str, float]:
        """Ejecuta imagen+prompt y devuelve texto junto con su latencia."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Modelo no cargado. Llame a load() antes de ask().")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        synchronize(self.device)
        start = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model_input_device(self.model))

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
            )

        synchronize(self.device)
        latency_ms = (time.perf_counter() - start) * 1000.0
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = self.processor.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        del inputs, output_ids, generated
        if uses_cuda(self.device):
            torch.cuda.empty_cache()
        return response, latency_ms

    def ask(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        """Interfaz compatible con auto_labeling.py: devuelve solo el texto."""
        response, _ = self.ask_timed(image, prompt, max_new_tokens)
        return response

    def unload(self) -> None:
        """Libera el modelo antes de cargar otro VLM."""
        self.model = None
        self.processor = None
        release_accelerator_memory()


def strip_thinking(response: str) -> str:
    """Elimina bloques de razonamiento completos o truncados."""
    clean = re.sub(r"<think>.*?</think>", " ", response, flags=re.DOTALL | re.I)
    clean = re.sub(r"<think>.*$", " ", clean, flags=re.DOTALL | re.I)
    return re.sub(r"^.*?</think>", " ", clean, flags=re.DOTALL | re.I)


def parse_boxes(response: str) -> list[list[float]]:
    """Extrae cajas Qwen [x1, y1, x2, y2] en escala 0-1000."""
    match = re.search(r"\[.*\]", strip_thinking(response), re.DOTALL)
    if match is None:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    if len(data) == 4 and all(isinstance(value, (int, float)) for value in data):
        data = [data]

    boxes: list[list[float]] = []
    for item in data:
        box = item
        if isinstance(item, dict):
            box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            boxes.append([float(value) for value in box])
    return boxes


def convert_to_yolo(box: list[float], class_id: int = config.CLASS_ID) -> str:
    """Convierte una caja 0-1000 a una línea YOLO normalizada."""
    x1, y1, x2, y2 = box
    x1_n, y1_n, x2_n, y2_n = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
    width = x2_n - x1_n
    height = y2_n - y1_n
    x_center = x1_n + width / 2
    y_center = y1_n + height / 2
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def normalize_binary_response(response: str) -> str:
    """Extrae Yes/No de una respuesta del VLM."""
    match = re.search(r"\b(yes|no)\b", strip_thinking(response).lower())
    return match.group(1) if match else "unknown"


# ---------- YOLOv8 (Fases 3 y 4) ----------


class YOLODetector:
    """Detector YOLO reutilizable con medición sincronizada de latencia."""

    def __init__(
        self,
        model_path: Path | str = config.YOLO_TRAINED,
        device: str | int | None = None,
        conf: float = config.CONF_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        imgsz: int = config.IMG_SIZE,
        half: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = resolve_device(device if device is not None else config.DEVICE)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.half = half and uses_cuda(self.device)
        self.model = None

    def load(self) -> "YOLODetector":
        """Carga los pesos y verifica que correspondan a detección."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"No existen los pesos: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        if self.model.task != "detect":
            raise ValueError(f"Se esperaba una tarea detect, no {self.model.task!r}")
        return self

    @property
    def names(self) -> dict[int, str]:
        if self.model is None:
            raise RuntimeError("Cargue el modelo antes de consultar sus clases.")
        return dict(self.model.names)

    def warmup(self, iterations: int = 3) -> None:
        """Estabiliza kernels y asignaciones antes de medir."""
        if self.model is None:
            raise RuntimeError("Cargue el modelo antes del calentamiento.")
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(max(0, iterations)):
            self.model.predict(
                dummy,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )
        synchronize(self.device)

    def predict(self, image: Any):
        """Devuelve el resultado de Ultralytics y la latencia medida."""
        if self.model is None:
            raise RuntimeError("Cargue el modelo antes de ejecutar inferencia.")
        synchronize(self.device)
        start = time.perf_counter()
        result = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]
        synchronize(self.device)
        return result, (time.perf_counter() - start) * 1000.0

    def extract(
        self,
        result: Any,
        frame_index: int,
        source_name: str,
    ) -> list[Detection]:
        """Convierte cajas de Ultralytics a registros serializables."""
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    frame_index=frame_index,
                    source=source_name,
                    class_id=class_id,
                    class_name=self.names[class_id],
                    confidence=float(box.conf[0].item()),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        """Interfaz simple compatible con versiones anteriores de utils.py."""
        result, _ = self.predict(image)
        detections = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            detections.append(
                {
                    "box": tuple(float(value) for value in box.xyxy[0].tolist()),
                    "conf": float(box.conf[0].item()),
                    "class_id": int(box.cls[0].item()),
                }
            )
        return detections

    def crop(
        self,
        image: Image.Image,
        box: tuple[float, float, float, float] | list[float],
        padding: int = config.CROP_PADDING,
        max_side: int | None = None,
    ) -> Image.Image:
        """Recorta una detección con margen y reducción opcional."""
        return crop_with_padding(image, box, padding, max_side)

    def unload(self) -> None:
        """Libera los pesos del detector."""
        self.model = None
        release_accelerator_memory()


# ---------- Flujo de despliegue (Fase 3) ----------


def iter_images(
    source: Path,
    max_frames: int | None,
) -> Iterator[tuple[int, str, np.ndarray]]:
    """Lee una imagen o una carpeta en orden alfabético."""
    paths = (
        [source]
        if source.is_file()
        else sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {source}")

    for index, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
        yield index, path.name, frame


def iter_video(
    source: str,
    max_frames: int | None,
) -> tuple[Iterator[tuple[int, str, np.ndarray]], cv2.VideoCapture]:
    """Abre un video o cámara y produce sus frames secuencialmente."""
    capture_source: str | int = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {source}")

    def generator() -> Iterator[tuple[int, str, np.ndarray]]:
        index = 0
        while max_frames is None or index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, source, frame
            index += 1

    return generator(), capture


def open_video_writer(
    path: Path,
    capture: cv2.VideoCapture,
    first_frame: np.ndarray,
) -> cv2.VideoWriter:
    """Crea un MP4 con las dimensiones y FPS del flujo."""
    height, width = first_frame.shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video: {path}")
    return writer


def run_yolo_stream(
    detector: YOLODetector,
    source: str | Path | int,
    run_dir: Path,
    save_annotated: bool = True,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Procesa una fuente y guarda evidencias, detecciones y métricas."""
    run_dir.mkdir(parents=True, exist_ok=False)
    annotated_dir = run_dir / "annotated"
    if save_annotated:
        annotated_dir.mkdir()

    source_text = str(source)
    source_path = Path(source_text)
    is_image_source = source_path.exists() and (
        source_path.is_dir() or source_path.suffix.lower() in IMAGE_EXTENSIONS
    )

    capture = None
    writer = None
    if is_image_source:
        frames = iter_images(source_path, max_frames)
    else:
        if (
            not source_text.isdigit()
            and source_path.suffix.lower() not in VIDEO_EXTENSIONS
        ):
            raise ValueError(f"Fuente no compatible: {source}")
        frames, capture = iter_video(source_text, max_frames)

    detections: list[Detection] = []
    frame_metrics: list[FrameMetric] = []
    wall_start = time.perf_counter()

    try:
        for frame_index, source_name, frame in frames:
            result, latency_ms = detector.predict(frame)
            current = detector.extract(result, frame_index, source_name)
            detections.extend(current)
            frame_metrics.append(
                FrameMetric(
                    frame_index=frame_index,
                    source=source_name,
                    detections=len(current),
                    latency_ms=latency_ms,
                    inference_fps=1000.0 / latency_ms if latency_ms > 0 else 0.0,
                )
            )

            if save_annotated:
                annotated = result.plot()
                if is_image_source:
                    if not cv2.imwrite(str(annotated_dir / source_name), annotated):
                        raise RuntimeError(f"No se pudo guardar: {source_name}")
                else:
                    if writer is None:
                        writer = open_video_writer(
                            annotated_dir / "stream_annotated.mp4",
                            capture,
                            annotated,
                        )
                    writer.write(annotated)

            print(
                f"[{frame_index + 1}] {source_name}: "
                f"{len(current)} detección(es), {latency_ms:.2f} ms"
            )
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()

    total_wall_s = time.perf_counter() - wall_start
    if not frame_metrics:
        raise RuntimeError("La fuente no produjo frames válidos.")

    write_dataclass_csv(
        run_dir / "detections.csv",
        detections,
        list(Detection.__dataclass_fields__),
    )
    write_dataclass_csv(
        run_dir / "frames.csv",
        frame_metrics,
        list(FrameMetric.__dataclass_fields__),
    )

    latencies = [metric.latency_ms for metric in frame_metrics]
    mean_latency_ms = sum(latencies) / len(latencies)
    summary = {
        "weights": str(detector.model_path),
        "source": source_text,
        "device": str(detector.device),
        "half": detector.half,
        "classes": detector.names,
        "confidence_threshold": detector.conf,
        "iou_threshold": detector.iou,
        "image_size": detector.imgsz,
        "frames": len(frame_metrics),
        "detections": len(detections),
        "mean_latency_ms": mean_latency_ms,
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "inference_fps": 1000.0 / mean_latency_ms,
        "total_wall_s": total_wall_s,
        "real_fps": len(frame_metrics) / total_wall_s,
    }
    json_dump_atomic(run_dir / "summary.json", summary)
    return summary


# ---------- Candidatos y validación híbrida (Fase 4) ----------


def list_test_pairs(
    images_dir: Path,
    labels_dir: Path,
    expected_images: int | None,
) -> list[Path]:
    """Valida que cada imagen tenga una etiqueta YOLO homónima."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No existe {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"No existe {labels_dir}")

    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels = sorted(labels_dir.glob("*.txt"))
    if expected_images is not None and len(images) != expected_images:
        raise ValueError(
            f"Se esperaban {expected_images} imágenes, pero se encontraron "
            f"{len(images)} en {images_dir}."
        )
    if not images:
        raise FileNotFoundError(f"No se encontraron imágenes en {images_dir}")
    if {path.stem for path in images} != {path.stem for path in labels}:
        raise ValueError("Los nombres de imágenes y etiquetas no coinciden.")
    return images


def crop_with_padding(
    image: Image.Image,
    box: tuple[float, float, float, float] | list[float],
    padding: int,
    max_side: int | None = None,
) -> Image.Image:
    """Recorta una caja respetando los límites de la imagen."""
    x1, y1, x2, y2 = box
    width, height = image.size
    bounds = (
        max(0, int(x1) - padding),
        max(0, int(y1) - padding),
        min(width, int(x2) + padding),
        min(height, int(y2) + padding),
    )
    crop = image.crop(bounds).convert("RGB")
    if max_side is not None:
        crop.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return crop


def candidate_signature(
    weights: Path,
    test_images: list[Path],
    conf: float,
    nms_iou: float,
    image_size: int,
    crop_padding: int,
    max_crop_side: int,
) -> dict[str, Any]:
    """Identifica las entradas asociadas con una caché de candidatos."""
    return {
        "weights_sha256": sha256_file(weights),
        "images": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in test_images
        ],
        "confidence_threshold": conf,
        "nms_iou": nms_iou,
        "image_size": image_size,
        "crop_padding": crop_padding,
        "max_crop_side": max_crop_side,
    }


def generate_or_load_candidates(
    weights: Path,
    test_images: list[Path],
    candidate_dir: Path,
    device: str | int,
    conf: float,
    nms_iou: float,
    image_size: int,
    crop_padding: int,
    max_crop_side: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Genera candidatos YOLO o recupera una caché compatible."""
    crop_dir = candidate_dir / "crops"
    csv_path = candidate_dir / "candidates.csv"
    summary_path = candidate_dir / "candidate_summary.json"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    signature = candidate_signature(
        weights,
        test_images,
        conf,
        nms_iou,
        image_size,
        crop_padding,
        max_crop_side,
    )
    if csv_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("signature") != signature:
            raise RuntimeError(
                "Los candidatos pertenecen a otra configuración. "
                "Cambie RUN_NAME."
            )
        frame = pd.read_csv(csv_path)
        if any(not Path(path).exists() for path in frame["crop_path"]):
            raise FileNotFoundError("Faltan recortes. Cambie RUN_NAME.")
        print(f"Candidatos recuperados: {len(frame)}")
        return frame, summary
    if csv_path.exists() or summary_path.exists():
        raise RuntimeError("La caché de candidatos está incompleta. Cambie RUN_NAME.")

    detector = YOLODetector(
        model_path=weights,
        device=device,
        conf=conf,
        iou=nms_iou,
        imgsz=image_size,
        half=True,
    ).load()
    rows: list[dict[str, Any]] = []
    synchronize(device)
    wall_start = time.perf_counter()

    try:
        for image_index, image_path in enumerate(test_images, start=1):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            result, _ = detector.predict(image)
            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)

            if boxes is not None:
                for detection_index, box in enumerate(boxes):
                    candidate_id = f"{image_path.stem}__{detection_index:03d}"
                    xyxy = [float(value) for value in box.xyxy[0].tolist()]
                    crop_path = (crop_dir / f"{candidate_id}.jpg").resolve()
                    detector.crop(
                        image,
                        xyxy,
                        padding=crop_padding,
                        max_side=max_crop_side,
                    ).save(crop_path, quality=95)
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "image_name": image_path.name,
                            "image_path": str(image_path.resolve()),
                            "crop_path": str(crop_path),
                            "class_id": int(box.cls[0].item()),
                            "confidence": float(box.conf[0].item()),
                            "x1": xyxy[0],
                            "y1": xyxy[1],
                            "x2": xyxy[2],
                            "y2": xyxy[3],
                        }
                    )
            print(
                f"[{image_index:02d}/{len(test_images):02d}] "
                f"{image_path.name}: {count} candidato(s)"
            )
    finally:
        synchronize(device)
        generation_wall_s = time.perf_counter() - wall_start
        detector.unload()

    if not rows:
        raise RuntimeError("YOLO no generó candidatos.")
    save_csv_atomic(csv_path, rows, list(rows[0]))
    summary = {
        "weights": str(weights.resolve()),
        "images_count": len(test_images),
        "candidates": len(rows),
        "generation_wall_s": generation_wall_s,
        "signature": signature,
    }
    json_dump_atomic(summary_path, summary)
    return pd.DataFrame(rows), summary


def load_validator_records(path: Path, model_id: str) -> list[dict[str, Any]]:
    """Recupera únicamente decisiones binarias completas del mismo modelo."""
    if not path.exists():
        return []
    records = pd.read_csv(path).to_dict("records")
    model_ids = {str(row["model_id"]) for row in records}
    if model_ids and model_ids != {model_id}:
        raise RuntimeError(f"{path} pertenece a otro modelo. Cambie RUN_NAME.")
    return [
        row
        for row in records
        if str(row.get("decision", "")).lower() in {"yes", "no"}
    ]


def run_binary_validator(
    label: str,
    vlm: QwenVLM,
    candidates: pd.DataFrame,
    validation_dir: Path,
    prompt: str,
    max_new_tokens: int,
    limit: int | None = None,
) -> pd.DataFrame:
    """Valida candidatos y guarda un punto de reanudación por recorte."""
    output_path = validation_dir / f"{label}.csv"
    metadata_path = validation_dir / f"{label}_metadata.json"
    candidate_ids = "\n".join(candidates["candidate_id"].astype(str))
    signature = {
        "model_id": vlm.model_id,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "max_new_tokens": max_new_tokens,
        "candidate_ids_sha256": hashlib.sha256(
            candidate_ids.encode("utf-8")
        ).hexdigest(),
    }

    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved != signature:
            raise RuntimeError(
                f"La reanudación de {label} usa otra configuración. "
                "Cambie RUN_NAME."
            )
    elif output_path.exists():
        raise RuntimeError(f"Falta la metadata de {output_path}. Cambie RUN_NAME.")
    else:
        json_dump_atomic(metadata_path, signature)

    records = load_validator_records(output_path, vlm.model_id)
    completed = {str(row["candidate_id"]) for row in records}
    pending = candidates[
        ~candidates["candidate_id"].astype(str).isin(completed)
    ]
    if limit is not None:
        pending = pending.head(limit)

    total = len(pending)
    for index, row in enumerate(pending.itertuples(), start=1):
        with Image.open(row.crop_path) as crop:
            image = crop.convert("RGB")
        raw, latency_ms = vlm.ask_timed(
            image,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        decision = normalize_binary_response(raw)
        records.append(
            {
                "candidate_id": row.candidate_id,
                "image_name": row.image_name,
                "confidence": float(row.confidence),
                "model_id": vlm.model_id,
                "decision": decision,
                "raw_response": raw,
                "latency_ms": latency_ms,
            }
        )
        save_csv_atomic(output_path, records, VALIDATOR_FIELDS)
        print(
            f"[{label} {index:03d}/{total:03d}] {row.candidate_id}: "
            f"{decision} ({latency_ms:.1f} ms)"
        )
    return pd.read_csv(output_path)


def validate_qwen_model(
    label: str,
    model_id: str,
    candidates: pd.DataFrame,
    validation_dir: Path,
    prompt: str,
    device: str | int,
    dtype: str,
    gpu_memory: str | None,
    cpu_memory: str,
    max_new_tokens: int,
    preflight_count: int,
    preflight_only: bool,
) -> pd.DataFrame:
    """Carga un Qwen, ejecuta preflight/validación y libera su memoria."""
    print(f"\nCargando {model_id}...")
    vlm = QwenVLM(
        model_id=model_id,
        device=device,
        dtype=dtype,
        gpu_memory=gpu_memory,
        cpu_memory=cpu_memory,
        max_new_tokens=max_new_tokens,
        attn_implementation="sdpa",
    ).load()
    try:
        results = run_binary_validator(
            label=label,
            vlm=vlm,
            candidates=candidates,
            validation_dir=validation_dir,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            limit=preflight_count,
        )
        recent = results.tail(min(preflight_count, len(results)))
        if (recent["decision"] == "unknown").any():
            raise RuntimeError(
                f"El preflight de {label} produjo respuestas no binarias. "
                f"Revise {validation_dir / (label + '.csv')}."
            )

        if not preflight_only:
            results = run_binary_validator(
                label=label,
                vlm=vlm,
                candidates=candidates,
                validation_dir=validation_dir,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        return results
    finally:
        vlm.unload()


# ---------- Evaluación de la cascada (Fase 4) ----------


def box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Calcula IoU entre dos cajas xyxy."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_ground_truths(
    test_images: list[Path],
    labels_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Convierte etiquetas YOLO normalizadas a cajas xyxy en píxeles."""
    ground_truths: dict[str, list[dict[str, Any]]] = {}
    for image_path in test_images:
        with Image.open(image_path) as image:
            width, height = image.size
        boxes: list[dict[str, Any]] = []
        label_path = labels_dir / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Etiqueta inválida en {label_path}: {line!r}")
            class_id, xc, yc, box_width, box_height = map(float, values)
            boxes.append(
                {
                    "class_id": int(class_id),
                    "box": (
                        (xc - box_width / 2) * width,
                        (yc - box_height / 2) * height,
                        (xc + box_width / 2) * width,
                        (yc + box_height / 2) * height,
                    ),
                }
            )
        ground_truths[image_path.name] = boxes
    return ground_truths


def evaluate_pipeline(
    candidates: pd.DataFrame,
    accepted_ids: set[str],
    ground_truths: dict[str, list[dict[str, Any]]],
    eval_iou: float,
) -> dict[str, Any]:
    """Evalúa predicciones ordenadas por confianza con AP de 101 puntos."""
    predictions = candidates[
        candidates["candidate_id"].astype(str).isin(accepted_ids)
    ].sort_values("confidence", ascending=False)
    total_gt = sum(len(boxes) for boxes in ground_truths.values())
    matched = {image: set() for image in ground_truths}
    true_positives: list[float] = []
    false_positives: list[float] = []

    for row in predictions.itertuples():
        predicted_box = (row.x1, row.y1, row.x2, row.y2)
        targets = ground_truths[row.image_name]
        best_iou = 0.0
        best_index = None
        for target_index, target in enumerate(targets):
            if target_index in matched[row.image_name]:
                continue
            if int(row.class_id) != target["class_id"]:
                continue
            current_iou = box_iou(predicted_box, target["box"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = target_index

        is_true_positive = best_index is not None and best_iou >= eval_iou
        if is_true_positive:
            matched[row.image_name].add(best_index)
        true_positives.append(1.0 if is_true_positive else 0.0)
        false_positives.append(0.0 if is_true_positive else 1.0)

    if predictions.empty:
        return {
            "precision_curve": np.array([]),
            "recall_curve": np.array([]),
            "ap50": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": total_gt,
            "predictions": 0,
        }

    tp_cumulative = np.cumsum(true_positives)
    fp_cumulative = np.cumsum(false_positives)
    recall = tp_cumulative / max(total_gt, 1)
    precision = tp_cumulative / np.maximum(
        tp_cumulative + fp_cumulative,
        1e-12,
    )
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_levels = np.linspace(0.0, 1.0, 101)
    interpolated = [
        precision_envelope[recall >= level].max()
        if np.any(recall >= level)
        else 0.0
        for level in recall_levels
    ]
    tp = int(tp_cumulative[-1])
    fp = int(fp_cumulative[-1])
    return {
        "precision_curve": precision,
        "recall_curve": recall,
        "ap50": float(np.mean(interpolated)),
        "tp": tp,
        "fp": fp,
        "fn": total_gt - tp,
        "predictions": len(predictions),
    }


def compare_pipelines(
    candidates: pd.DataFrame,
    candidate_summary: dict[str, Any],
    validator_results: dict[str, pd.DataFrame],
    ground_truths: dict[str, list[dict[str, Any]]],
    image_count: int,
    eval_iou: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Calcula métricas comunes y latencia extremo a extremo."""
    accepted = {
        "YOLOv8s solo": set(candidates["candidate_id"].astype(str)),
        "YOLOv8s + Qwen 0.8B": set(
            validator_results["qwen_0_8b"].loc[
                validator_results["qwen_0_8b"]["decision"] == "yes",
                "candidate_id",
            ].astype(str)
        ),
        "YOLOv8s + Qwen 2B": set(
            validator_results["qwen_2b"].loc[
                validator_results["qwen_2b"]["decision"] == "yes",
                "candidate_id",
            ].astype(str)
        ),
    }
    evaluations = {
        name: evaluate_pipeline(candidates, ids, ground_truths, eval_iou)
        for name, ids in accepted.items()
    }

    yolo_seconds = float(candidate_summary["generation_wall_s"])
    qwen08_seconds = validator_results["qwen_0_8b"]["latency_ms"].sum() / 1000.0
    qwen2_seconds = validator_results["qwen_2b"]["latency_ms"].sum() / 1000.0
    total_seconds = {
        "YOLOv8s solo": yolo_seconds,
        "YOLOv8s + Qwen 0.8B": yolo_seconds + qwen08_seconds,
        "YOLOv8s + Qwen 2B": yolo_seconds + qwen2_seconds,
    }

    rows: list[dict[str, Any]] = []
    for pipeline_name, evaluation in evaluations.items():
        elapsed = total_seconds[pipeline_name]
        tp, fp, fn = evaluation["tp"], evaluation["fp"], evaluation["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "pipeline": pipeline_name,
                "map50_common": evaluation["ap50"],
                "precision_at_conf": precision,
                "recall_at_conf": recall,
                "f1_at_conf": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "accepted_predictions": evaluation["predictions"],
                "latency_ms_per_image": elapsed * 1000.0 / image_count,
                "real_fps": image_count / elapsed if elapsed > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows), evaluations


def save_pr_curves(
    evaluations: dict[str, dict[str, Any]],
    metrics_dir: Path,
) -> list[Path]:
    """Guarda tres curvas individuales y una comparación conjunta."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "YOLOv8s solo": "#1f77b4",
        "YOLOv8s + Qwen 0.8B": "#2ca02c",
        "YOLOv8s + Qwen 2B": "#d62728",
    }
    slugs = {
        "YOLOv8s solo": "yolov8s_solo",
        "YOLOv8s + Qwen 0.8B": "yolov8s_plus_qwen_0_8b",
        "YOLOv8s + Qwen 2B": "yolov8s_plus_qwen_2b",
    }
    paths: list[Path] = []
    for pipeline_name, evaluation in evaluations.items():
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.plot(
            evaluation["recall_curve"],
            evaluation["precision_curve"],
            color=colors[pipeline_name],
            linewidth=2,
            label=f"AP@0.5={evaluation['ap50']:.4f}",
        )
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precisión")
        axis.set_title(f"Precisión–Recall — {pipeline_name}")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = metrics_dir / f"pr_{slugs[pipeline_name]}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        paths.append(path)

    figure, axis = plt.subplots(figsize=(8, 6))
    for pipeline_name, evaluation in evaluations.items():
        axis.plot(
            evaluation["recall_curve"],
            evaluation["precision_curve"],
            color=colors[pipeline_name],
            linewidth=2,
            label=f"{pipeline_name}: {evaluation['ap50']:.4f}",
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precisión")
    axis.set_title("Comparación Precisión–Recall @ IoU 0.5")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    comparison_path = metrics_dir / "pr_comparison.png"
    figure.savefig(comparison_path, dpi=200)
    plt.close(figure)
    paths.append(comparison_path)
    return paths


def validate_completed_results(
    results: pd.DataFrame,
    candidates: pd.DataFrame,
    label: str,
) -> None:
    """Evita comparar resultados parciales, duplicados o desconocidos."""
    if len(results) != len(candidates):
        raise RuntimeError(
            f"{label} procesó {len(results)} de {len(candidates)} candidatos."
        )
    if results["candidate_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{label} contiene candidate_id duplicados.")
    unknown = int((results["decision"] == "unknown").sum())
    if unknown:
        raise RuntimeError(
            f"{label} produjo {unknown} respuesta(s) no binarias. "
            "Revise raw_response."
        )


def validate_hybrid_settings(
    candidate_conf: float,
    nms_iou: float,
    eval_iou: float,
    preflight_count: int,
    max_new_tokens: int,
    image_size: int,
    crop_padding: int,
    max_crop_side: int,
) -> None:
    """Valida la configuración editable de hybrid_inference.py."""
    if not 0.0 <= candidate_conf <= 1.0:
        raise ValueError("CANDIDATE_CONF debe estar entre 0 y 1.")
    if not 0.0 <= nms_iou <= 1.0 or not 0.0 <= eval_iou <= 1.0:
        raise ValueError("NMS_IOU y EVAL_IOU deben estar entre 0 y 1.")
    if preflight_count < 0 or max_new_tokens < 1:
        raise ValueError("PREFLIGHT_COUNT debe ser no negativo y los tokens positivos.")
    if image_size < 1 or max_crop_side < 1 or crop_padding < 0:
        raise ValueError("Las dimensiones deben ser positivas y el padding no negativo.")
