"""Fase 3: despliegue del detector YOLOv8s ajustado.

Procesa una imagen, una carpeta, un video o una cámara. Guarda detecciones,
latencias, FPS y evidencias anotadas. La clase YOLOStreamDetector se reutiliza
en la Fase 4 para incorporar la validación con Qwen.

Ejemplos:
    python src/deployment_inference.py --source data/test/images
    python src/deployment_inference.py --source video.mp4
    python src/deployment_inference.py --source 0 --max-frames 300
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


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
    """Métricas de inferencia para un frame."""

    frame_index: int
    source: str
    detections: int
    latency_ms: float
    inference_fps: float


def discover_weights(models_dir: Path) -> Path:
    """Encuentra un único best.pt; acepta otro .pt si no existe best.pt."""
    if not models_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de modelos: {models_dir}")

    best = sorted(models_dir.rglob("best.pt"))
    candidates = best or sorted(models_dir.rglob("*.pt"))

    if not candidates:
        raise FileNotFoundError(f"No se encontraron pesos .pt en {models_dir}")
    if len(candidates) > 1:
        paths = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Se encontraron varios modelos. Use --weights para elegir uno:\n"
            f"  {paths}"
        )
    return candidates[0]


def resolve_device(requested: str) -> str | int:
    """Traduce auto a la primera GPU disponible o a CPU."""
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if requested.lower() == "cpu":
        return "cpu"
    return int(requested) if requested.isdigit() else requested


def uses_cuda(device: str | int) -> bool:
    """Indica si el dispositivo seleccionado ejecuta CUDA."""
    return device != "cpu" and torch.cuda.is_available()


class YOLOStreamDetector:
    """Detector reutilizable con medición sincronizada de latencia."""

    def __init__(
        self,
        weights: Path,
        device: str | int = "auto",
        conf: float = 0.15,
        iou: float = 0.45,
        imgsz: int = 640,
        half: bool = True,
    ) -> None:
        self.weights = Path(weights)
        self.device = resolve_device(str(device))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.half = half and uses_cuda(self.device)
        self.model: YOLO | None = None

    def load(self) -> "YOLOStreamDetector":
        """Carga el modelo y verifica que sea un detector."""
        if not self.weights.exists():
            raise FileNotFoundError(f"No existen los pesos: {self.weights}")

        self.model = YOLO(str(self.weights))
        if self.model.task != "detect":
            raise ValueError(f"Se esperaba una tarea detect, no {self.model.task!r}")
        return self

    @property
    def names(self) -> dict[int, str]:
        """Devuelve el mapa de clases del modelo cargado."""
        if self.model is None:
            raise RuntimeError("Cargue el modelo antes de consultar sus clases.")
        return dict(self.model.names)

    def _synchronize(self) -> None:
        if uses_cuda(self.device):
            torch.cuda.synchronize()

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
        self._synchronize()

    def predict(self, frame: np.ndarray):
        """Regresa el resultado de Ultralytics y la latencia sincronizada."""
        if self.model is None:
            raise RuntimeError("Cargue el modelo antes de ejecutar inferencia.")

        self._synchronize()
        start = time.perf_counter()

        result = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        self._synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        return result, latency_ms

    def extract(
        self,
        result,
        frame_index: int,
        source_name: str,
    ) -> list[Detection]:
        """Convierte las cajas de Ultralytics a registros serializables."""
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
            yield index, str(source), frame
            index += 1

    return generator(), capture


def open_video_writer(
    path: Path,
    capture: cv2.VideoCapture,
    first_frame: np.ndarray,
) -> cv2.VideoWriter:
    """Crea un MP4 con las dimensiones reales del flujo."""
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


def write_csv(path: Path, rows: list, fieldnames: list[str]) -> None:
    """Escribe una colección de dataclasses en CSV."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def run_stream(
    detector: YOLOStreamDetector,
    source: str,
    run_dir: Path,
    save_annotated: bool,
    max_frames: int | None,
) -> dict:
    """Procesa la fuente y conserva evidencias y métricas."""
    run_dir.mkdir(parents=True, exist_ok=False)
    annotated_dir = run_dir / "annotated"
    if save_annotated:
        annotated_dir.mkdir()

    source_path = Path(source)
    is_image_source = source_path.exists() and (
        source_path.is_dir() or source_path.suffix.lower() in IMAGE_EXTENSIONS
    )

    capture = None
    writer = None
    if is_image_source:
        frames = iter_images(source_path, max_frames)
    else:
        if not source.isdigit() and source_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Fuente no compatible: {source}")
        frames, capture = iter_video(source, max_frames)

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
                    saved = cv2.imwrite(str(annotated_dir / source_name), annotated)
                    if not saved:
                        raise RuntimeError(f"No se pudo guardar: {source_name}")
                else:
                    if writer is None:
                        video_path = annotated_dir / "stream_annotated.mp4"
                        writer = open_video_writer(video_path, capture, annotated)
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

    write_csv(
        run_dir / "detections.csv",
        detections,
        list(Detection.__dataclass_fields__),
    )
    write_csv(
        run_dir / "frames.csv",
        frame_metrics,
        list(FrameMetric.__dataclass_fields__),
    )

    latencies = [metric.latency_ms for metric in frame_metrics]
    summary = {
        "weights": str(detector.weights),
        "source": source,
        "device": str(detector.device),
        "half": detector.half,
        "classes": detector.names,
        "confidence_threshold": detector.conf,
        "iou_threshold": detector.iou,
        "image_size": detector.imgsz,
        "frames": len(frame_metrics),
        "detections": len(detections),
        "mean_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "inference_fps": 1000.0 / (sum(latencies) / len(latencies)),
        "total_wall_s": total_wall_s,
        "real_fps": len(frame_metrics) / total_wall_s,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        help="Pesos .pt. Si se omite, busca models/**/best.pt.",
    )
    parser.add_argument(
        "--source",
        default=str(ROOT / "data/test/images"),
        help="Imagen, carpeta, video o índice de cámara.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/deployment",
        help="Directorio base de resultados.",
    )
    parser.add_argument("--run-name", help="Nombre del experimento.")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Desactiva FP16 aun cuando se use CUDA.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guarda imágenes ni video anotado.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    weights = args.weights or discover_weights(ROOT / "models")
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output / run_name

    detector = YOLOStreamDetector(
        weights=weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        half=not args.no_half,
    ).load()

    print("Pesos:", detector.weights)
    print("Dispositivo:", detector.device)
    print("FP16:", detector.half)
    print("Clases:", detector.names)
    print("Resultados:", run_dir)

    detector.warmup(args.warmup)
    summary = run_stream(
        detector=detector,
        source=args.source,
        run_dir=run_dir,
        save_annotated=not args.no_save,
        max_frames=args.max_frames,
    )

    print("\nResumen")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
