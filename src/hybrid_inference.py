"""Fase 4: evaluación local de YOLOv8s con validación binaria Qwen3.5.

Genera una sola colección de candidatos YOLO y entrega los mismos recortes a
Qwen3.5-0.8B y Qwen3.5-2B. Compara los tres pipelines con AP@0.5, TP, FP, FN,
latencia y FPS. Los resultados de cada VLM se guardan después de cada recorte,
por lo que una ejecución interrumpida puede reanudarse con el mismo --run-name.

Ejemplo completo, desde la raíz del repositorio:
    python src/hybrid_inference.py --device auto

Prueba corta de integración:
    python src/hybrid_inference.py --run-name smoke --preflight-only
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_MODELS = {
    "qwen_0_8b": "Qwen/Qwen3.5-0.8B",
    "qwen_2b": "Qwen/Qwen3.5-2B",
}

PROMPT_VALIDATION = (
    "Inspect the animal visible in this image crop. "
    "Answer Yes only if the animal is a Siberian husky dog. "
    "Answer No if it is another dog breed, another animal, "
    "there is no animal, or the visual evidence is insufficient. "
    "Return exactly one word: Yes or No."
)

VALIDATOR_FIELDS = [
    "candidate_id",
    "image_name",
    "confidence",
    "model_id",
    "decision",
    "raw_response",
    "latency_ms",
]


def json_dump_atomic(path: Path, payload: Any) -> None:
    """Escribe JSON sin dejar un archivo parcial si la sesión se interrumpe."""
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


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 sin cargar el archivo completo en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_weights(models_dir: Path) -> Path:
    """Prioriza el modelo ajustado del proyecto y evita ambigüedad."""
    if not models_dir.exists():
        raise FileNotFoundError(f"No existe el directorio: {models_dir}")

    preferred = sorted(models_dir.rglob("yolov8_finetuned_armando.pt"))
    best = sorted(models_dir.rglob("best.pt"))
    candidates = preferred or best or sorted(models_dir.rglob("*.pt"))

    if not candidates:
        raise FileNotFoundError(f"No se encontraron pesos .pt en {models_dir}")
    if len(candidates) > 1:
        options = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Se encontraron varios modelos. Seleccione uno con --weights:\n"
            f"  {options}"
        )
    return candidates[0]


def resolve_device(requested: str) -> str | int:
    """Resuelve auto, cpu, cuda y los índices numéricos de GPU."""
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
    raise ValueError("--device debe ser auto, cpu, cuda o un índice de GPU.")


def uses_cuda(device: str | int) -> bool:
    return device != "cpu" and torch.cuda.is_available()


def synchronize(device: str | int) -> None:
    if uses_cuda(device):
        torch.cuda.synchronize(int(device))


def list_test_pairs(
    images_dir: Path,
    labels_dir: Path,
    expected_images: int | None,
) -> list[Path]:
    """Valida que cada imagen tenga una etiqueta YOLO con el mismo nombre."""
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
    max_side: int,
) -> Image.Image:
    """Recorta una caja, respeta los límites y reduce recortes grandes."""
    x1, y1, x2, y2 = box
    width, height = image.size
    bounds = (
        max(0, int(x1) - padding),
        max(0, int(y1) - padding),
        min(width, int(x2) + padding),
        min(height, int(y2) + padding),
    )
    crop = image.crop(bounds).convert("RGB")
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
    """Describe las entradas que deben coincidir al reusar candidatos."""
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
    """Genera una colección común de candidatos o reanuda una compatible."""
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
                "Los candidatos guardados pertenecen a otra configuración. "
                "Use un --run-name nuevo."
            )
        frame = pd.read_csv(csv_path)
        missing = [path for path in frame["crop_path"] if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                "Faltan recortes guardados. Use un --run-name nuevo."
            )
        print(f"Candidatos recuperados: {len(frame)}")
        return frame, summary

    if csv_path.exists() or summary_path.exists():
        raise RuntimeError(
            "La caché de candidatos está incompleta. Use un --run-name nuevo."
        )

    model = YOLO(str(weights))
    if model.task != "detect":
        raise ValueError(f"Se esperaba una tarea detect, no {model.task!r}.")

    rows: list[dict[str, Any]] = []
    synchronize(device)
    wall_start = time.perf_counter()

    for image_index, image_path in enumerate(test_images, start=1):
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        result = model.predict(
            image,
            conf=conf,
            iou=nms_iou,
            imgsz=image_size,
            device=device,
            half=uses_cuda(device),
            verbose=False,
        )[0]

        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        if boxes is not None:
            for detection_index, box in enumerate(boxes):
                candidate_id = f"{image_path.stem}__{detection_index:03d}"
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                crop_path = (crop_dir / f"{candidate_id}.jpg").resolve()
                crop_with_padding(
                    image,
                    xyxy,
                    crop_padding,
                    max_crop_side,
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

    synchronize(device)
    generation_wall_s = time.perf_counter() - wall_start
    del model
    gc.collect()
    if uses_cuda(device):
        torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("YOLO no generó candidatos.")

    fieldnames = list(rows[0])
    save_csv_atomic(csv_path, rows, fieldnames)
    summary = {
        "weights": str(weights.resolve()),
        "images_count": len(test_images),
        "candidates": len(rows),
        "generation_wall_s": generation_wall_s,
        "signature": signature,
    }
    json_dump_atomic(summary_path, summary)
    return pd.DataFrame(rows), summary


def normalize_binary_response(response: str) -> str:
    """Extrae Yes/No ignorando bloques de razonamiento completos o truncados."""
    clean = re.sub(r"<think>.*?</think>", " ", response, flags=re.DOTALL | re.I)
    clean = re.sub(r"<think>.*$", " ", clean, flags=re.DOTALL | re.I)
    clean = re.sub(r"^.*?</think>", " ", clean, flags=re.DOTALL | re.I)
    match = re.search(r"\b(yes|no)\b", clean.lower())
    return match.group(1) if match else "unknown"


def torch_dtype_from_name(name: str, device: str | int) -> torch.dtype:
    """Selecciona un dtype seguro para GPU o CPU."""
    if name == "auto":
        return torch.float16 if uses_cuda(device) else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_vlm(
    model_id: str,
    device: str | int,
    dtype_name: str,
    gpu_memory: str | None,
):
    """Carga un VLM; en GPU permite offload automático si falta VRAM."""
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch_dtype_from_name(dtype_name, device)
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }

    if uses_cuda(device):
        kwargs["device_map"] = "auto"
        if gpu_memory:
            kwargs["max_memory"] = {
                int(device): gpu_memory,
                "cpu": "24GiB",
            }

    model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
    if not uses_cuda(device):
        model.to("cpu")
    model.eval()
    return processor, model


def model_input_device(model) -> torch.device:
    """Obtiene el dispositivo de entrada de un modelo normal o distribuido."""
    device = getattr(model, "device", None)
    if device is not None and device.type != "meta":
        return device
    return next(parameter.device for parameter in model.parameters() if not parameter.is_meta)


def ask_binary(
    processor,
    model,
    image: Image.Image,
    prompt: str,
    device: str | int,
    max_new_tokens: int,
) -> tuple[str, str, float]:
    """Solicita una decisión binaria y mide su latencia sincronizada."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    synchronize(device)
    start = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model_input_device(model))

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000.0
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    response = processor.decode(generated, skip_special_tokens=True).strip()
    decision = normalize_binary_response(response)

    del inputs, output_ids, generated
    if uses_cuda(device):
        torch.cuda.empty_cache()
    return response, decision, latency_ms


def load_validator_records(path: Path, model_id: str) -> list[dict[str, Any]]:
    """Carga una reanudación y verifica que corresponda al mismo modelo."""
    if not path.exists():
        return []
    records = pd.read_csv(path).to_dict("records")
    model_ids = {str(row["model_id"]) for row in records}
    if model_ids and model_ids != {model_id}:
        raise RuntimeError(
            f"{path} pertenece a {sorted(model_ids)}, no a {model_id}. "
            "Use otro --run-name."
        )
    # Una respuesta unknown no se considera terminada y se reintenta.
    return [
        row
        for row in records
        if str(row.get("decision", "")).lower() in {"yes", "no"}
    ]


def run_validator(
    label: str,
    model_id: str,
    processor,
    model,
    candidates: pd.DataFrame,
    validation_dir: Path,
    prompt: str,
    device: str | int,
    max_new_tokens: int,
    limit: int | None = None,
) -> pd.DataFrame:
    """Valida candidatos y guarda un punto de reanudación tras cada recorte."""
    output_path = validation_dir / f"{label}.csv"
    metadata_path = validation_dir / f"{label}_metadata.json"
    candidate_ids = "\n".join(candidates["candidate_id"].astype(str))
    validation_signature = {
        "model_id": model_id,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "max_new_tokens": max_new_tokens,
        "candidate_ids_sha256": hashlib.sha256(
            candidate_ids.encode("utf-8")
        ).hexdigest(),
    }
    if metadata_path.exists():
        saved_signature = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved_signature != validation_signature:
            raise RuntimeError(
                f"La reanudación de {label} usa otra configuración. "
                "Use un --run-name nuevo."
            )
    elif output_path.exists():
        raise RuntimeError(
            f"Falta la metadata de {output_path}. Use un --run-name nuevo."
        )
    else:
        json_dump_atomic(metadata_path, validation_signature)

    records = load_validator_records(output_path, model_id)
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

        raw, decision, latency_ms = ask_binary(
            processor,
            model,
            image,
            prompt,
            device,
            max_new_tokens,
        )
        records.append(
            {
                "candidate_id": row.candidate_id,
                "image_name": row.image_name,
                "confidence": float(row.confidence),
                "model_id": model_id,
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


def release_vlm_memory() -> None:
    """Solicita recolección de memoria después de eliminar el VLM activo."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        best_index: int | None = None

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


def save_pr_curves(
    evaluations: dict[str, dict[str, Any]],
    metrics_dir: Path,
) -> list[Path]:
    """Guarda tres curvas individuales y una comparación conjunta."""
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
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Recall",
            ylabel="Precisión",
        )
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
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Recall",
        ylabel="Precisión",
    )
    axis.set_title("Comparación Precisión–Recall @ IoU 0.5")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    comparison_path = metrics_dir / "pr_comparison.png"
    figure.savefig(comparison_path, dpi=200)
    plt.close(figure)
    paths.append(comparison_path)
    return paths


def compare_pipelines(
    candidates: pd.DataFrame,
    candidate_summary: dict[str, Any],
    validator_results: dict[str, pd.DataFrame],
    ground_truths: dict[str, list[dict[str, Any]]],
    image_count: int,
    eval_iou: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Calcula métricas comunes y latencia extremo a extremo por imagen."""
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
    qwen08_seconds = (
        validator_results["qwen_0_8b"]["latency_ms"].sum() / 1000.0
    )
    qwen2_seconds = (
        validator_results["qwen_2b"]["latency_ms"].sum() / 1000.0
    )
    total_seconds = {
        "YOLOv8s solo": yolo_seconds,
        "YOLOv8s + Qwen 0.8B": yolo_seconds + qwen08_seconds,
        "YOLOv8s + Qwen 2B": yolo_seconds + qwen2_seconds,
    }

    rows: list[dict[str, Any]] = []
    for pipeline_name, evaluation in evaluations.items():
        elapsed = total_seconds[pipeline_name]
        tp = evaluation["tp"]
        fp = evaluation["fp"]
        fn = evaluation["fn"]
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


def validate_completed_results(
    results: pd.DataFrame,
    candidates: pd.DataFrame,
    label: str,
) -> None:
    """Evita comparar una validación parcial o respuestas desconocidas."""
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
            "Revise raw_response antes de comparar."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Pesos ajustados .pt.")
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "data/test/images",
        help="Directorio de imágenes de prueba.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data/test/labels",
        help="Directorio de etiquetas YOLO de referencia.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/phase4_hybrid",
        help="Directorio base de resultados.",
    )
    parser.add_argument(
        "--run-name",
        default="phase4_local",
        help="Nombre estable; permite reanudar una ejecución interrumpida.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, 0, ...")
    parser.add_argument("--candidate-conf", type=float, default=0.15)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--eval-iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--crop-padding", type=int, default=10)
    parser.add_argument("--max-crop-side", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--preflight", type=int, default=5)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Procesa solo el preflight de cada VLM y no calcula métricas.",
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=30,
        help="Cantidad esperada; use 0 para no exigir una cantidad fija.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--gpu-memory",
        help='Límite para device_map, por ejemplo "3GiB" en una GPU de 4 GB.',
    )
    parser.add_argument(
        "--qwen-0-8b",
        default=DEFAULT_MODELS["qwen_0_8b"],
        help="ID o ruta local de Qwen 0.8B.",
    )
    parser.add_argument(
        "--qwen-2b",
        default=DEFAULT_MODELS["qwen_2b"],
        help="ID o ruta local de Qwen 2B.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    expected_images = args.expected_images or None
    weights = (args.weights or discover_weights(ROOT / "models")).resolve()
    test_images = list_test_pairs(args.images, args.labels, expected_images)

    if not 0.0 <= args.candidate_conf <= 1.0:
        raise ValueError("--candidate-conf debe estar entre 0 y 1.")
    if not 0.0 <= args.nms_iou <= 1.0 or not 0.0 <= args.eval_iou <= 1.0:
        raise ValueError("--nms-iou y --eval-iou deben estar entre 0 y 1.")
    if args.preflight < 0 or args.max_new_tokens < 1:
        raise ValueError("--preflight no puede ser negativo y --max-new-tokens debe ser positivo.")
    if args.imgsz < 1 or args.max_crop_side < 1 or args.crop_padding < 0:
        raise ValueError("Las dimensiones deben ser positivas y el padding no negativo.")

    gpu_memory = args.gpu_memory
    if uses_cuda(device) and gpu_memory is None:
        total_gib = torch.cuda.get_device_properties(int(device)).total_memory / 1024**3
        if total_gib < 6.0:
            # Reserva margen para el escritorio, activaciones y caché CUDA.
            gpu_memory = f"{max(1, int(total_gib - 0.75))}GiB"

    run_dir = args.output / args.run_name
    candidate_dir = run_dir / "candidates"
    validation_dir = run_dir / "validators"
    metrics_dir = run_dir / "metrics"
    for directory in (candidate_dir, validation_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_ids = {
        "qwen_0_8b": args.qwen_0_8b,
        "qwen_2b": args.qwen_2b,
    }

    print("Pesos:", weights)
    print("Imágenes:", len(test_images))
    print("Dispositivo:", device)
    if gpu_memory:
        print("Límite de VRAM para Qwen:", gpu_memory)
    print("Resultados:", run_dir)

    candidates, candidate_summary = generate_or_load_candidates(
        weights=weights,
        test_images=test_images,
        candidate_dir=candidate_dir,
        device=device,
        conf=args.candidate_conf,
        nms_iou=args.nms_iou,
        image_size=args.imgsz,
        crop_padding=args.crop_padding,
        max_crop_side=args.max_crop_side,
    )

    validator_results: dict[str, pd.DataFrame] = {}
    for label, model_id in model_ids.items():
        print(f"\nCargando {model_id}...")
        processor, model = load_vlm(
            model_id,
            device,
            args.dtype,
            gpu_memory,
        )
        try:
            results = run_validator(
                label=label,
                model_id=model_id,
                processor=processor,
                model=model,
                candidates=candidates,
                validation_dir=validation_dir,
                prompt=PROMPT_VALIDATION,
                device=device,
                max_new_tokens=args.max_new_tokens,
                limit=args.preflight,
            )
            recent = results.tail(min(args.preflight, len(results)))
            if (recent["decision"] == "unknown").any():
                raise RuntimeError(
                    f"El preflight de {label} produjo respuestas no binarias. "
                    f"Revise {validation_dir / (label + '.csv')}."
                )

            if not args.preflight_only:
                results = run_validator(
                    label=label,
                    model_id=model_id,
                    processor=processor,
                    model=model,
                    candidates=candidates,
                    validation_dir=validation_dir,
                    prompt=PROMPT_VALIDATION,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            validator_results[label] = results
        finally:
            # Elimine las referencias antes de cargar el siguiente modelo.
            del model, processor
            release_vlm_memory()

    if args.preflight_only:
        print("\nPreflight completado. Ejecute nuevamente sin --preflight-only.")
        return

    for label, results in validator_results.items():
        validate_completed_results(results, candidates, label)

    ground_truths = load_ground_truths(test_images, args.labels)
    comparison, evaluations = compare_pipelines(
        candidates,
        candidate_summary,
        validator_results,
        ground_truths,
        len(test_images),
        args.eval_iou,
    )
    plot_paths = save_pr_curves(evaluations, metrics_dir)

    comparison_csv = metrics_dir / "phase4_comparison.csv"
    comparison_json = metrics_dir / "phase4_comparison.json"
    comparison.to_csv(comparison_csv, index=False)
    payload = {
        "session_id": session_id,
        "evaluation_iou": args.eval_iou,
        "candidate_confidence": args.candidate_conf,
        "candidate_count": len(candidates),
        "ground_truth_count": sum(len(v) for v in ground_truths.values()),
        "common_evaluator_results": comparison.to_dict("records"),
    }
    json_dump_atomic(comparison_json, payload)

    manifest = {
        "session_id": session_id,
        "weights": str(weights),
        "weights_sha256": candidate_summary["signature"]["weights_sha256"],
        "models": model_ids,
        "prompt": PROMPT_VALIDATION,
        "candidates": str(candidate_dir / "candidates.csv"),
        "validators": {
            label: str(validation_dir / f"{label}.csv") for label in model_ids
        },
        "validator_metadata": {
            label: str(validation_dir / f"{label}_metadata.json")
            for label in model_ids
        },
        "comparison_csv": str(comparison_csv),
        "comparison_json": str(comparison_json),
        "pr_curves": [str(path) for path in plot_paths],
    }
    manifest_path = run_dir / f"phase4_manifest_{session_id}.json"
    json_dump_atomic(manifest_path, manifest)

    print("\nFASE 4 COMPLETADA")
    print(comparison.to_string(index=False))
    print("Resultados:", run_dir)


if __name__ == "__main__":
    main()
