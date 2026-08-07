"""
Fase 3 (Deployment) + Fase 4 (Validación Híbrida en Cascada con VLM).

config.HYBRID_MODE elige la fase: "yolo_only" corre solo el detector;
"cascade" además valida cada caja recortada con Qwen (config.QWEN_VALIDATOR).
También calcula las métricas de Fase 5 (mAP@0.5, precision, recall,
latencia, FPS) y guarda el .json + imágenes anotadas + curva P-R.

Para las 3 configuraciones de la tarea, correr 3 veces cambiando
HYBRID_MODE/QWEN_VALIDATOR en config.py. EVAL_DIR (abajo) elige qué holdout
evaluar.

Uso:
    python src/hybrid_inference.py
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).parent.parent))
import config
import metrics
from utils import QwenVLM, YOLODetector

ES_CASCADA = config.HYBRID_MODE == "cascade"

# Cambiar la ruta de donde se van a tomar las imagenes de prueba
EVAL_DIR = config.TEST_DIR
#EVAL_DIR = config.VALIDATION_DIR


@dataclass
class ResultadoImagen:
    detecciones: list[dict]
    n_gt: int
    latencia_ms: float
    descartadas_por_vlm: int


def nombre_corrida() -> tuple[str, str | None]:
    """Nombre de la corrida (para results/*/<esto>) y llave corta del
    validador (ej. "0.8b"). config.RUN_LABEL, si está definido, reemplaza
    el nombre por completo."""
    if not ES_CASCADA:
        run_name, validador_size = "yolo_only", None
    else:
        run_name, validador_size = "cascade_validador", "validador"
        for key, model_id in config.QWEN_MODELS.items():
            if model_id == config.QWEN_VALIDATOR:
                run_name, validador_size = f"cascade_{key.replace('.', '')}", key
                break

    if config.RUN_LABEL:
        run_name = config.RUN_LABEL
    return run_name, validador_size


def listar_imagenes_eval() -> list[Path]:
    images_dir = EVAL_DIR / "images"
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMG_EXTENSIONS
    )
    if not image_paths:
        sys.exit(f"No se encontraron imágenes en {images_dir}. Corre split_dataset.py (si EVAL_DIR=TEST_DIR) o auto_labeling.py (si EVAL_DIR=VALIDATION_DIR) primero.")
    return image_paths


def filtrar_con_cascada(
    detecciones: list[dict], image: Image.Image, detector: YOLODetector, validador: QwenVLM
) -> tuple[list[dict], int]:
    """Le pregunta a Qwen, por cada detección, si de verdad es un husky."""
    conservadas = []
    descartadas = 0
    for det in detecciones:
        crop = detector.crop(image, det["box"])
        if crop.width < 10 or crop.height < 10:
            descartadas += 1
            continue

        respuesta = validador.ask(
            crop, config.PROMPT_VALIDATION, max_new_tokens=config.VALIDATION_MAX_NEW_TOKENS
        )
        if respuesta.strip().lower().startswith("yes"):
            conservadas.append(det)
        else:
            descartadas += 1

    return conservadas, descartadas


def deduplicate_boxes(
    detecciones: list[dict], iou_threshold: float = config.DEDUP_IOU_THRESHOLD
) -> list[dict]:
    """Segunda pasada tipo NMS, después de la cascada -- el validador VLM no
    detecta cajas duplicadas sobre el mismo perro por sí solo."""
    ordenadas = sorted(detecciones, key=lambda d: d["conf"], reverse=True)
    conservadas = []
    for det in ordenadas:
        es_duplicada = any(
            metrics.iou(det["box"], c["box"]) >= iou_threshold for c in conservadas
        )
        if not es_duplicada:
            conservadas.append(det)
    return conservadas


def dibujar_detecciones(image: Image.Image, detecciones: list[dict]) -> Image.Image:
    """Verde si hizo match con ground truth (TP), naranja si no (FP)."""
    salida = image.copy()
    draw = ImageDraw.Draw(salida)
    for det in detecciones:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = "lime" if det.get("is_tp") else "orange"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1, max(y1 - 12, 0)), f"{det['conf']:.2f}", fill=color)
    return salida


def procesar_imagen(
    detector: YOLODetector, validador: QwenVLM, img_path: Path, labels_dir: Path
) -> ResultadoImagen:
    image = Image.open(img_path).convert("RGB")
    gt_boxes = metrics.load_ground_truth(labels_dir / f"{img_path.stem}.txt", *image.size)

    t0 = time.perf_counter()
    detecciones = detector.detect(image)

    descartadas_por_vlm = 0
    if ES_CASCADA:
        detecciones, descartadas_por_vlm = filtrar_con_cascada(detecciones, image, detector, validador)

    latencia_ms = (time.perf_counter() - t0) * 1000

    detecciones = deduplicate_boxes(detecciones)
    detecciones = metrics.match_detections(detecciones, gt_boxes)

    return ResultadoImagen(detecciones, len(gt_boxes), latencia_ms, descartadas_por_vlm)


def calcular_metricas_finales(
    all_detections: list[dict], n_gt_total: int, total_latency_ms: float, n_imagenes: int
) -> dict:
    ap, precisions, recalls = metrics.compute_ap(all_detections, n_gt_total)

    tp_total = sum(1 for d in all_detections if d["is_tp"])
    fp_total = sum(1 for d in all_detections if not d["is_tp"])
    fn_total = n_gt_total - tp_total
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / n_gt_total if n_gt_total else 0.0

    avg_latency = total_latency_ms / n_imagenes
    fps = 1000 / avg_latency if avg_latency > 0 else 0.0

    return {
        "map50": ap,
        "precision": precision,
        "recall": recall,
        "false_positives": fp_total,
        "false_negatives": fn_total,
        "avg_latency_ms": avg_latency,
        "fps": fps,
        "precisions": precisions,
        "recalls": recalls,
    }


def main():
    run_name, validador_size = nombre_corrida()
    figures_dir = config.FIGURES_DIR / run_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not config.YOLO_TRAINED.exists():
        sys.exit(f"No existe {config.YOLO_TRAINED}. Corre train_yolo.py primero.")

    print(f"Cargando detector: {config.YOLO_TRAINED}")
    detector = YOLODetector(config.YOLO_TRAINED).load()

    validador = None
    if ES_CASCADA:
        print(f"Cargando validador: {config.QWEN_VALIDATOR}")
        validador = QwenVLM(config.QWEN_VALIDATOR).load()

    image_paths = listar_imagenes_eval()
    labels_dir = EVAL_DIR / "labels"
    print(f"Evaluando '{run_name}' sobre {len(image_paths)} imagen(es) de {EVAL_DIR.name}...")

    detector.detect(Image.open(image_paths[0]).convert("RGB"))  # warm-up, no cuenta en latencia

    all_detections: list[dict] = []
    n_gt_total = 0
    total_latency_ms = 0.0
    total_discarded_by_vlm = 0

    for img_path in image_paths:
        resultado = procesar_imagen(detector, validador, img_path, labels_dir)

        n_gt_total += resultado.n_gt
        total_latency_ms += resultado.latencia_ms
        total_discarded_by_vlm += resultado.descartadas_por_vlm
        all_detections.extend(resultado.detecciones)

        imagen_original = Image.open(img_path).convert("RGB")
        dibujar_detecciones(imagen_original, resultado.detecciones).save(figures_dir / img_path.name)

        print(f"{img_path.name}: {len(resultado.detecciones)} conservada(s)")

    metricas = calcular_metricas_finales(all_detections, n_gt_total, total_latency_ms, len(image_paths))

    print(f"\n==== Resultados: {run_name} ====")
    print(f"mAP@0.5:            {metricas['map50']:.4f}")
    print(f"Precision:          {metricas['precision']:.4f}")
    print(f"Recall:             {metricas['recall']:.4f}")
    print(f"Falsos positivos:   {metricas['false_positives']}")
    print(f"Falsos negativos:   {metricas['false_negatives']}")
    if ES_CASCADA:
        print(f"Cajas descartadas por el VLM: {total_discarded_by_vlm}")
    print(f"Latencia promedio:  {metricas['avg_latency_ms']:.2f} ms")
    print(f"FPS reales:         {metricas['fps']:.2f}")

    resultados = {
        "run_name": run_name,
        "mode": config.HYBRID_MODE,
        "qwen_model": validador_size,
        "discarded_by_vlm": total_discarded_by_vlm if ES_CASCADA else None,
        "n_gt_total": n_gt_total,
        **metricas,
    }
    resultados_path = config.METRICS_DIR / f"{run_name}.json"
    resultados_path.write_text(json.dumps(resultados, indent=2))
    grafica_path = metrics.plot_precision_recall(
        metricas["precisions"], metricas["recalls"], run_name, metricas["map50"]
    )
    print(f"\nResultados guardados en: {resultados_path}")
    print(f"Imágenes anotadas en: {figures_dir}/")
    print(f"Curva Precision-Recall guardada en: {grafica_path}")


if __name__ == "__main__":
    main()
