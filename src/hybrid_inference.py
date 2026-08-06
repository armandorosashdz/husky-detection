"""
Fase 3 (Deployment) + Fase 4 (Validación Híbrida en Cascada con VLM).

Un mismo script cubre las dos fases según config.HYBRID_MODE:
  "yolo_only" -> Fase 3: solo YOLOv8s procesando el flujo de imágenes de
                 test, midiendo latencia y FPS reales.
  "cascade"   -> Fase 4: además, cada detección se recorta y se manda a Qwen
                 VL (config.QWEN_VALIDATOR) con una pregunta binaria de
                 confirmación; solo las cajas aprobadas se conservan.

También calcula, para la corrida actual, las métricas de la Fase 5 (mAP@0.5,
precision, recall, falsos positivos/negativos, latencia, FPS) usando
metrics.py, y guarda el .json + las imágenes anotadas — lo necesario para la
tabla comparativa y las curvas Precision-Recall del reporte.

Sin argumentos de consola: para comparar las 3 configuraciones que pide la
tarea, corre este script 3 veces cambiando HYBRID_MODE/QWEN_VALIDATOR en
config.py:
    1) HYBRID_MODE = "yolo_only"                                (línea base)
    2) HYBRID_MODE = "cascade", QWEN_VALIDATOR = QWEN_MODELS["0.8b"]
    3) HYBRID_MODE = "cascade", QWEN_VALIDATOR = QWEN_MODELS["2b"]

EVAL_DIR (variable de este archivo, no de config.py) elige qué carpeta
evaluar: config.TEST_DIR (default, 30 imágenes -- usadas también como val
durante el entrenamiento) o config.VALIDATION_DIR (40 imágenes que el
entrenamiento nunca vio, un holdout más limpio -- requiere haber corrido antes
auto_labeling.py apuntado a validación para generar sus pseudo-etiquetas).

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

# Se lee una sola vez de config.py; evita comparar el string "cascade" en
# cinco lugares distintos del archivo.
ES_CASCADA = config.HYBRID_MODE == "cascade"

# Carpeta a evaluar: por defecto el test set real (data/test/), que Ultralytics
# ya usó como val durante train_yolo.py para elegir el mejor checkpoint -- no
# es un holdout 100% limpio (ver nota en CLAUDE.md). Para evaluar contra las
# 40 imágenes de validación que el entrenamiento nunca vio, comentar la línea
# de abajo y descomentar la de validación -- requiere haber generado antes sus
# pseudo-etiquetas con auto_labeling.py (ver su propio toggle INPUT_DIR).
# Se espera el mismo layout images/+labels/ en ambos casos.
#EVAL_DIR = config.TEST_DIR
EVAL_DIR = config.VALIDATION_DIR


@dataclass
class ResultadoImagen:
    """Lo que procesar_imagen() calcula para UNA imagen. Con nombre en vez de
    tupla posicional para no tener que recordar el orden en cada llamada."""
    detecciones: list[dict]
    n_gt: int
    latencia_ms: float
    descartadas_por_vlm: int


def nombre_corrida() -> tuple[str, str | None]:
    """Nombre de la corrida actual (para results/metrics/<esto>.json y
    results/figures/<esto>/) y la llave corta del validador (ej. "0.8b", para
    guardarla en el .json) -- ambos derivados de config.HYBRID_MODE/QWEN_VALIDATOR.
    En modo yolo_only no hay validador, así que ese segundo valor es None."""
    if not ES_CASCADA:
        run_name, validador_size = "yolo_only", None
    else:
        run_name, validador_size = "cascade_validador", "validador"  # fallback si QWEN_VALIDATOR no matchea ningún tamaño conocido
        for key, model_id in config.QWEN_MODELS.items():
            if model_id == config.QWEN_VALIDATOR:
                run_name, validador_size = f"cascade_{key.replace('.', '')}", key
                break

    # config.RUN_LABEL reemplaza el nombre por completo si está definido --
    # mode/qwen_model no dependen de run_name, así que esto es seguro incluso
    # con un nombre arbitrario.
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
    """Fase 4: le pregunta a Qwen, por cada detección, si de verdad es un husky
    (recorte + Yes/No). Regresa (detecciones_aprobadas, cantidad_descartada)."""
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
    """Elimina cajas redundantes que sobreviven la cascada pero se superponen
    demasiado con otra de mayor confianza sobre el mismo perro. El validador
    VLM no puede detectar esto por sí solo (contesta "sí es husky", no "ya hay
    otra caja sobre este mismo husky") — se resuelve aquí con una segunda
    pasada tipo NMS, después de la cascada."""
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
    """Dibuja las detecciones finales sobre una copia de la imagen: verde si
    hizo match con ground truth (TP), naranja si no (FP)."""
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
    """Corre detección (+ cascada, si aplica) sobre una imagen y la empareja
    contra su ground truth."""
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
    """Agrega las detecciones de TODAS las imágenes ya procesadas en las
    métricas de la corrida completa: mAP@0.5, precision, recall, latencia, FPS."""
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

    # Warm-up: la primera inferencia siempre es más lenta (inicialización de
    # CUDA/cuDNN); no se cuenta en las métricas de latencia/FPS.
    detector.detect(Image.open(image_paths[0]).convert("RGB"))

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
