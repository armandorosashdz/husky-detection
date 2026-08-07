"""Fase 5: IoU, matching contra ground truth, mAP@0.5. Funciones puras que
usa hybrid_inference.py para las métricas de cada corrida."""

from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics.utils.metrics import compute_ap as _compute_ap_ultralytics

import config


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection over Union entre dos cajas [x1, y1, x2, y2] en píxeles."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """Lee un .txt YOLO (class_id x_center y_center width height, 0-1) y
    regresa las cajas en píxeles [x1, y1, x2, y2]. Inverso de convert_to_yolo."""
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        _, xc, yc, w, h = line.split()
        xc, yc, w, h = float(xc), float(yc), float(w), float(h)
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def match_detections(
    detecciones: list[dict],
    gt_boxes: list[list[float]],
    iou_threshold: float = config.MAP_IOU_THRESHOLD,
) -> list[dict]:
    """Empareja detecciones de una imagen contra su ground truth: greedy por
    confianza, cada GT se usa una sola vez. Agrega "is_tp" a cada detección."""
    ordenadas = sorted(detecciones, key=lambda d: d["conf"], reverse=True)
    gt_matched = [False] * len(gt_boxes)

    for det in ordenadas:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            v = iou(det["box"], gt)
            if v > best_iou:
                best_iou, best_idx = v, i

        det["is_tp"] = best_iou >= iou_threshold
        if det["is_tp"]:
            gt_matched[best_idx] = True

    return ordenadas


def compute_ap(
    detections: list[dict], n_gt_total: int
) -> tuple[float, list[float], list[float]]:
    """AP@0.5 sobre las detecciones de todas las imágenes (pooled, ya con
    "is_tp"): acumula TP/FP por confianza descendente y delega el área
    (interpolación de 101 puntos estilo COCO) a ultralytics.utils.metrics."""
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []

    for d in detections:
        if d["is_tp"]:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt_total if n_gt_total > 0 else 0)

    if not precisions:
        return 0.0, [], []

    ap, _, _ = _compute_ap_ultralytics(recalls, precisions)
    return float(ap), precisions, recalls


def plot_precision_recall(
    precisions: list[float], recalls: list[float], run_name: str, ap: float
) -> Path:
    """Guarda la curva Precision-Recall en config.GRAPHS_DIR/<run_name>_pr_curve.png."""
    config.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, color="tab:blue", linewidth=2, label=f"mAP@0.5 = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision-Recall — {run_name}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    output_path = config.GRAPHS_DIR / f"{run_name}_pr_curve.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
