"""
Grafica varias curvas Precision-Recall en una sola figura, una por cada
JSON en RUN_JSONS (results/metrics/<archivo>.json). Utilidad de reporte,
no la llama hybrid_inference.py.

Uso:
    python src/compare_pr_curves.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
import config

# Archivos a comparar (con o sin ".json"). Agregar/quitar líneas aquí cambia
# cuántas curvas salen, sin tocar el resto del script.
#RUN_JSONS = [
#    "Yolov8s.json",
#    "Yolov8s + Qwen 0.8B.json",
#    "Yolov8s + Qwen 2.0B.json",
#]
#OUTPUT_NAME = "comparacion_pr_test"
#DIFERENCIADOR = "Imágenes de test"

RUN_JSONS = [
    "Yolov8s val.json",
    "Yolov8s + Qwen 0.8B val.json",
    "Yolov8s + Qwen 2.0B val.json",
]
OUTPUT_NAME = "comparacion_pr_validacion"
DIFERENCIADOR = "Imágenes de validación"

# Paleta fija (Okabe-Ito, segura para daltonismo), asignada por posición en
# RUN_JSONS -- no por ciclo, para que el color de una corrida no cambie.
PALETA = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"]


def cargar_corrida(nombre: str) -> dict:
    nombre = nombre if nombre.endswith(".json") else f"{nombre}.json"
    path = config.METRICS_DIR / nombre
    if not path.exists():
        sys.exit(f"No existe {path}. Corre hybrid_inference.py para esa configuración primero.")
    return json.loads(path.read_text())


def main():
    if not RUN_JSONS:
        sys.exit("RUN_JSONS está vacío -- agrega al menos un nombre de archivo.")
    if len(RUN_JSONS) > len(PALETA):
        sys.exit(
            f"RUN_JSONS tiene {len(RUN_JSONS)} corridas pero PALETA solo tiene "
            f"{len(PALETA)} colores -- agrega más colores a PALETA."
        )

    fig, ax = plt.subplots(figsize=(7, 6))

    for color, nombre in zip(PALETA, RUN_JSONS):
        datos = cargar_corrida(nombre)
        etiqueta = f"{datos['run_name']} (mAP={datos['map50']:.3f})"
        ax.plot(datos["recalls"], datos["precisions"], color=color, linewidth=2, label=etiqueta)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision-Recall mAP@0.5 — {DIFERENCIADOR}")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    config.GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.GRAPHS_DIR / f"{OUTPUT_NAME}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado en {output_path}")


if __name__ == "__main__":
    main()
