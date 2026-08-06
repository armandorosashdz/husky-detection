"""
Grafica varias curvas Precision-Recall en la misma figura, una por cada
JSON de resultados listado en RUN_JSONS (results/metrics/<archivo>.json,
generados por hybrid_inference.py). Agregar o quitar nombres de esa lista
agrega o quita líneas de la gráfica automáticamente -- si hay 2 nombres
grafica 2 líneas, si hay 5 grafica 5, sin tocar el resto del script.

No es parte del pipeline (hybrid_inference.py no lo llama) -- es una
utilidad de reporte que se corre a mano cuando ya se tienen las corridas
que se quieren comparar en una sola gráfica.

Uso:
    python src/compare_pr_curves.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
import config

# Qué corridas comparar: nombres de archivo dentro de results/metrics/ (con
# o sin ".json"). Sin límite de cantidad -- agregar/quitar líneas aquí es
# lo único que hay que tocar para cambiar cuántas curvas salen.
#RUN_JSONS = [
#    "Yolov8s.json",
#    "Yolov8s + Qwen 0.8B.json",
#    "Yolov8s + Qwen 2.0B.json",
#]
#OUTPUT_NAME = "comparacion_pr_test"

# Diferenciador que se agrega al título de la gráfica: "Precision-Recall
# mAP@0.5 -- <esto>". Sirve para dejar claro, sin abrir el archivo, a qué
# corresponde la comparación (qué holdout, qué corte, etc.).
#DIFERENCIADOR = "Imágenes de test"

# Alternativa: las mismas 3 configuraciones pero sobre el holdout de
# validación (data/validation/, ver CLAUDE.md).
RUN_JSONS = [
    "Yolov8s val.json",
    "Yolov8s + Qwen 0.8B val.json",
    "Yolov8s + Qwen 2.0B val.json",
]
OUTPUT_NAME = "comparacion_pr_validacion"
DIFERENCIADOR = "Imágenes de validación"

# Paleta fija (Okabe-Ito, segura para daltonismo). Los colores se asignan en
# este orden según la posición en RUN_JSONS -- nunca por ciclo aleatorio, así
# la misma corrida mantiene el mismo color si se reordena o compara en otra
# gráfica con las mismas primeras entradas.
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
