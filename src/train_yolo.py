"""
Fase 2: Transfer Learning (ajuste fino de YOLOv8s).

Carga YOLOv8s preentrenado (config.YOLO_BASE) y lo afina sobre el dataset definido
en DATASET_YAML_PATH (train/val), usando los hiperparámetros centralizados en
config.py (EPOCHS, IMG_SIZE, BATCH, PATIENCE, SEED, OPTIMIZER, LR0, FREEZE, AUGMENT).

Nota de diseño (ver CLAUDE.md, "Fase 2 design decision"): se usa nc=1 (solo
"husky"), lo que reinicializa la cabeza de detección y pierde las 80 clases de
COCO originales. Fue una decisión deliberada, no un descuido — investigamos la
alternativa (ConcatHead, requiere parchear ultralytics) y no era viable con el
tiempo/hardware disponibles.

Sin argumentos de consola: qué dataset.yaml usar se define aquí abajo (por
defecto el fixture, para probar sin arriesgar tiempo de cómputo real; cambiar al
real ya verificado que el flujo funciona), igual que hicimos en split_dataset.py.

Uso:
    python src/train_yolo.py
"""

from pathlib import Path
import sys

from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
import config

# ---------- Parámetros (editar aquí a mano) ----------
#DATASET_YAML_PATH = config.DATASET_YAML_FIXTURE

# Dataset real, para cuando ya se haya probado con el fixture:
DATASET_YAML_PATH = config.DATASET_YAML


def main():
    if not DATASET_YAML_PATH.exists():
        sys.exit(
            f"No existe {DATASET_YAML_PATH}. Corre split_dataset.py primero "
            f"(con las rutas correspondientes)."
        )

    # Ultralytics guarda los resultados en <project>/<name>/. Derivamos ambos de
    # config.YOLO_TRAINED para que el resultado siempre caiga en la ruta que el
    # resto del pipeline (utils.py) espera, sin ir acumulando train2/, train3/...
    train_dir = config.YOLO_TRAINED.parent.parent
    project_dir = train_dir.parent

    print(f"Cargando {config.YOLO_BASE} (preentrenado)...")
    model = YOLO(config.YOLO_BASE)

    print(f"Entrenando sobre {DATASET_YAML_PATH} ({config.EPOCHS} épocas, en {config.DEVICE})...")
    model.train(
        data=str(DATASET_YAML_PATH),
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH,
        patience=config.PATIENCE,
        seed=config.SEED,
        device=config.DEVICE,
        optimizer=config.OPTIMIZER,
        lr0=config.LR0,
        freeze=config.FREEZE,
        project=str(project_dir),
        name=train_dir.name,
        exist_ok=True,
        **config.AUGMENT,
    )

    print(f"\nListo. Pesos guardados en {config.YOLO_TRAINED}")


if __name__ == "__main__":
    main()
