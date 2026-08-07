"""
Fase 2: ajuste fino de YOLOv8s sobre el dataset de DATASET_YAML_PATH, con
los hiperparámetros de config.py.

Usa nc=1 (solo "husky"), lo que reinicializa la cabeza de detección y
pierde las 80 clases de COCO -- decisión deliberada.

Uso:
    python src/train_yolo.py
"""

import os
from pathlib import Path
import shutil
import sys

from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
import config

#DATASET_YAML_PATH = config.DATASET_YAML_FIXTURE
DATASET_YAML_PATH = config.DATASET_YAML


def main():
    os.chdir(config.ROOT)  # ver nota en split_dataset.py

    if not DATASET_YAML_PATH.exists():
        sys.exit(
            f"No existe {DATASET_YAML_PATH}. Corre split_dataset.py primero "
            f"(con las rutas correspondientes)."
        )

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
        project=str(config.YOLO_RUNS_DIR),
        name=config.YOLO_RUN_NAME,
        exist_ok=True,
        **config.AUGMENT,
    )

    # Copia el mejor checkpoint al modelo "activo" del pipeline (models/, se comitea).
    pesos_entrenados = config.YOLO_RUNS_DIR / config.YOLO_RUN_NAME / "weights" / "best.pt"
    config.YOLO_TRAINED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pesos_entrenados, config.YOLO_TRAINED)

    print(f"\nListo. Pesos copiados a {config.YOLO_TRAINED}")


if __name__ == "__main__":
    main()
