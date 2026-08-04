"""
Prepara el dataset para entrenamiento: toma pares imagen+label de una carpeta
fuente, los divide en train/test (aleatorio, reproducible) y los copia a la
estructura images/+labels/ que espera Ultralytics, generando también el
dataset.yaml correspondiente.

Paso previo a train_yolo.py (Fase 2), separado a propósito — ver CLAUDE.md.

Sin argumentos de consola: las rutas fuente/destino y el ratio de split se
configuran aquí abajo a mano (por defecto apuntan a las carpetas fixture, para
probar sin tocar los datos reales; cambiar a las rutas reales comentadas cuando
ya se haya verificado que todo funciona).

Uso:
    python src/split_dataset.py
"""

import random
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

# ---------- Parámetros (editar aquí a mano) ----------
#SOURCE_IMAGES_DIR = config.RAW_FIXTURE_DIR
#SOURCE_LABELS_DIR = config.LABELS_AUTO_FIXTURE_DIR
#DEST_TRAIN_DIR = config.TRAIN_FIXTURE_DIR
#DEST_TEST_DIR = config.TEST_FIXTURE_DIR
#DEST_YAML_PATH = config.DATASET_YAML_FIXTURE

# Rutas reales, para cuando ya se haya probado con el fixture:
SOURCE_IMAGES_DIR = config.RAW_DIR
SOURCE_LABELS_DIR = config.LABELS_AUTO_DIR
DEST_TRAIN_DIR = config.TRAIN_DIR
DEST_TEST_DIR = config.TEST_DIR
DEST_YAML_PATH = config.DATASET_YAML

TRAIN_RATIO = config.TRAIN_RATIO
SEED = config.SEED


def listar_pares(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Junta cada imagen de images_dir con su .txt correspondiente en labels_dir
    (mismo nombre base). Ignora (avisando) las imágenes que no tengan label."""
    pares = []
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in config.IMG_EXTENSIONS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"  Aviso: {img_path.name} no tiene .txt en {labels_dir}, se omite.")
            continue
        pares.append((img_path, label_path))
    return pares


def dividir(
    pares: list[tuple[Path, Path]], train_ratio: float, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Mezcla los pares (con semilla fija, reproducible) y los corta en train/test
    según train_ratio."""
    mezclados = pares.copy()
    random.Random(seed).shuffle(mezclados)

    corte = round(len(mezclados) * train_ratio)
    return mezclados[:corte], mezclados[corte:]


def copiar_pares(pares: list[tuple[Path, Path]], dest_dir: Path) -> None:
    """Copia cada imagen a dest_dir/images/ y su label a dest_dir/labels/."""
    images_dir = dest_dir / "images"
    labels_dir = dest_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in pares:
        shutil.copy2(img_path, images_dir / img_path.name)
        shutil.copy2(label_path, labels_dir / label_path.name)


def escribir_dataset_yaml(train_dir: Path, test_dir: Path, yaml_path: Path) -> None:
    """Escribe un dataset.yaml en el formato que espera Ultralytics: path raíz +
    rutas relativas a las imágenes de train/val + el diccionario de clases."""
    train_rel = (train_dir / "images").relative_to(config.ROOT).as_posix()
    test_rel = (test_dir / "images").relative_to(config.ROOT).as_posix()

    contenido = (
        f"path: {config.ROOT.as_posix()}\n"
        f"train: {train_rel}\n"
        f"val: {test_rel}\n"
        f"names:\n"
        f"  {config.CLASS_ID}: {config.CLASS_NAME}\n"
    )
    yaml_path.write_text(contenido)


def verificar_dataset_yaml(yaml_path: Path) -> None:
    """Usa la propia utilidad de Ultralytics para validar que el dataset.yaml
    generado realmente se pueda leer (rutas existen, imágenes con su label, etc.),
    sin necesidad de entrenar nada."""
    from ultralytics.data.utils import check_det_dataset

    try:
        check_det_dataset(str(yaml_path))
        print(f"Verificación OK: {yaml_path} es un dataset.yaml válido para Ultralytics.")
    except Exception as e:
        print(f"Verificación FALLÓ para {yaml_path}: {e}")


def main():
    print(f"Buscando pares imagen+label en {SOURCE_IMAGES_DIR} / {SOURCE_LABELS_DIR}...")
    pares = listar_pares(SOURCE_IMAGES_DIR, SOURCE_LABELS_DIR)
    if not pares:
        sys.exit("No se encontraron pares imagen+label válidos.")

    train_pares, test_pares = dividir(pares, TRAIN_RATIO, SEED)
    print(f"Total: {len(pares)} | train: {len(train_pares)} | test: {len(test_pares)}")

    copiar_pares(train_pares, DEST_TRAIN_DIR)
    copiar_pares(test_pares, DEST_TEST_DIR)
    print(f"Copiado a {DEST_TRAIN_DIR} y {DEST_TEST_DIR}")

    escribir_dataset_yaml(DEST_TRAIN_DIR, DEST_TEST_DIR, DEST_YAML_PATH)
    print(f"dataset.yaml escrito en {DEST_YAML_PATH}")

    verificar_dataset_yaml(DEST_YAML_PATH)


if __name__ == "__main__":
    main()
