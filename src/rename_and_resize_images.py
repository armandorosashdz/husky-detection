"""
Renombra las imágenes de TARGET_DIR (ver toggle abajo, default data/raw/) al
esquema husky_000.jpg ... husky_NNN.jpg y redimensiona las que excedan
config.MAX_IMAGE_DIM en su lado más grande.

Uso:
    python src/rename_and_resize_images.py

Correr ANTES de auto_labeling.py apuntado a la misma carpeta. Si ya generaste
etiquetas para esa carpeta, renombrar aquí rompe el emparejamiento imagen ↔
.txt (la parte de resize sola sí es segura de re-correr en cualquier momento).
"""

from pathlib import Path
import sys

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
import config

# Carpeta sobre la que actúa: por defecto la real de Fase 1 (data/raw/). Para
# aplicar el mismo renombrado+resize a las 40 imágenes de validación (antes de
# generar sus pseudo-etiquetas con auto_labeling.py -- mismo orden que Fase 1:
# esto primero, etiquetado después), comentar la línea de abajo y descomentar
# la de validación. Igual que en auto_labeling.py, si ya generaste etiquetas
# para esa carpeta, volver a renombrarla rompe el emparejamiento imagen ↔ .txt
# (ver docstring arriba) -- la parte de resize sola sí es segura de re-correr.
TARGET_DIR = config.RAW_DIR
#TARGET_DIR = config.VALIDATION_IMAGES_DIR


def redimensionar_si_necesario(path: Path, max_dim: int) -> bool:
    """Si la imagen excede max_dim en su lado más grande, la reescala manteniendo
    el aspecto (nunca la agranda). Regresa True si se modificó el archivo."""
    with Image.open(path) as img:
        img.load()
        if max(img.size) <= max_dim:
            return False
        img = img.convert("RGB")

    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    img.save(path, quality=90)
    return True


def main():
    carpeta = TARGET_DIR
    if not carpeta.exists():
        sys.exit(f"No existe la carpeta {carpeta}")

    imagenes = sorted(p for p in carpeta.iterdir()
                      if p.is_file() and p.suffix.lower() in config.IMG_EXTENSIONS)

    if not imagenes:
        sys.exit(f"No se encontraron imágenes en {carpeta}")

    print(f"{len(imagenes)} imágenes encontradas en {carpeta}\n")

    # Paso 1: a nombres temporales, para evitar colisiones si algún archivo
    # ya se llama como uno de los destinos (ej. husky_005.jpg existente).
    temporales = []
    for i, p in enumerate(imagenes, start=0):
        destino = carpeta / f"{config.IMG_PREFIX}_{i:03d}{p.suffix.lower()}"
        print(f"  {p.name:<40} → {destino.name}")
        tmp = carpeta / f".tmp_{i:03d}{p.suffix.lower()}"
        p.rename(tmp)
        temporales.append((tmp, destino))

    # Paso 2: de temporales al nombre final
    for tmp, destino in temporales:
        tmp.rename(destino)

    print(f"\nListo. {len(temporales)} imágenes renombradas.")

    # Paso 3: redimensionar las que sean demasiado grandes.
    print(f"\nRevisando tamaños (máximo {config.MAX_IMAGE_DIM}px por lado)...")
    redimensionadas = 0
    for _, destino in temporales:
        if redimensionar_si_necesario(destino, config.MAX_IMAGE_DIM):
            print(f"  {destino.name}: redimensionada")
            redimensionadas += 1

    print(f"\nListo. {redimensionadas} imagen(es) redimensionada(s).")


if __name__ == "__main__":
    main()
