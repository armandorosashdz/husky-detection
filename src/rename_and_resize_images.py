"""
Renombra las imágenes de data/raw/ al esquema husky_000.jpg ... husky_099.jpg
y redimensiona las que excedan config.MAX_IMAGE_DIM en su lado más grande.

Uso:
    python src/rename_and_resize_images.py

Correr ANTES de auto_labeling.py. Si ya generaste etiquetas, renombrar aquí
rompe el emparejamiento imagen ↔ .txt.
"""

from pathlib import Path
import sys

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
import config


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
    carpeta = config.RAW_DIR
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
