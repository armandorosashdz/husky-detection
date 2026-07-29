"""
Renombra las imágenes de data/raw/ al esquema husky_000.jpg ... husky_099.jpg

Uso:
    python src/rename_images.py

Correr ANTES de auto_labeling.py. Si ya generaste etiquetas, renombrar aquí
rompe el emparejamiento imagen ↔ .txt.
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config


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


if __name__ == "__main__":
    main()