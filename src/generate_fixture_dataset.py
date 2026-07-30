"""
SCRIPT DE PRUEBA DESCARTABLE - no es parte del pipeline final.

Genera un dataset "fixture": imágenes en blanco con cajas aleatorias, más sus
anotaciones YOLO, imitando EXACTAMENTE el formato que produce auto_labeling.py
(usa dibujar_cajas() y convert_to_yolo() reales, no una reimplementación aparte).

Sirve para probar el futuro script de split train/test sin depender de que la
corrida real de Qwen sobre las 100 imágenes haya terminado, y sin ningún riesgo de
mezclar datos falsos con los reales: se genera en carpetas completamente separadas
(data/raw_fixture/, data/labels_auto_fixture/, data/labels_check_fixture/).

PREFIJO por defecto es "husky", pero se puede cambiar aquí abajo para simular
otra clase/proyecto sin tocar nada del pipeline real.

Uso:
    python src/generate_fixture_dataset.py
"""

import random
from pathlib import Path
import sys

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
import config
from auto_labeling import dibujar_cajas
from vlm_utils import convert_to_yolo

# ---------- Parámetros (editar aquí a mano) ----------
PREFIJO = "husky"          # cambiar para simular otra clase/proyecto
N_IMAGENES = 100
ANCHO, ALTO = 640, 480
MIN_CAJAS, MAX_CAJAS = 1, 3
SEED = 123                 # fijo, para que el fixture sea reproducible

# Rutas fixture centralizadas en config.py (no se redefinen aquí) para que
# generate_fixture_dataset.py y split_dataset.py siempre apunten al mismo lugar.
RAW_FIXTURE_DIR = config.RAW_FIXTURE_DIR
LABELS_AUTO_FIXTURE_DIR = config.LABELS_AUTO_FIXTURE_DIR
LABELS_CHECK_FIXTURE_DIR = config.LABELS_CHECK_FIXTURE_DIR


def caja_aleatoria(rng: random.Random) -> list[float]:
    """Genera una caja [x1, y1, x2, y2] aleatoria en escala 0-1000 (mismo formato
    que produce parse_boxes() sobre una respuesta real de Qwen), con ancho/alto
    mínimo de 100 para evitar cajas degeneradas."""
    x1 = rng.uniform(0, 700)
    y1 = rng.uniform(0, 700)
    x2 = x1 + rng.uniform(100, 1000 - x1)
    y2 = y1 + rng.uniform(100, 1000 - y1)
    return [x1, y1, x2, y2]


def imagen_en_blanco(rng: random.Random) -> Image.Image:
    """Imagen "en blanco" con un color de fondo aleatorio, solo para tener algo
    de variedad visual sin depender de fotos reales."""
    color = (rng.randint(180, 255), rng.randint(180, 255), rng.randint(180, 255))
    return Image.new("RGB", (ANCHO, ALTO), color)


def main():
    rng = random.Random(SEED)

    RAW_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_AUTO_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_CHECK_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(N_IMAGENES):
        nombre = f"{PREFIJO}_{i:03d}"
        image = imagen_en_blanco(rng)

        n_cajas = rng.randint(MIN_CAJAS, MAX_CAJAS)
        cajas = [caja_aleatoria(rng) for _ in range(n_cajas)]

        # .txt YOLO, igual que auto_labeling.py -> procesar_imagen()
        lineas_yolo = [convert_to_yolo(caja) for caja in cajas]
        txt_path = LABELS_AUTO_FIXTURE_DIR / f"{nombre}.txt"
        txt_path.write_text("\n".join(lineas_yolo) + "\n")

        # imagen "cruda" (sin cajas dibujadas), como estaría en data/raw/
        image.save(RAW_FIXTURE_DIR / f"{nombre}.jpg")

        # visualización con cajas dibujadas, como en data/labels_check/
        visualizacion = dibujar_cajas(image, cajas)
        visualizacion.save(LABELS_CHECK_FIXTURE_DIR / f"{nombre}.jpg")

        print(f"[{i + 1}/{N_IMAGENES}] {nombre}: {n_cajas} caja(s)")

    print(
        f"\nListo. Generado en:\n"
        f"  {RAW_FIXTURE_DIR}\n"
        f"  {LABELS_AUTO_FIXTURE_DIR}\n"
        f"  {LABELS_CHECK_FIXTURE_DIR}"
    )


if __name__ == "__main__":
    main()
