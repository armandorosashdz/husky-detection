"""
SCRIPT DE PRUEBA DESCARTABLE - no es parte del pipeline final.

Ya se confirmó visualmente que Qwen3.5 devuelve las cajas como [x1,y1,x2,y2]
(ver commits/conversación anterior). Este script ahora verifica de punta a punta
que convert_to_yolo() hace bien la conversión: dibuja, sobre varias imágenes, el
bounding box reconstruido a partir de la línea YOLO generada (en vez de las
coordenadas crudas de Qwen), para confirmar que el round-trip caja -> .txt YOLO
-> caja en píxeles da un rectángulo correcto.

Guarda los resultados en data/labels_check/ (prefijo test_orden_), sobrescribiendo
las imágenes de la prueba anterior.

Uso:
    python src/test_box_order.py
"""

from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).parent.parent))
import config
from vlm_utils import QwenVLM, convert_to_yolo, parse_boxes

IMAGENES_PRUEBA = ["husky_000.jpg", "husky_001.jpg", "husky_002.jpg"]


def yolo_a_pixeles(linea_yolo, ancho, alto):
    """Convierte una línea YOLO ("class_id x_center y_center width height",
    normalizada 0-1) de vuelta a un rectángulo en píxeles (x1,y1,x2,y2), para
    poder dibujarla y verificar visualmente que convert_to_yolo no rompió nada."""
    _, x_center, y_center, width, height = linea_yolo.split()
    x_center, y_center, width, height = map(float, (x_center, y_center, width, height))

    x1 = (x_center - width / 2) * ancho
    y1 = (y_center - height / 2) * alto
    x2 = (x_center + width / 2) * ancho
    y2 = (y_center + height / 2) * alto
    return (x1, y1, x2, y2)


def main():
    modelo = config.QWEN_VALIDATORS["0.8b"]
    print(f"Cargando modelo {modelo}...")
    vlm = QwenVLM(modelo).load()

    config.LABELS_CHECK_DIR.mkdir(parents=True, exist_ok=True)

    for nombre in IMAGENES_PRUEBA:
        imagen_path = config.RAW_DIR / nombre
        if not imagen_path.exists():
            print(f"\n{nombre}: no existe, se omite.")
            continue

        image = Image.open(imagen_path).convert("RGB")
        ancho, alto = image.size
        print(f"\nImagen: {nombre} ({ancho}x{alto})")

        respuesta = vlm.ask(image, config.PROMPT_LABELING)
        cajas = parse_boxes(respuesta)
        print(f"Cajas parseadas: {cajas}")
        if not cajas:
            print("No se parseó ninguna caja, se omite el dibujo.")
            continue

        salida = image.copy()
        draw = ImageDraw.Draw(salida)

        for caja in cajas:
            linea_yolo = convert_to_yolo(caja)
            print(f"  {caja} -> {linea_yolo}")
            rect = yolo_a_pixeles(linea_yolo, ancho, alto)
            draw.rectangle(rect, outline="lime", width=4)

        salida_path = config.LABELS_CHECK_DIR / f"test_orden_{nombre}"
        salida.save(salida_path)
        print(f"Guardada en: {salida_path}")


if __name__ == "__main__":
    main()
