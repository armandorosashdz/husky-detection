"""
Orquesta el pipeline completo, en el orden documentado en el README:
renombrar+redimensionar -> auto-etiquetado (Fase 1) -> split train/test
(prep Fase 2) -> entrenar YOLOv8s (Fase 2) -> detección híbrida (Fases
3-4-5), esta última corrida automáticamente para las 3 configuraciones que
pide la tarea (yolo_only, cascade+0.8B, cascade+2B).

FUENTE_ACTIVA (abajo) es el único lugar que hay que tocar para correr todo
esto sobre una carpeta de imágenes distinta a data/raw/ (ej. data/validation/,
para generar sus pseudo-etiquetas) -- automatiza tener que abrir a mano
rename_and_resize_images.py/auto_labeling.py/hybrid_inference.py y
comentar/descomentar sus toggles TARGET_DIR/INPUT_DIR/LABELS_AUTO_OUT/
LABELS_CHECK_OUT/EVAL_DIR uno por uno. Ver FUENTES para qué constante de
cada script controla cada fuente, y PASOS_POR_FUENTE para qué pasos aplican
a cada una (data/validation/ es un holdout, no se usa para split/entrenar).

Cada script se corre exactamente como si se llamara a mano
"python src/<script>.py" -- este orquestador solo parcha temporalmente las
constantes de FUENTES (y, para hybrid_inference.py, HYBRID_MODE/
QWEN_VALIDATOR/RUN_LABEL en config.py) justo antes de cada corrida, y
restaura cada archivo a su contenido original apenas termina esa corrida
-- incluso si algo falla a medias. El resto de los toggles que cada script
ya tenga (el bloque fixture-vs-real de split_dataset.py, DATASET_YAML_PATH
de train_yolo.py, etc.) no se tocan.

Se detiene en el primer paso que falle (código de salida distinto de 0).
No hace pausa para la revisión visual de data/labels_check/ que el README
recomienda entre el etiquetado y el split -- al pedir correr el pipeline
completo sin intervención, esa revisión queda para hacerse aparte,
después, no como un bloqueo a mitad de la corrida.

Sin argumentos de consola, mismo criterio que el resto del proyecto.

Uso:
    python main.py
"""

import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
CONFIG_PATH = ROOT / "config.py"

# Qué conjunto de imágenes procesar. Cambiar aquí para correr todo el
# pipeline (o solo etiquetado + evaluación) sobre una fuente distinta a
# data/raw/, sin editar cada script por separado.
FUENTE_ACTIVA = "raw"
# FUENTE_ACTIVA = "validation"

# Qué variable(s) parchar en cada script, por fuente -- mismo mecanismo que
# los toggles que cada script ya trae comentados/descomentados a mano.
FUENTES = {
    "raw": {
        "rename_and_resize_images.py": {"TARGET_DIR": "config.RAW_DIR"},
        "auto_labeling.py": {
            "INPUT_DIR": "config.RAW_DIR",
            "LABELS_AUTO_OUT": "config.LABELS_AUTO_DIR",
            "LABELS_CHECK_OUT": "config.LABELS_CHECK_DIR",
        },
        "hybrid_inference.py": {"EVAL_DIR": "config.TEST_DIR"},
    },
    "validation": {
        "rename_and_resize_images.py": {"TARGET_DIR": "config.VALIDATION_IMAGES_DIR"},
        "auto_labeling.py": {
            "INPUT_DIR": "config.VALIDATION_IMAGES_DIR",
            "LABELS_AUTO_OUT": "config.VALIDATION_LABELS_DIR",
            "LABELS_CHECK_OUT": "config.VALIDATION_LABELS_CHECK_DIR",
        },
        "hybrid_inference.py": {"EVAL_DIR": "config.VALIDATION_DIR"},
    },
}

# Qué pasos corren para cada fuente -- "validation" no pasa por split ni
# entrenamiento (data/validation/ es un holdout, no se usa para entrenar).
PASOS_POR_FUENTE = {
    "raw": ["rename_and_resize_images.py", "auto_labeling.py", "split_dataset.py", "train_yolo.py"],
    "validation": ["rename_and_resize_images.py", "auto_labeling.py"],
}

# Las 3 configuraciones que pide la tarea para hybrid_inference.py. El
# nombre es el que usaría nombre_corrida() de hybrid_inference.py con
# RUN_LABEL=None (yolo_only/cascade_08b/cascade_2b) -- se fuerza RUN_LABEL
# a None durante estas corridas para garantizar nombres de archivo
# distintos y que no se sobreescriban entre sí.
CONFIGURACIONES_HYBRID = [
    ("yolo_only", "yolo_only", None),
    #("cascade_08b", "cascade", "0.8b"),
    #("cascade_2b", "cascade", "2b"),
]


def parchar_variables(texto: str, variables: dict[str, str]) -> str:
    """Reemplaza, dentro de texto, la línea activa (sin '#' al inicio) de
    cada 'nombre = valor' en variables por 'nombre = valor_nuevo'. Mismo
    mecanismo que los comandos `sed -i` que se usan a mano en Kaggle (ver
    README) -- las alternativas comentadas quedan intactas."""
    for nombre, valor in variables.items():
        texto, n = re.subn(rf'(?m)^{re.escape(nombre)}\s*=\s*.*$', f"{nombre} = {valor}", texto, count=1)
        if n == 0:
            sys.exit(f"No se encontró una línea activa '{nombre} = ...' -- revisa que no esté comentada.")
    return texto


@contextmanager
def parche_temporal(path: Path, variables: dict[str, str]):
    """Parcha `path` con parchar_variables() y restaura su contenido
    original al salir del bloque `with`, incluso si algo falla adentro."""
    if not variables:
        yield
        return
    original = path.read_text(encoding="utf-8")
    try:
        path.write_text(parchar_variables(original, variables), encoding="utf-8")
        yield
    finally:
        path.write_text(original, encoding="utf-8")


def correr(script: str) -> None:
    print(f"\n{'=' * 60}\n{script}\n{'=' * 60}")
    resultado = subprocess.run([sys.executable, str(SRC / script)], cwd=ROOT)
    if resultado.returncode != 0:
        sys.exit(f"\n{script} terminó con error (código {resultado.returncode}). Deteniendo el pipeline.")


def correr_hybrid_inference_3_configs() -> None:
    variables_fuente = FUENTES[FUENTE_ACTIVA].get("hybrid_inference.py", {})
    with parche_temporal(SRC / "hybrid_inference.py", variables_fuente):
        for nombre, hybrid_mode, validador_key in CONFIGURACIONES_HYBRID:
            print(f"\n{'=' * 60}\nhybrid_inference.py -- {nombre}\n{'=' * 60}")

            variables_config = {"HYBRID_MODE": f'"{hybrid_mode}"', "RUN_LABEL": "None"}
            if validador_key is not None:
                variables_config["QWEN_VALIDATOR"] = f'QWEN_MODELS["{validador_key}"]'

            with parche_temporal(CONFIG_PATH, variables_config):
                resultado = subprocess.run([sys.executable, str(SRC / "hybrid_inference.py")], cwd=ROOT)

            if resultado.returncode != 0:
                sys.exit(f"\nhybrid_inference.py ({nombre}) terminó con error. Deteniendo el pipeline.")


def main():
    if FUENTE_ACTIVA not in FUENTES:
        sys.exit(f"FUENTE_ACTIVA={FUENTE_ACTIVA!r} no está en FUENTES -- opciones válidas: {list(FUENTES)}")

    variables_por_script = FUENTES[FUENTE_ACTIVA]

    for script in PASOS_POR_FUENTE[FUENTE_ACTIVA]:
        with parche_temporal(SRC / script, variables_por_script.get(script, {})):
            correr(script)

    correr_hybrid_inference_3_configs()

    print("\nPipeline completo. Revisa results/metrics/, results/figures/ y results/graphs/.")


if __name__ == "__main__":
    main()
