"""
Orquesta el pipeline completo: rename -> auto_labeling (Fase 1) -> split ->
train_yolo (Fase 2) -> hybrid_inference (Fases 3-4-5, las 3 configuraciones
de la tarea). Se detiene en el primer paso que falle.

FUENTE_ACTIVA elige entre data/raw/ y data/validation/, parchando los
toggles de cada script (ver FUENTES/PASOS_POR_FUENTE) y restaurándolos al
terminar cada corrida. No hace la pausa de revisión manual de
data/labels_check/ que recomienda el README -- toca hacerla aparte.

IMPORTANTE: el paso hybrid_inference.py necesita un modelo YOLO ya
entrenado en config.YOLO_TRAINED (models/). Con FUENTE_ACTIVA="raw" esto lo
genera train_yolo.py como parte del pipeline; con "validation" no hay paso
de entrenamiento, así que ese modelo tiene que existir de antes (main.py lo
valida al inicio, antes de correr nada).

Uso:
    python main.py
"""

import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
CONFIG_PATH = ROOT / "config.py"

sys.path.append(str(ROOT))
import config


# ============================================================
# CONFIGURACIÓN -- lo único que hace falta tocar para correr el pipeline
# ============================================================

# Qué conjunto de imágenes procesar.
FUENTE_ACTIVA = "raw"
# FUENTE_ACTIVA = "validation"


@dataclass
class ConfigHybrid:
    nombre: str
    hybrid_mode: str
    qwen_validador: str | None  # None -> "yolo_only", sin validador


# Configuraciones de hybrid_inference.py a correr. Comentar/descomentar
# líneas para elegir cuáles -- la tarea pide las 3.
CONFIGURACIONES_HYBRID = [
    ConfigHybrid("yolo_only", "yolo_only", None),
    # ConfigHybrid("cascade_08b", "cascade", "0.8b"),
    # ConfigHybrid("cascade_2b", "cascade", "2b"),
]

# Pasos por fuente. "validation" no entrena (data/validation/ es un holdout).
PASOS_POR_FUENTE = {
    "raw": [
        "rename_and_resize_images.py",
        "auto_labeling.py",
        "split_dataset.py",
        "train_yolo.py",
    ],
    "validation": [
        "rename_and_resize_images.py",
        "auto_labeling.py",
    ],
}

# Qué variable parchar en cada script, por fuente.
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


# ============================================================
# MOTOR -- no hace falta tocar nada de aquí para abajo
# ============================================================

def parchar_variables(texto: str, variables: dict[str, str]) -> str:
    """Reemplaza la línea activa (sin '#') de cada 'nombre = valor' -- mismo
    mecanismo que los `sed -i` que se usan a mano en Kaggle."""
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


def encabezado(texto: str) -> None:
    print(f"\n{'=' * 60}\n{texto}\n{'=' * 60}")


def correr(script: str, paso: int, total: int) -> None:
    encabezado(f"[{paso}/{total}] {script}")
    t0 = time.perf_counter()
    resultado = subprocess.run([sys.executable, str(SRC / script)], cwd=ROOT)
    if resultado.returncode != 0:
        sys.exit(f"\n{script} terminó con error (código {resultado.returncode}). Deteniendo el pipeline.")
    print(f"[{paso}/{total}] {script} listo ({time.perf_counter() - t0:.1f}s)")


def correr_hybrid_inference() -> None:
    variables_fuente = FUENTES[FUENTE_ACTIVA].get("hybrid_inference.py", {})
    with parche_temporal(SRC / "hybrid_inference.py", variables_fuente):
        for i, cfg in enumerate(CONFIGURACIONES_HYBRID, start=1):
            encabezado(f"hybrid_inference.py [{i}/{len(CONFIGURACIONES_HYBRID)}] -- {cfg.nombre}")

            variables_config = {"HYBRID_MODE": f'"{cfg.hybrid_mode}"', "RUN_LABEL": "None"}
            if cfg.qwen_validador is not None:
                variables_config["QWEN_VALIDATOR"] = f'QWEN_MODELS["{cfg.qwen_validador}"]'

            t0 = time.perf_counter()
            with parche_temporal(CONFIG_PATH, variables_config):
                resultado = subprocess.run([sys.executable, str(SRC / "hybrid_inference.py")], cwd=ROOT)

            if resultado.returncode != 0:
                sys.exit(f"\nhybrid_inference.py ({cfg.nombre}) terminó con error. Deteniendo el pipeline.")
            print(f"hybrid_inference.py -- {cfg.nombre} listo ({time.perf_counter() - t0:.1f}s)")


def main():
    if FUENTE_ACTIVA not in FUENTES:
        sys.exit(f"FUENTE_ACTIVA={FUENTE_ACTIVA!r} no está en FUENTES -- opciones válidas: {list(FUENTES)}")

    pasos = PASOS_POR_FUENTE[FUENTE_ACTIVA]
    nombres_config = ", ".join(c.nombre for c in CONFIGURACIONES_HYBRID)

    # hybrid_inference.py necesita un modelo YOLO ya entrenado en
    # config.YOLO_TRAINED. Si el pipeline no incluye train_yolo.py (ej.
    # FUENTE_ACTIVA="validation"), ese modelo tiene que existir de antes --
    # se valida aquí, antes de correr nada, para no gastar tiempo en
    # rename/auto_labeling y fallar hasta el último paso.
    if "train_yolo.py" not in pasos and not config.YOLO_TRAINED.exists():
        sys.exit(
            f"No existe {config.YOLO_TRAINED}. Este pipeline (FUENTE_ACTIVA={FUENTE_ACTIVA!r}) "
            f"no entrena un modelo nuevo -- corre train_yolo.py primero (o usa FUENTE_ACTIVA='raw')."
        )

    print("Pipeline husky-detection")
    print(f"  Fuente:          {FUENTE_ACTIVA}")
    print(f"  Pasos:           {' -> '.join(pasos)} -> hybrid_inference.py")
    print(f"  Configuraciones: {nombres_config}")
    print(f"  Modelo YOLO:     {config.YOLO_TRAINED}"
          f"{' (ya existe)' if config.YOLO_TRAINED.exists() else ' (lo genera train_yolo.py)'}")

    t_inicio = time.perf_counter()

    for i, script in enumerate(pasos, start=1):
        with parche_temporal(SRC / script, FUENTES[FUENTE_ACTIVA].get(script, {})):
            correr(script, i, len(pasos))

    correr_hybrid_inference()

    minutos = (time.perf_counter() - t_inicio) / 60
    encabezado(f"Pipeline completo en {minutos:.1f} min")
    print("Revisa results/metrics/, results/figures/ y results/graphs/.")


if __name__ == "__main__":
    main()
