# husky-detection

## Cómo correr el pipeline desde cero

Instrucciones para alguien que solo tiene las imágenes sin anotar en `data/raw/` y quiere llegar a los 3 modelos evaluados. Ningún script recibe argumentos de consola — lo que cambia entre corridas (qué modelo de Qwen usar, qué carpeta procesar, etc.) es una variable al inicio del script o en `config.py`, que se edita a mano antes de correr. Antes de nada:

```bash
pip install -r requirements.txt
```
(en una máquina sin GPU dedicada, correr en su lugar con `conda run -n tarea3 python src/<script>.py` o el entorno conda equivalente).

**En Kaggle específicamente, `pip install -r requirements.txt` no basta**: Kaggle ya trae `transformers`/`accelerate`/`ultralytics` preinstalados pero desactualizados — como esos paquetes ya están instalados, `pip install` normal los deja tal cual (no los actualiza) y `transformers` termina siendo muy viejo para reconocer la arquitectura `qwen3_5` de Qwen3.5 (`KeyError: 'qwen3_5'`). Hace falta forzar la actualización explícitamente:
```bash
!pip install -U transformers accelerate ultralytics -q
```
y reiniciar el kernel del notebook después (Kaggle no recarga los paquetes actualizados en la sesión activa). También ajustar `config.DEVICE`/`DTYPE` a `"cuda"`/`"float16"` antes de correr para aprovechar la GPU.

**Atajo: correr todo de una vez**
```bash
python main.py
```
Ejecuta en orden los pasos 1, 2, 4 y 5 de abajo, y para el paso 6 corre automáticamente las configuraciones activas en `CONFIGURACIONES_HYBRID` (por defecto solo `yolo_only`; descomentar las de cascada para las 3 que pide la tarea), editando `HYBRID_MODE`/`QWEN_VALIDATOR`/`RUN_LABEL` en `config.py` entre cada una y dejando `config.py` exactamente como estaba al terminar. `FUENTE_ACTIVA` (variable al inicio de `main.py`, `"raw"` por defecto, alternativa `"validation"`) es el único lugar que hay que tocar para correr todo esto sobre `data/validation/` en vez de `data/raw/` — automáticamente parcha `TARGET_DIR`/`INPUT_DIR`/`LABELS_AUTO_OUT`/`LABELS_CHECK_OUT`/`EVAL_DIR` en los scripts correspondientes (y con `"validation"` se saltan los pasos 4 y 5, que no aplican a un holdout). Imprime un resumen al inicio (fuente, pasos, configuraciones, y si ya existe un modelo YOLO entrenado) y el tiempo de cada paso. Se detiene en el primer paso que falle — incluyendo, antes de correr nada, si el paso 6 necesita un modelo ya entrenado (`config.YOLO_TRAINED`) que no existe y el pipeline no lo va a generar (por ejemplo con `FUENTE_ACTIVA="validation"`, que no entrena). No hace la pausa de revisión visual del paso 3 — si se corre así, esa revisión hay que hacerla aparte. El resto de esta sección explica cada paso por separado, útil para correrlos uno por uno o para entender qué hace `main.py` internamente.

**1. Renombrar y redimensionar las imágenes crudas**
```bash
python src/rename_and_resize_images.py
```
Toma todo lo que haya en `TARGET_DIR` (variable al inicio del script, por defecto `data/raw/`) y lo renombra al esquema `husky_000.jpg, husky_001.jpg, ...`, luego achica (nunca agranda) cualquier imagen que exceda `config.MAX_IMAGE_DIM` en su lado más grande — necesario porque algunas fotos muy grandes hacen que Qwen intente reservar demasiada memoria de golpe en el siguiente paso. Se corre **una sola vez, antes de etiquetar** — si ya generaste etiquetas y vuelves a renombrar, se rompe la correspondencia imagen↔`.txt`.

**2. Auto-etiquetar con Qwen (Fase 1)**
```bash
python src/auto_labeling.py
```
Le manda cada imagen de `INPUT_DIR` (por defecto `data/raw/`) a Qwen (`config.QWEN_LABELER`) pidiéndole las cajas de los huskies, convierte la respuesta a formato YOLO y escribe un `.txt` por imagen en `data/labels_auto/`, además de una versión con las cajas dibujadas en `data/labels_check/` para revisar a simple vista que quedaron bien. Si `config.VERIFY_BREED` está activo, cada caja detectada se recorta y se le hace una segunda consulta a Qwen (`config.PROMPT_VERIFY_BREED`) preguntando si de verdad es un Husky Siberiano — las que no se confirman se descartan antes de escribirse (funciona igual sin importar el tamaño de `QWEN_LABELER`, ya que reutiliza el mismo modelo ya cargado). `config.AUTO_LABELING_LIMIT` deja probar con pocas imágenes antes de correr todas. Este paso es el más pesado en RAM/VRAM — sin GPU conviene usar el modelo `"0.8b"` (ya configurado por defecto); con GPU real, cambiar `config.QWEN_LABELER` a `"2b"`/`"4b"` y `config.DEVICE`/`DTYPE` a `"cuda"`/`"float16"` (ambos cambios juntos, o intentará cargar un modelo grande en float32 sobre CPU y falla).

**3. Revisar visualmente `data/labels_check/`** y confirmar que las cajas se ven bien antes de seguir — si Qwen se equivocó mucho en algunas imágenes, hay que corregir el `.txt` correspondiente a mano antes del siguiente paso.

**4. Preparar el split de entrenamiento**
```bash
python src/split_dataset.py
```
Revisa primero qué bloque de rutas está activo al inicio del archivo (hay un bloque real y uno fixture, comentados/descomentados entre sí). Toma `data/raw/` + `data/labels_auto/`, separa 70%/30% en `data/train/`/`data/test/` (semilla fija, mismo split siempre) y genera `dataset.yaml`, el archivo que Ultralytics necesita para entrenar.

**5. Entrenar YOLOv8s (Fase 2)**
```bash
python src/train_yolo.py
```
Revisa `DATASET_YAML_PATH` al inicio del archivo (real vs. fixture) y `config.EPOCHS` (100 para una corrida real). Descarga `yolov8s.pt` preentrenado si hace falta, ajusta la cabeza a una sola clase ("husky") y entrena con los hiperparámetros de `config.py`. Al terminar, copia el mejor checkpoint a `models/<config.YOLO_TRAINED_NAME>` (`config.YOLO_TRAINED`) — el modelo que usa todo lo demás (`YOLODetector`, `hybrid_inference.py`, `main.py`) — y deja las curvas de entrenamiento en `runs/detect/train/`. `YOLO_TRAINED_NAME` es el único valor que hace falta cambiar para guardar con otro nombre (sin pisar el de otra persona) o para apuntar a un modelo ya entrenado por alguien más sin reentrenar. Este paso tarda; en GPU (Kaggle/Colab) es lo más práctico.

**6. Correr la detección híbrida (Fases 3-4-5), una vez por configuración**
```bash
python src/hybrid_inference.py
```
Antes de cada corrida, ajustar en `config.py`:
- `HYBRID_MODE = "yolo_only"` (solo el detector) o `"cascade"` (+ validación con Qwen).
- `QWEN_VALIDATOR` (si es cascada): `QWEN_MODELS["0.8b"]` o `["2b"]`.
- `RUN_LABEL` opcional, para nombrar la corrida a mano en vez del nombre genérico.

Y opcionalmente, `EVAL_DIR` al inicio del script (`config.TEST_DIR` por defecto, o `config.VALIDATION_DIR` si ya se etiquetó ese holdout — ver más abajo). Cada corrida calcula mAP@0.5/precision/recall/latencia/FPS y guarda: el JSON en `results/metrics/`, las imágenes anotadas en `results/figures/`, y la curva Precision-Recall en `results/graphs/`. Para tener las 3 configuraciones que pide la tarea, se corre 3 veces cambiando `HYBRID_MODE`/`QWEN_VALIDATOR` entre corrida y corrida.

**7. (Opcional) Evaluar también contra un holdout más limpio**
`data/test/` fue usado por Ultralytics como validación durante el entrenamiento (paso 5), así que sus métricas están un poco optimistas. Si se quiere un número más confiable: conseguir imágenes nuevas nunca vistas por el entrenamiento, correr sobre ellas los pasos 1 y 2 (con `TARGET_DIR`/`INPUT_DIR` apuntando a esa carpeta en vez de `data/raw/`) para generar pseudo-etiquetas, y luego el paso 6 con `EVAL_DIR = config.VALIDATION_DIR`.

**8. (Opcional) Graficar varias corridas juntas para el reporte**
```bash
python src/compare_pr_curves.py
```
Lee la lista `RUN_JSONS` (nombres de archivo en `results/metrics/`, editable a mano al inicio del script) y dibuja todas esas curvas Precision-Recall en una sola figura, en `results/graphs/`.

## Estructura del repositorio

```text
husky-detection/
├── README.md                    # setup, cómo correr, quién hace qué
├── main.py                      # corre el pipeline completo de un jalón (ver sección de arriba)
├── Especificaciones de Tarea 4...pdf   # enunciado original de la tarea
├── requirements.txt              # torch, torchvision, transformers, accelerate, ultralytics, pillow, matplotlib (`pip install -r requirements.txt`)
├── .gitignore                   # __pycache__, *.pt (excepto models/*.pt), data/*_fixture/, dataset_fixture.yaml
├── config.py                    # model_ids, prompts, thresholds, rutas, límites — todo lo configurable vive aquí
├── src/
│   ├── utils.py                 # QwenVLM + parse_boxes + convert_to_yolo + YOLODetector
│   ├── rename_and_resize_images.py  # renombra+redimensiona TARGET_DIR (implementado)
│   ├── auto_labeling.py         # Fase 1: INPUT_DIR → LABELS_AUTO_OUT + LABELS_CHECK_OUT, VERIFY_BREED opcional
│   ├── split_dataset.py         # prepara train/test (70/30) + dataset.yaml, previo a Fase 2 (implementado)
│   ├── train_yolo.py            # Fase 2: ajuste fino de YOLOv8s (implementado, corrido para real)
│   ├── hybrid_inference.py      # Fases 3-4: YOLO + cascada Qwen, EVAL_DIR configurable (implementado)
│   ├── metrics.py               # Fase 5: IoU, mAP@0.5, curva P-R (implementado, lo usa hybrid_inference.py)
│   ├── compare_pr_curves.py     # utilidad de reporte: combina N curvas P-R de results/metrics/ en una figura
│   ├── generate_fixture_dataset.py  # script de prueba descartable: genera dataset falso
│   └── test_box_order.py        # script de prueba descartable, no es parte del pipeline
├── dataset.yaml                 # generado por split_dataset.py (ya corrido con datos reales)
├── yolov8s.pt                   # pesos preentrenados, descargados por Ultralytics al entrenar (gitignored)
├── runs/detect/train/           # resultados del entrenamiento real: curvas, matriz de confusión, results.csv
│   └── weights/                 # best.pt/last.pt NO se comitean (ver weights/README.txt) -- duplican models/
├── models/                      # modelos finales del equipo, sí se comitean (excepción a *.pt)
│   ├── yolov8_finetuned_armando.pt   # = config.YOLO_TRAINED (nombre configurable: YOLO_TRAINED_NAME)
│   └── yolov8_finetuned_pedro.pt     # modelo de comparación; cambiar YOLO_TRAINED_NAME para usarlo directo
├── data/
│   ├── raw/                     # 100 imágenes sin anotar (crudas, redimensionadas) + Dataset2.zip (respaldo)
│   ├── labels_auto/             # .txt generados por Qwen (Fase 1 completa: 100/100)
│   ├── labels_check/            # visualizaciones con BB dibujadas (Fase 1 completa: 100/100)
│   ├── train/                   # 70 imágenes+labels (split real ya generado)
│   ├── test/                    # 30 imágenes+labels (split real ya generado) -- usado también como val
│   │                             # de Ultralytics durante el entrenamiento (ver nota en la sección de abajo)
│   └── validation/              # 40 imágenes, holdout limpio (nunca visto por el entrenamiento)
│       ├── images/               # husky_000..039.jpeg (renombradas+redimensionadas)
│       ├── labels/               # pseudo-ground-truth generado con auto_labeling.py (Qwen 4B, Kaggle)
│       └── labels_check/         # visualizaciones para QA manual
├── results/                      # comitido: 6 corridas (yolo_only/cascade_08b/cascade_2b × test/validation)
│   ├── metrics/                 # un JSON por configuración (hybrid_inference.py)
│   ├── figures/                 # imágenes anotadas por corrida (verde=TP/naranja=FP)
│   └── graphs/                  # curva Precision-Recall por corrida (metrics.plot_precision_recall)
└── report/
    ├── figures/                 # figuras para el reporte (ej. comparación cascada 0.8B vs 2B en validación)
    ├── metrics/
    └── reporte.pdf
```

Nota: `data/*_fixture/` y `dataset_fixture.yaml` (dataset falso de `generate_fixture_dataset.py`/`split_dataset.py`, para probar sin arriesgar los datos reales) están gitignored — se regeneran corriendo esos scripts, no viven en el repo.

## Las 3 configuraciones evaluadas (Fase 3/4/5)

`hybrid_inference.py` ya se corrió (Kaggle GPU) para las 3 configuraciones que pide la tarea, sobre `data/test/` (30 img, 79 cajas) y sobre `data/validation/` (40 img, 41 cajas). `data/test/` fue usado también por Ultralytics como conjunto de validación durante el entrenamiento (para elegir el mejor checkpoint), así que sus métricas están un poco optimistas; `data/validation/` es un holdout genuinamente limpio que el entrenamiento nunca vio, para tener un número más confiable. Resultados completos en `results/metrics/*.json` y `results/graphs/*_pr_curve.png`:

| Config | mAP@0.5 test | mAP@0.5 validación |
|---|---|---|
| YOLOv8s solo | 0.9584 | 0.9933 |
| + Qwen 0.8B (cascada) | 0.9412 | 0.9933 |
| + Qwen 2.0B (cascada) | 0.9609 | 0.9029 |

Curioso: el validador 2.0B gana en test pero es claramente el peor en validación (más falsos negativos que 0.8B) — en `report/figures/qwen2b_falsos_negativos_validacion.png` hay un análisis de por qué (varios de los rechazos parecen tener rasgos de Alaskan Malamute en vez de Husky Siberiano puro).

Nota sobre `.gitkeep`: solo quedan en las carpetas que siguen vacías (`report/figures/`, `report/metrics/`). Se quitaron de las que ya tienen contenido real trackeado (git no los necesita ahí).
