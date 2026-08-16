# Detección de Huskies Siberianos: auto-etiquetado con VLM y transfer learning

## 1. Introducción

Este repositorio implementa un pipeline autónomo de detección de objetos especializado en la raza Husky Siberiano. El sistema combina tres componentes principales: un modelo de lenguaje visual (Qwen3.5 VL) empleado para generar automáticamente las anotaciones de entrenamiento a partir de imágenes sin etiquetar, un detector YOLOv8s ajustado mediante transfer learning sobre esas anotaciones, y una etapa de validación en cascada en la que el mismo modelo de lenguaje visual confirma o descarta cada detección del YOLOv8s antes de reportarla como definitiva.

El pipeline se organiza en cinco fases: (1) auto-etiquetado de las imágenes crudas mediante el VLM, (2) ajuste fino de YOLOv8s sobre las anotaciones generadas, (3) inferencia en despliegue con el detector entrenado, (4) validación en cascada de cada detección con el VLM, y (5) cálculo de métricas de evaluación (mAP@0.5, precisión, recall, latencia y FPS). Todas las fases están implementadas como scripts independientes en `src/`, orquestables individualmente o de punta a punta mediante `main.py`.

Esta rama contiene únicamente el código y las 100 imágenes originales de `data/raw/`, sin ningún artefacto generado por el pipeline (anotaciones, modelo entrenado, métricas o figuras). Está preparada para ejecutar el pipeline completo desde cero: al correr `python main.py` (Sección 4), todas las carpetas de salida se generan automáticamente en el orden correcto, sin pasos manuales adicionales.

## 2. Estructura del repositorio

```text
husky-detection/
├── main.py                # Orquesta el pipeline completo (Sección 4)
├── config.py               # Configuración centralizada (Sección 3)
├── requirements.txt
├── src/
│   ├── utils.py                    # QwenVLM (VLM) y YOLODetector
│   ├── rename_and_resize_images.py # Normaliza imágenes crudas
│   ├── auto_labeling.py            # Fase 1: auto-etiquetado con VLM
│   ├── split_dataset.py            # Split train/test + dataset.yaml
│   ├── train_yolo.py               # Fase 2: ajuste fino de YOLOv8s
│   ├── hybrid_inference.py         # Fases 3-4: detección + cascada
│   ├── metrics.py                  # Fase 5: IoU, mAP@0.5, curva P-R
│   ├── compare_pr_curves.py        # Compara curvas P-R en una figura
│   ├── generate_fixture_dataset.py # Auxiliar, no forma parte del
│   │                                # pipeline (Sección 5.9)
│   └── test_box_order.py           # Auxiliar, no forma parte del
│                                    # pipeline (Sección 5.9)
├── data/
│   ├── raw/          # Imágenes originales, sin anotar (100)
│   ├── labels_auto/  # Anotaciones YOLO del VLM (la genera Fase 1)
│   ├── labels_check/ # Visualizaciones para revisión manual (ídem)
│   ├── train/, test/ # Partición 70/30 (la genera split_dataset.py)
│   └── validation/   # Conjunto adicional reservado como holdout;
│                      # solo trae las imágenes, sin etiquetar
├── models/             # Vacío; train_yolo.py guarda aquí el modelo
├── runs/detect/train/  # Vacío; lo genera train_yolo.py (curvas,
│                       # matriz de confusión, results.csv)
└── results/
    ├── metrics/  # Vacío; un JSON por configuración evaluada
    ├── figures/  # Vacío; imágenes anotadas por configuración
    └── graphs/   # Vacío; curvas Precision-Recall
```

## 3. Configuración centralizada (`config.py`)

Todos los parámetros ajustables del pipeline —rutas, identificadores de los modelos de Qwen, prompts, umbrales de confianza e IoU, hiperparámetros de entrenamiento y nombres de las corridas— residen exclusivamente en `config.py`, organizado por fase. Ningún script del pipeline recibe argumentos de línea de comandos: el comportamiento de una corrida se determina por completo mediante los valores definidos en este archivo antes de ejecutarlo. Antes de correr el pipeline en una máquina nueva, conviene revisar en particular:

- `DEVICE` y `DTYPE`: `"cuda"`/`"float16"` en una máquina con GPU NVIDIA; `"cpu"`/`"float32"` en su ausencia.
- `QWEN_LABELER` y `QWEN_VALIDATOR`: seleccionan, independientemente, qué tamaño de Qwen3.5 (0.8B, 2B, 4B o 9B) se usa para etiquetar y para validar en cascada; el tamaño adecuado depende de la memoria disponible.
- `HYBRID_MODE`: alterna entre `"yolo_only"` (solo el detector) y `"cascade"` (detector + validación VLM).
- `YOLO_TRAINED_NAME`: nombre bajo el cual se guarda el modelo entrenado en `models/`, para no sobrescribir el de otra corrida.

Los hiperparámetros de entrenamiento (`EPOCHS`, `OPTIMIZER`, `LR0`, `FREEZE`, `AUGMENT`, entre otros) y los umbrales de inferencia (`CONF_THRESHOLD`, `IOU_THRESHOLD`, `MAP_IOU_THRESHOLD`) se definen en un único bloque por fase, documentado con el criterio empleado para cada valor, de forma que una corrida completa pueda reproducirse o modificarse revisando un solo archivo.

## 4. Ejecución del pipeline completo

```bash
pip install -r requirements.txt
python main.py
```

Si se ejecuta el pipeline en Kaggle, `transformers`/`accelerate`/`ultralytics` ya vienen preinstalados pero desactualizados, por lo que `pip install -r requirements.txt` no los actualiza. Es necesario forzar la actualización antes de correr el pipeline:
```bash
!pip install -U transformers accelerate ultralytics -q
```
y reiniciar el kernel del notebook para que la versión actualizada quede cargada.

`main.py` ejecuta en orden el renombrado/redimensionado de imágenes, el auto-etiquetado (Fase 1), la partición del dataset, el ajuste fino de YOLOv8s (Fase 2) y, finalmente, la inferencia híbrida (Fases 3-4-5) una vez por cada configuración definida en `CONFIGURACIONES_HYBRID` —por defecto, las tres configuraciones que evalúa este trabajo: YOLOv8s solo, YOLOv8s con validación en cascada de Qwen 0.8B, y YOLOv8s con validación en cascada de Qwen 2.0B—. Para cada configuración, `main.py` ajusta automáticamente `HYBRID_MODE`/`QWEN_VALIDATOR`/`RUN_LABEL` en `config.py`, ejecuta `hybrid_inference.py` y restaura el archivo a su estado original al finalizar.

La variable `FUENTE_ACTIVA`, al inicio de `main.py`, selecciona el conjunto de imágenes de entrada (`"raw"` para el dataset principal, `"validation"` para evaluar sobre el holdout adicional) y ajusta en consecuencia las rutas leídas por cada script.

Al ejecutarse, `main.py` imprime un resumen inicial con la fuente activa, los pasos a correr, las configuraciones a evaluar y el estado del modelo YOLO, y detiene la ejecución en el primer paso que falle.

## 5. Ejecución de los módulos por separado

**5.1 Normalización de las imágenes crudas**
```bash
python src/rename_and_resize_images.py
```
Renombra las imágenes de `data/raw/` al esquema `husky_000.jpg, husky_001.jpg, ...` y redimensiona (sin ampliar) cualquier imagen que exceda `config.MAX_IMAGE_DIM` en su lado más largo. Debe ejecutarse una única vez, antes del etiquetado, ya que un renombrado posterior a la generación de etiquetas rompería la correspondencia entre cada imagen y su anotación.

**5.2 Auto-etiquetado con el VLM (Fase 1)**
```bash
python src/auto_labeling.py
```
Para cada imagen de `data/raw/`, consulta a Qwen (`config.QWEN_LABELER`) mediante `config.PROMPT_LABELING`, convierte la respuesta al formato YOLO y escribe la anotación correspondiente en `data/labels_auto/`, además de una visualización con las cajas dibujadas en `data/labels_check/` para su revisión. Cuando `config.VERIFY_BREED` está activo, cada caja detectada se somete a una segunda consulta binaria al mismo modelo (`config.PROMPT_VERIFY_BREED`), que confirma si la región corresponde específicamente a un Husky Siberiano antes de conservarla.

**5.3 Revisión de las anotaciones generadas**

Se recomienda inspeccionar visualmente el contenido de `data/labels_check/` antes de continuar con el entrenamiento, y corregir manualmente cualquier anotación de `data/labels_auto/` que no corresponda con la imagen.

**5.4 Partición del dataset**
```bash
python src/split_dataset.py
```
Separa las imágenes anotadas en un 70 % de entrenamiento y un 30 % de prueba (proporción y semilla definidas en `config.py`), organiza ambos subconjuntos en la estructura `images/`+`labels/` requerida por Ultralytics, y genera `dataset.yaml`.

**5.5 Ajuste fino de YOLOv8s (Fase 2)**
```bash
python src/train_yolo.py
```
Entrena YOLOv8s sobre el dataset generado en el paso anterior, con los hiperparámetros definidos en `config.py`, y copia el mejor punto de control obtenido a `models/<config.YOLO_TRAINED_NAME>`, que es el modelo empleado por el resto del pipeline.

**5.6 Inferencia híbrida y evaluación (Fases 3-4-5)**
```bash
python src/hybrid_inference.py
```
Ejecuta el detector YOLOv8s entrenado sobre el conjunto de evaluación y, si `config.HYBRID_MODE = "cascade"`, valida cada detección con Qwen (`config.QWEN_VALIDATOR`) antes de conservarla. Calcula mAP@0.5, precisión, recall, latencia promedio y FPS reales, y guarda el resultado en `results/metrics/`, las imágenes anotadas en `results/figures/` y la curva Precision-Recall correspondiente en `results/graphs/`. Para obtener las tres configuraciones evaluadas, este script se ejecuta tres veces, alternando `HYBRID_MODE` y `QWEN_VALIDATOR` entre corridas.

**5.7 Evaluación adicional sobre el holdout de validación**

`data/test/` es empleado por Ultralytics como conjunto de validación durante el entrenamiento (paso 5.5), por lo que sus métricas resultan levemente optimistas. `data/validation/` es un conjunto adicional nunca visto durante el entrenamiento, útil para una evaluación más estricta. Como solo trae las imágenes, primero se le deben generar pseudo-etiquetas repitiendo los pasos 5.1 y 5.2 con `FUENTE_ACTIVA = "validation"` (o `TARGET_DIR`/`INPUT_DIR` apuntando a `data/validation/images/`), y después ejecutar el paso 5.6 con `EVAL_DIR = config.VALIDATION_DIR`.

**5.8 Comparación de corridas**
```bash
python src/compare_pr_curves.py
```
Combina las curvas Precision-Recall de las corridas indicadas en `results/metrics/` sobre una misma figura, empleada para comparar las configuraciones evaluadas.

**5.9 Scripts auxiliares (no forman parte del pipeline)**

`src/generate_fixture_dataset.py` y `src/test_box_order.py` no son invocados por `main.py` ni por ninguno de los módulos anteriores. El primero genera un dataset sintético de prueba, empleado para validar el funcionamiento de `split_dataset.py` sin arriesgar los datos reales. El segundo es una utilidad de verificación que confirma el orden de las coordenadas devueltas por el VLM y el resultado de `convert_to_yolo`, dibujando ambas interpretaciones sobre imágenes de muestra.
