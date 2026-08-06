# husky-detection

## Estructura del repositorio

```text
husky-detection/
├── README.md                    # setup, cómo correr, quién hace qué
├── CLAUDE.md                    # guía del repo para Claude Code (arquitectura, estado, decisiones)
├── Especificaciones de Tarea 4...pdf   # enunciado original de la tarea
├── requirements.txt              # torch, torchvision, transformers, accelerate, ultralytics, pillow, matplotlib (`pip install -r requirements.txt`)
├── .gitignore                   # __pycache__, *.pt (excepto models/*.pt), data/*_fixture/, dataset_fixture.yaml
├── config.py                    # model_ids, prompts, thresholds, rutas, límites — todo lo configurable vive aquí
├── src/
│   ├── utils.py                 # QwenVLM + parse_boxes + convert_to_yolo + YOLODetector
│   ├── rename_and_resize_images.py  # renombra+redimensiona TARGET_DIR (implementado)
│   ├── auto_labeling.py         # Fase 1: INPUT_DIR → LABELS_AUTO_OUT + LABELS_CHECK_OUT (implementado)
│   ├── split_dataset.py         # prepara train/test (70/30) + dataset.yaml, previo a Fase 2 (implementado)
│   ├── train_yolo.py            # Fase 2: ajuste fino de YOLOv8s (implementado, corrido para real)
│   ├── hybrid_inference.py      # Fases 3-4: YOLO + cascada Qwen, EVAL_DIR configurable (implementado)
│   ├── metrics.py               # Fase 5: IoU, mAP@0.5, curva P-R (implementado, lo usa hybrid_inference.py)
│   ├── generate_fixture_dataset.py  # script de prueba descartable: genera dataset falso
│   └── test_box_order.py        # script de prueba descartable, no es parte del pipeline
├── dataset.yaml                 # generado por split_dataset.py (ya corrido con datos reales)
├── yolov8s.pt                   # pesos preentrenados, descargados por Ultralytics al entrenar (gitignored)
├── runs/detect/train/           # resultados del entrenamiento real: curvas, matriz de confusión, results.csv
│   └── weights/                 # best.pt/last.pt NO se comitean (ver weights/README.txt) -- duplican models/
├── models/                      # modelos finales del equipo, sí se comitean (excepción a *.pt)
│   ├── yolov8_finetuned_armando.pt   # = config.YOLO_TRAINED, el que usa el resto del pipeline
│   └── yolov8_finetuned_pedro.pt     # modelo de comparación, no referenciado por ningún script
├── data/
│   ├── raw/                     # 100 imágenes sin anotar (crudas, redimensionadas) + Dataset2.zip (respaldo)
│   ├── labels_auto/             # .txt generados por Qwen (Fase 1 completa: 100/100)
│   ├── labels_check/            # visualizaciones con BB dibujadas (Fase 1 completa: 100/100)
│   ├── train/                   # 70 imágenes+labels (split real ya generado)
│   ├── test/                    # 30 imágenes+labels (split real ya generado) -- usado también como val
│   │                             # de Ultralytics durante el entrenamiento, ver nota en CLAUDE.md
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

`hybrid_inference.py` ya se corrió (Kaggle GPU) para las 3 configuraciones que pide la tarea, sobre `data/test/` (30 img, 79 cajas) y sobre `data/validation/` (40 img, 41 cajas — holdout que el entrenamiento nunca vio). Resultados completos en `results/metrics/*.json` y `results/graphs/*_pr_curve.png`:

| Config | mAP@0.5 test | mAP@0.5 validación |
|---|---|---|
| YOLOv8s solo | 0.9584 | 0.9933 |
| + Qwen 0.8B (cascada) | 0.9412 | 0.9933 |
| + Qwen 2.0B (cascada) | 0.9609 | 0.9029 |

Curioso: el validador 2.0B gana en test pero es claramente el peor en validación (más falsos negativos que 0.8B) — en `report/figures/qwen2b_falsos_negativos_validacion.png` hay un análisis de por qué (varios de los rechazos parecen tener rasgos de Alaskan Malamute en vez de Husky Siberiano puro).

## Setup

```bash
pip install -r requirements.txt
```
En esta laptop (sin GPU) los scripts se corren con `conda run -n tarea3 python src/<script>.py` — ver "Runtime environment" en `CLAUDE.md` para detalles y para correr en Kaggle/Colab.

Nota sobre `.gitkeep`: solo quedan en las carpetas que siguen vacías (`report/figures/`, `report/metrics/`). Se quitaron de las que ya tienen contenido real trackeado (git no los necesita ahí).
