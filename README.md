# husky-detection

## Estructura del repositorio

```text
husky-detection/
├── README.md                    # setup, cómo correr, quién hace qué
├── CLAUDE.md                    # guía del repo para Claude Code (arquitectura, estado, decisiones)
├── Especificaciones de Tarea 4...pdf   # enunciado original de la tarea
├── requirements.txt              # torch, torchvision, transformers, accelerate, ultralytics, pillow (`pip install -r requirements.txt`)
├── .gitignore                   # __pycache__, *.pt, runs/, data/*_fixture/, dataset_fixture.yaml (excepto models/*.pt, sí se comitean)
├── config.py                    # model_ids, prompts, thresholds, rutas, límites — todo lo configurable vive aquí
├── src/
│   ├── utils.py                 # QwenVLM + parse_boxes + convert_to_yolo + YOLODetector
│   ├── auto_labeling.py         # Fase 1: raw/ → labels_auto/ + labels_check/ (implementado)
│   ├── split_dataset.py         # prepara train/test (70/30) + dataset.yaml, previo a Fase 2 (implementado)
│   ├── train_yolo.py            # Fase 2: ajuste fino de YOLOv8s (implementado)
│   ├── hybrid_inference.py      # Fases 3-4 (pendiente)
|   ├── rename_and_resize_images.py         # Renombra y redimensiona las imagenes de data/raw/ (implementado)
│   ├── generate_fixture_dataset.py  # script de prueba descartable: genera dataset falso
│   ├── test_box_order.py        # script de prueba descartable, no es parte del pipeline
│   └── metrics.py               # mAP, FP/FN, latencia, curvas P-R (pendiente)
├── dataset.yaml                 # generado por split_dataset.py (ya corrido con datos reales)
├── yolov8s.pt                   # pesos preentrenados, descargados por Ultralytics al entrenar (gitignored)
├── runs/                        # resultados de entrenamiento de Ultralytics, incl. runs/detect/train/weights/best.pt (gitignored)
├── models/                      # modelos finales del equipo, sí se comitean (excepción a *.pt)
│   ├── yolov8_finetuned_armando.pt
│   └── yolov8_finetuned_pedro.pt
├── data/                        # sí está en el repo (incluyendo las 100 imágenes crudas)
│   ├── raw/                     # 100 imágenes sin anotar (crudas, redimensionadas) + Dataset2.zip (respaldo)
│   ├── labels_auto/             # .txt generados por Qwen (Fase 1 completa: 100/100)
│   ├── labels_check/            # visualizaciones con BB dibujadas (Fase 1 completa: 100/100)
│   ├── train/                   # 70 imágenes+labels (split real ya generado)
│   ├── test/                    # 30 imágenes+labels (split real ya generado)
│   └── validation/              # vacía por ahora (config.VALIDATION_DIR, aún sin usar en ningún script)
├── results/
│   ├── metrics/                 # un JSON por configuración
│   └── figures/                 # curvas, matriz de confusión, ejemplos
└── report/
    └── reporte.pdf
```

Nota: `data/*_fixture/` y `dataset_fixture.yaml` (dataset falso de `generate_fixture_dataset.py`/`split_dataset.py`, para probar sin arriesgar los datos reales) están gitignored — se regeneran corriendo esos scripts, no viven en el repo.

Nota sobre `.gitkeep`: solo quedan en las carpetas que siguen vacías (`data/validation/`, `report/figures/`, `report/metrics/`, `results/`). Se quitaron de las que ya tienen contenido real trackeado (git no los necesita ahí).