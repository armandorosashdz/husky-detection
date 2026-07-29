# husky-detection

## Estructura del repositorio

```text
husky-detection/
├── README.md                    # setup, cómo correr, quién hace qué
├── CLAUDE.md                    # guía del repo para Claude Code (arquitectura, estado, decisiones)
├── Especificaciones de Tarea 4...pdf   # enunciado original de la tarea
├── requirements.txt              # vacío por ahora; dependencias reales viven en el env conda "tarea3" (ver CLAUDE.md)
├── .gitignore                   # __pycache__ y estándar Python; TODAVÍA NO ignora *.pt/runs/ (agregar antes de entrenar)
├── config.py                    # model_ids, prompts, thresholds, rutas, límites — todo lo configurable vive aquí
├── src/
│   ├── vlm_utils.py             # QwenVLM (carga+ask) + parse_boxes + convert_to_yolo
│   ├── yolo_utils.py            # YOLODetector: carga YOLOv8 entrenado + detect + crop
│   ├── auto_labeling.py         # Fase 1: raw/ → labels_auto/ + labels_check/ (implementado)
│   ├── train_yolo.py            # Fase 2 (pendiente)
│   ├── hybrid_inference.py      # Fases 3-4 (pendiente)
|   ├── rename_images.py         # Renombra las imagenes en data/raw/ (implementado)
│   ├── test_box_order.py        # script de prueba descartable, no es parte del pipeline
│   └── metrics.py               # mAP, FP/FN, latencia, curvas P-R (pendiente)
├── dataset.yaml                 # vacío por ahora (se llena en Fase 2)
├── data/                        # sí está en el repo (incluyendo las 100 imágenes crudas)
│   ├── raw/                     # 100 imágenes sin anotar (crudas)
│   ├── labels_auto/             # .txt generados por Qwen (Fase 1, en progreso)
│   ├── labels_check/            # visualizaciones con BB dibujadas (Fase 1, en progreso)
│   ├── train/
│   │   ├── images/              # 70
│   │   └── labels/
│   └── test/
│       ├── images/              # 30
│       └── labels/
├── results/
│   ├── metrics/                 # un JSON por configuración
│   └── figures/                 # curvas, matriz de confusión, ejemplos
└── report/
    └── reporte.pdf
```