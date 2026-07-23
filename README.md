# husky-detection

# Estructura

husky-detection/
├── README.md              # setup, cómo correr, quién hace qué
├── requirements.txt
├── .gitignore             # *.pt, runs/, data/, __pycache__
├── config.py              # model_ids, prompts, thresholds, rutas
├── src/
│   ├── qwen_utils.py      # cargar Qwen + parsear + convertir a YOLO
│   ├── auto_labeling.py   # Fase 1: raw/ → labels_auto/
│   ├── train_yolo.py      # Fase 2
│   ├── hybrid_inference.py# Fases 3-4 (--validator 0.8b|2b)
│   └── metrics.py         # mAP, FP/FN, latencia, curvas P-R
├── dataset.yaml
├── data/                  # link en README, no al repo
│   ├── raw/               # 100 imágenes sin anotar (crudas)
│   ├── labels_auto/       # .txt generados por Qwen
│   ├── labels_check/      # visualizaciones con BB dibujadas
│   ├── train/{images,labels}/   # 70
│   └── test/{images,labels}/    # 30
├── results/
│   ├── metrics/           # un JSON por configuración
│   └── figures/           # curvas, matriz de confusión, ejemplos
└── report/
    └── reporte.pdf