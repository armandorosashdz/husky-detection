# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is an early-stage academic project (VLM-assisted auto-labeling + YOLO transfer learning pipeline for husky detection). `src/rename_images.py`, `src/vlm_utils.py`, and `src/yolo_utils.py` are implemented. `src/auto_labeling.py` currently only contains a smoke test (loads the model, runs one image, prints the parsed boxes) — the real loop over all 100 images into `data/labels_auto/` is not written yet. `src/train_yolo.py`, `src/hybrid_inference.py`, `src/metrics.py` are still **empty stubs**. `requirements.txt` and `dataset.yaml` are also empty placeholders. Before assuming a function/module exists, check the file is non-empty.

`src/test_box_order.py` is a throwaway verification script (not part of the pipeline) used to visually confirm the coordinate order Qwen actually returns and to sanity-check `convert_to_yolo`'s round-trip; it overwrites `data/labels_check/test_orden_*.jpg` each run.

### Runtime environment (important — no GPU on this machine)

This laptop has **no NVIDIA/CUDA GPU** (AMD integrated graphics only), despite the assignment asking to run Qwen on GPU. Consequences baked into the current code:
- `config.py`: `DEVICE = "cpu"`, `DTYPE = "float32"` (float16 generation isn't well supported on CPU). Switch back to `"cuda"`/`"float16"` if this ever runs on a machine with an NVIDIA GPU.
- The smoke test in `auto_labeling.py` deliberately uses the **0.8B** model (`config.QWEN_VALIDATORS["0.8b"]`), not `QWEN_LABELER` (4B) — the 2B model already runs out of RAM (~14GB total, ~7GB free) when loaded in float32 on CPU. The real Fase 1 loop still needs to target `config.QWEN_LABELER`, but be aware of this memory ceiling when deciding which size to actually use for the full 100-image run.
- Dependencies (`torch`, `transformers`, `accelerate`) live in the **conda env `tarea3`** (`C:\Users\Armando\anaconda3\envs\tarea3`), which already had `torch`/`torchvision`/`ultralytics`/`pillow` from a prior assignment. Run scripts with `conda run -n tarea3 python src/<script>.py`, or `conda activate tarea3` first. Not reflected in `requirements.txt` (still empty).
- Considered Ollama (quantized GGUF, much lower RAM) as an alternative runtime if memory keeps being a blocker — not adopted yet, would require rewriting `QwenVLM` to talk to Ollama's HTTP API instead of `transformers`.

### Model IDs — real names differ from the PDF spec

The assignment PDF's example code uses `"Qwen/Qwen3.5-4B-VL"`-style IDs and `AutoModelForVision2Seq`. Neither is correct for the actually-published models:
- Real HF repo IDs have **no `-VL` suffix** (Qwen3.5 is natively multimodal): `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B` — already fixed in `config.py`.
- The correct `transformers` class is **`AutoModelForMultimodalLM`**, not `AutoModelForVision2Seq` (deprecated) or `AutoModelForImageTextToText` (also wrong for this model family) — per the official model card usage example.
- `QwenVLM.load()`/`.ask()` follow the model card's official single-call pattern (`processor.apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")`), not the older two-step pattern (`tokenize=False` + separate `processor(text=..., images=...)`) used by some other VL models.

## Pipeline architecture

The intended pipeline runs in phases, each corresponding to a module in `src/`:

1. **Rename** (`src/rename_images.py`, implemented) — normalizes raw images in `data/raw/` to `husky_000.jpg ... husky_099.jpg`. Must run **before** auto-labeling, since re-running it after labels exist breaks the image↔label pairing (it does a two-pass rename through temp names to avoid collisions).
2. **Auto-labeling** (`src/auto_labeling.py`, stub, orchestrator only) — Phase 1: `data/raw/` → `data/labels_auto/`, using a large Qwen VLM (`QWEN_LABELER` in `config.py`) to produce YOLO-format bounding boxes from `PROMPT_LABELING`. Model loading, inference, response parsing, and YOLO-format conversion live in `src/vlm_utils.py` (see below); `auto_labeling.py` just calls into it and writes the `.txt` files.
3. **Training** (`src/train_yolo.py`, stub) — Phase 2: fine-tunes a YOLOv8 model (`YOLO_BASE`) on `data/train/`.
4. **Hybrid inference** (`src/hybrid_inference.py`, stub) — Phases 3-4: runs YOLO detection then validates crops with a cascade of small Qwen VLMs (`QWEN_VALIDATORS`, selected via `--validator 0.8b|2b` flag) using `PROMPT_VALIDATION`.
5. **Metrics** (`src/metrics.py`, stub) — computes mAP, FP/FN, latency, and P-R curves into `results/metrics/` and `results/figures/`.

All shared configuration — paths, model IDs, prompts, thresholds, hyperparameters — lives centrally in `config.py`. New scripts should import it (`import config`) rather than hardcoding paths or constants. Scripts run from the repo root add the parent dir to `sys.path` to import `config` (see `src/rename_images.py`).

### `src/vlm_utils.py` design

Despite the name (kept generic on purpose — see comments in the file for what lives where), this module holds everything Qwen/VLM-related plus the one generic bbox-format helper, reused by both `auto_labeling.py` (Fase 1) and `hybrid_inference.py` (Fase 4) so the model is loaded/wrapped in one place:

- `QwenVLM` class — `__init__(model_id)`, `load()`, `ask(image, prompt) -> str` (generic: runs the model and returns raw text, used both for the detection prompt and the binary Yes/No validation prompt).
- `parse_boxes(response) -> list[[x1, y1, x2, y2]]` — Qwen-specific, parses its raw text response into boxes on the 0-1000 scale. **Important:** despite `PROMPT_LABELING` literally asking for `[ymin, xmin, ymax, xmax]`, Qwen3.5 ignores that and always responds in its native "grounding" schema `[{"bbox_2d": [x1, y1, x2, y2], "label": ...}, ...]`. Verified visually by drawing both interpretations on 3 sample images (see `data/labels_check/test_orden_*.jpg`, generated by `src/test_box_order.py`) — the `bbox_2d` `[x1,y1,x2,y2]` order was the one that actually bounded the dogs correctly. `PROMPT_LABELING` was updated to ask for this native format directly instead of fighting it.
- `convert_to_yolo(box, class_id) -> str` — generic geometry/format conversion (0-1000 scale `[x1,y1,x2,y2]` → normalized YOLO `class_id x_center y_center width height`), not actually Qwen-specific but kept here to avoid an extra file; comment it clearly as such. Round-trip verified by `src/test_box_order.py` (Qwen box → YOLO line → back to pixels → drawn).

### `src/yolo_utils.py` design

Mirrors the same pattern as `QwenVLM`, but for the trained YOLOv8 detector — separate file because it's a different concern (running the detector, Fase 3/4) from `vlm_utils.py` (generating training labels, Fase 1):

- `YOLODetector` class — `__init__(model_path=config.YOLO_TRAINED)`, `load()` (wraps `ultralytics.YOLO`), `detect(image) -> list[dict]` (runs inference with `config.CONF_THRESHOLD`/`IOU_THRESHOLD`, returns `{"box": (x1,y1,x2,y2) in pixels, "conf": float, "class_id": int}` per detection), `crop(image, box, padding=config.CROP_PADDING) -> Image` (crops a detection with margin, clamped to image bounds, for feeding into the Qwen validator in Fase 4).
- Not yet used anywhere — `train_yolo.py` and `hybrid_inference.py` are still stubs.

## Data flow / directory layout

```
data/raw/            # unannotated source images (husky_NNN.ext), input to auto-labeling
data/labels_auto/    # YOLO-format .txt labels generated by the Qwen labeler
data/labels_check/   # visualizations with drawn bounding boxes, for manual QA
data/train/          # 70% split: images/ + labels/
data/test/           # 30% split: images/ + labels/
results/metrics/     # one JSON per evaluated configuration
results/figures/     # curves, confusion matrices, example detections
```

The train/test split ratio (`TRAIN_RATIO = 0.7`) and reproducibility seed (`SEED = 42`) are fixed in `config.py`.

## Commands

```bash
python src/rename_images.py
```
Renames all images in `data/raw/` to the `husky_NNN.ext` scheme. Run once, before any labeling step.

No build, lint, or test tooling is configured yet (`requirements.txt` is empty, no test suite exists).

## Assignment specification (Tarea #4)

This repo implements a graded assignment ("Pipeline Autónomo de Detección – Auto-etiquetado con Qwen VL, Transfer Learning y Validación"). Full spec: `Especificaciones de Tarea 4_ Pipeline Autónomo con VLM y Transfer Learning.pdf` (repo root). Key constraints future work must respect:

- **Fase 1 (auto_labeling.py):** 100 unlabeled husky images (multiple dogs per image), auto-labeled locally with **Qwen3.5** (0.8B/2B/4B — see model ID note above; PDF says 2B or 4B, but GPU is unavailable here so size choice is memory-constrained). Qwen actually returns boxes as `[x1, y1, x2, y2]` on a 0-1000 scale (native format — see `vlm_utils.py` design note, this differs from what the PDF's example prompt asks for); must be converted to YOLO format (`class_id x_center y_center width height`, normalized 0-1) via `convert_to_yolo` — one `.txt` per image, same basename as the image. Bounding boxes must then be visually verified (drawn on images, → `data/labels_check/`).
- **Fase 2 (train_yolo.py):** Fine-tune **YOLOv8s** (Ultralytics, pretrained) on the 70-image train split via `dataset.yaml`; keep the other pretrained COCO classes active (don't replace the head). Use aggressive augmentation (mosaic, mixup, hsv_h, etc. — already parameterized in `config.py`'s `AUGMENT` dict) since the dataset is small. The remaining 30 images are test-only.
- **Fase 3 (hybrid_inference.py — deployment part):** Optimized Python inference script loading the trained YOLOv8s weights, simulating a production image stream.
- **Fase 4 (hybrid_inference.py — validation part):** Cascade validation — every YOLO detection is cropped and sent to a Qwen VL validator with a binary Yes/No prompt; "No" discards the detection as a false positive. Must run and compare **both** validator sizes: 0.8B and 2B (`QWEN_VALIDATORS` in `config.py`, selected via `--validator 0.8b|2b` CLI flag on `hybrid_inference.py`). Recommended: keep `CONF_THRESHOLD` low so more raw detections reach the cascade (already set in `config.py`).
- **Metrics (metrics.py):** For each of the 3 pipeline configs (YOLOv8s alone, +0.8B validator, +2B validator), compute mAP@0.5, false positives, false negatives, inference latency (ms), and real FPS — feeds the comparison table and 3 Precision-Recall curves required in the report.

### Required deliverables
- Source: `auto_labeling.py`, `train_yolo.py` (or notebook), `hybrid_inference.py` (all currently empty stubs — see Project status above).
- Technical report (PDF): filled comparison table (3 configs × mAP/FP/FN/latency/FPS), training curves (bbox loss, objectness loss, precision/recall/mAP@0.5), confusion matrix on the 30-image test set, visual results (successes, false positives mitigated by each validator, failure cases), and answers to the critical-analysis questionnaire (compute cost of 0.8B vs 2B cascade and real-time viability, error propagation from Fase 1 labeling bias into YOLO's bbox regression loss, prompt-engineering sensitivity, and CNN classifier vs VLM validator tradeoffs).
