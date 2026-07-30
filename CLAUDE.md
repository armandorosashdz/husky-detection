# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is an early-stage academic project (VLM-assisted auto-labeling + YOLO transfer learning pipeline for husky detection). `src/rename_images.py`, `src/utils.py` (Qwen + YOLO utilities, merged into one file — see design note below), `src/auto_labeling.py` (Fase 1, full loop over `data/raw/`), `src/split_dataset.py` (train/test split + `dataset.yaml` generation, prep step before Fase 2), and `src/train_yolo.py` (Fase 2, fine-tunes YOLOv8s) are implemented. `src/hybrid_inference.py`, `src/metrics.py` are still **empty stubs**. `requirements.txt` and `dataset.yaml` (the real one, not `dataset_fixture.yaml`) are also empty placeholders. Before assuming a function/module exists, check the file is non-empty.

`train_yolo.py` has only been run against the **fixture** dataset so far (2 epochs, sanity check) — never against the real 100-image dataset. The real run needs `config.EPOCHS` back at its full value (currently `100`, was temporarily set to `2` for the fixture test) and `DATASET_YAML_PATH` in `train_yolo.py` switched to `config.DATASET_YAML`. On this CPU-only laptop, 2 epochs over the 70-image fixture train split took ~1.5 min each — budget accordingly for a real 100-epoch run (could be hours).

`config.AUTO_LABELING_LIMIT` is currently `None` (set for the real full run), but `data/labels_auto/`/`data/labels_check/` only have output for 5 images so far — the full 100-image run hasn't actually been executed yet. Don't assume all 100 are labeled without checking.

**Project convention: no CLI arguments.** Scripts (`auto_labeling.py`, and presumably `hybrid_inference.py`/`train_yolo.py` once written) take no `argparse` flags — anything that needs to vary between runs (which Qwen size to use, how many images to process, etc.) is a variable in `config.py` instead, edited by hand. This was an explicit project decision, not an oversight — don't reintroduce CLI args without checking first.

`src/test_box_order.py` is a throwaway verification script (not part of the pipeline) used to visually confirm the coordinate order Qwen actually returns and to sanity-check `convert_to_yolo`'s round-trip; it overwrites `data/labels_check/test_orden_*.jpg` each run.

### Runtime environment (important — no GPU on this machine)

This laptop has **no NVIDIA/CUDA GPU** (AMD integrated graphics only), despite the assignment asking to run Qwen on GPU. Consequences baked into the current code:
- `config.py`: `DEVICE = "cpu"`, `DTYPE = "float32"` (float16 generation isn't well supported on CPU). Switch back to `"cuda"`/`"float16"` if this ever runs on a machine with an NVIDIA GPU.
- `config.QWEN_LABELER` is currently set to `QWEN_MODELS["0.8b"]`, not the 4B the assignment asks for — the 2B model already runs out of RAM (~14GB total, ~7GB free) when loaded in float32 on CPU, so 4B has no chance. `QWEN_MODELS` (in `config.py`) holds all three sizes; change which one `QWEN_LABELER` points to by editing that one line, no code changes needed. On a machine with a real GPU, switch it to `QWEN_MODELS["2b"]` or `["4b"]` per the assignment.
- `config.AUTO_LABELING_LIMIT` caps how many images `auto_labeling.py` processes per run (`None` = all of `data/raw/`). Set it to a small int for quick local testing.
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
2. **Auto-labeling** (`src/auto_labeling.py`, implemented, orchestrator only) — Phase 1: loops over `data/raw/` (up to `config.AUTO_LABELING_LIMIT` images), asks Qwen (`config.QWEN_LABELER`) for boxes via `PROMPT_LABELING`, writes YOLO-format `.txt` to `data/labels_auto/`, and saves a drawn-box visualization of each image to `data/labels_check/` for manual QA. Model loading, inference, response parsing, and YOLO-format conversion live in `src/utils.py` (see below); `auto_labeling.py` just calls into it.
3. **Training** (`src/train_yolo.py`, implemented) — Phase 2: fine-tunes a YOLOv8 model (`YOLO_BASE`) using the hyperparameters in `config.py` (`EPOCHS`, `IMG_SIZE`, `BATCH`, `PATIENCE`, `SEED`, `AUGMENT`), on whichever `dataset.yaml` `DATASET_YAML_PATH` (top of the file) points to. Preceded by `src/split_dataset.py` (implemented, separate script — see design note below), which turns `data/raw/` + `data/labels_auto/` into `data/train/`+`data/test/` and `dataset.yaml`.
4. **Hybrid inference** (`src/hybrid_inference.py`, stub) — Phases 3-4: runs YOLO detection (`YOLODetector` in `src/utils.py`) then validates crops with a cascade of small Qwen VLMs (`QWEN_VALIDATORS` in `config.py`) using `PROMPT_VALIDATION`. Per the no-CLI-args convention, which validator size(s) to run should be a `config.py` variable, not a `--validator` flag (the assignment PDF's example suggests a CLI flag, but that contradicts this project's convention — see note above).
5. **Metrics** (`src/metrics.py`, stub) — computes mAP, FP/FN, latency, and P-R curves into `results/metrics/` and `results/figures/`.

All shared configuration — paths, model IDs, prompts, thresholds, hyperparameters — lives centrally in `config.py`. New scripts should import it (`import config`) rather than hardcoding paths or constants. Scripts run from the repo root add the parent dir to `sys.path` to import `config` (see `src/rename_images.py`).

### `src/utils.py` design

Holds both Qwen/VLM stuff and the YOLO detector wrapper together in one file — an explicit project decision (previously split across `vlm_utils.py`/`yolo_utils.py`; merged on request into a single general-purpose `utils.py`). Reused by `auto_labeling.py` (Fase 1) and, once written, `hybrid_inference.py` (Fase 3/4).

Qwen section:
- `QwenVLM` class — `__init__(model_id)`, `load()`, `ask(image, prompt) -> str` (generic: runs the model and returns raw text, used both for the detection prompt and the binary Yes/No validation prompt).
- `parse_boxes(response) -> list[[x1, y1, x2, y2]]` — Qwen-specific, parses its raw text response into boxes on the 0-1000 scale. **Important:** despite `PROMPT_LABELING` literally asking for `[ymin, xmin, ymax, xmax]`, Qwen3.5 ignores that and always responds in its native "grounding" schema `[{"bbox_2d": [x1, y1, x2, y2], "label": ...}, ...]`. Verified visually by drawing both interpretations on 3 sample images (see `data/labels_check/test_orden_*.jpg`, generated by `src/test_box_order.py`) — the `bbox_2d` `[x1,y1,x2,y2]` order was the one that actually bounded the dogs correctly. `PROMPT_LABELING` was updated to ask for this native format directly instead of fighting it.
- `convert_to_yolo(box, class_id) -> str` — generic geometry/format conversion (0-1000 scale `[x1,y1,x2,y2]` → normalized YOLO `class_id x_center y_center width height`), not actually Qwen-specific but kept here anyway per the merge decision above. Round-trip verified by `src/test_box_order.py` (Qwen box → YOLO line → back to pixels → drawn).

YOLO section:
- `YOLODetector` class — `__init__(model_path=config.YOLO_TRAINED)`, `load()` (wraps `ultralytics.YOLO`), `detect(image) -> list[dict]` (runs inference with `config.CONF_THRESHOLD`/`IOU_THRESHOLD`, returns `{"box": (x1,y1,x2,y2) in pixels, "conf": float, "class_id": int}` per detection), `crop(image, box, padding=config.CROP_PADDING) -> Image` (crops a detection with margin, clamped to image bounds, for feeding into the Qwen validator in Fase 4).
- Not yet used anywhere — `train_yolo.py` trains via `ultralytics.YOLO` directly (training isn't `YOLODetector`'s job); `YOLODetector` is meant for `hybrid_inference.py`, still a stub.

### `src/split_dataset.py` design

Prep step before Fase 2 (`train_yolo.py`), kept as a separate script rather than folded into training — same reasoning as `rename_images.py` being separate from `auto_labeling.py`: it's a one-time (or occasional) data-prep operation, not something that should re-run every time you train.

- `listar_pares(images_dir, labels_dir)` — pairs each image with its `.txt` by matching basename, skips (with a warning) any image missing a label.
- `dividir(pares, train_ratio, seed)` — shuffles with a fixed seed and splits by ratio. Deliberately isolated from the copy/write logic so it's the one place to change if a different splitting strategy is ever needed (e.g. k-fold) — **not implemented**, was explicitly scoped out as over-engineering for what the assignment asks (a single fixed 70/30 split).
- `copiar_pares(pares, dest_dir)` — copies into the `images/`+`labels/` structure Ultralytics expects (it discovers labels by string-replacing `images`→`labels` in the path, so this exact layout matters).
- `escribir_dataset_yaml(...)` — writes the Ultralytics-format `dataset.yaml` (`path`/`train`/`val`/`names`).
- `verificar_dataset_yaml(...)` — calls `ultralytics.data.utils.check_det_dataset()` on the generated yaml as a real validation (not just "the file was written"), catching structural problems before a training run would hit them.
- All source/dest paths and `DEST_YAML_PATH` are variables at the top of the file (no CLI args), defaulting to the **fixture** paths (see below) with the real `config.RAW_DIR`/`config.LABELS_AUTO_DIR`/`config.TRAIN_DIR`/`config.TEST_DIR`/`config.DATASET_YAML` commented out alongside — swap which block is active before running for real.

### `src/train_yolo.py` design

Same no-CLI-args/fixture-toggle pattern as `split_dataset.py`:

- `DATASET_YAML_PATH` is a variable at the top of the file, defaulting to `config.DATASET_YAML_FIXTURE` (real one commented out alongside) — swap before running for real.
- All training hyperparameters come straight from `config.py` (`YOLO_BASE`, `EPOCHS`, `IMG_SIZE`, `BATCH`, `PATIENCE`, `SEED`, `DEVICE`, `AUGMENT` dict unpacked as kwargs) — no hardcoded values in the script itself.
- `project`/`name` passed to `model.train()` are derived from `config.YOLO_TRAINED` (`.parent.parent` for the train dir, `.parent` of that for the project dir) with `exist_ok=True`, so results always land at the exact path `config.YOLO_TRAINED` and `YOLODetector` (in `utils.py`) expect — instead of Ultralytics' default behavior of creating `train2/`, `train3/`, etc. on repeated runs.
- Verified against the fixture dataset (2 epochs, `EPOCHS` temporarily overridden — see Project status note above): downloads `yolov8s.pt`, correctly overrides `nc=80` → `nc=1`, trains, and saves `best.pt`/`last.pt` at `runs/detect/train/weights/`.

### Fixture pattern for testing pipeline steps safely

Established pattern (first used for `split_dataset.py`, likely reusable for future steps): generate throwaway fake data instead of testing against the real dataset, so bugs in a new script can't corrupt `data/raw/`, `data/labels_auto/`, `data/train/`, `data/test/`, or the real `dataset.yaml`.

- `config.py` centralizes the fixture paths (`RAW_FIXTURE_DIR`, `LABELS_AUTO_FIXTURE_DIR`, `LABELS_CHECK_FIXTURE_DIR`, `TRAIN_FIXTURE_DIR`, `TEST_FIXTURE_DIR`, `DATASET_YAML_FIXTURE`) — same convention as the real paths, deliberately kept in the central config even though they're only used by throwaway scripts, per an explicit project decision to avoid scattering hardcoded paths across scripts.
- `src/generate_fixture_dataset.py` (throwaway, not part of the pipeline) — generates `N_IMAGENES` blank images with 1-3 random boxes each, writing them through the **real** `dibujar_cajas()` (imported from `auto_labeling.py`) and `convert_to_yolo()` (from `utils.py`), so the fixture format exactly matches what the real pipeline produces. `PREFIJO` (default `"husky"`) is configurable so the fixture can simulate a different naming scheme without touching the real pipeline.
- `src/split_dataset.py` then runs against `RAW_FIXTURE_DIR`/`LABELS_AUTO_FIXTURE_DIR` by default, writing to `TRAIN_FIXTURE_DIR`/`TEST_FIXTURE_DIR`/`DATASET_YAML_FIXTURE` — verified working (100→70/30 split, `check_det_dataset` passes).

### Fase 2 design decision: single class, COCO classes NOT preserved

The assignment PDF says *"dejen las otras clases pre-entrenadas activas"* (keep the pretrained COCO classes active) when fine-tuning YOLOv8s. In practice, standard Ultralytics fine-tuning with a `dataset.yaml` that has `nc: 1` **reinitializes the detection head** — the model ends up only able to predict "husky", losing the other 80 COCO classes. This was researched before deciding how to proceed:
- Freezing layers alone does **not** preserve other classes — weights aren't class-specific, so training only on husky data still degrades COCO-class performance (catastrophic forgetting) even with the backbone frozen.
- A technique exists to genuinely preserve COCO classes (see [y-t-g.github.io/tutorials/yolov8n-add-classes](https://y-t-g.github.io/tutorials/yolov8n-add-classes/)): freeze the first 22 layers, train a *second* detection head on only the new class, then merge both heads' outputs with a custom `ConcatHead` layer. Doesn't require the COCO dataset itself, but requires patching/cloning the `ultralytics` library (custom layer, modified model YAML, training callbacks) — moderately complex, and the tutorial recommends 1000+ images for the new class (we have 70 train images).
- **Decision: went with the simple/standard approach — `nc: 1`, single class, COCO classes are lost.** Given no GPU, a small dataset, and limited time for an academic assignment, patching the library wasn't worth it. This is a known, deliberate limitation — document it in the technical report (relevant to the assignment's critical-analysis questionnaire) rather than treating it as an oversight.

## Data flow / directory layout

```
data/raw/            # unannotated source images (husky_NNN.ext), input to auto-labeling
data/labels_auto/    # YOLO-format .txt labels generated by the Qwen labeler
data/labels_check/   # visualizations with drawn bounding boxes, for manual QA
data/train/          # 70% split: images/ + labels/, written by split_dataset.py
data/test/           # 30% split: images/ + labels/, written by split_dataset.py
data/*_fixture/       # throwaway fake dataset (raw/labels_auto/labels_check/train/test),
                      # generated by generate_fixture_dataset.py, for testing pipeline
                      # scripts without touching real data — see fixture pattern note above
results/metrics/     # one JSON per evaluated configuration
results/figures/     # curves, confusion matrices, example detections
```

The train/test split ratio (`TRAIN_RATIO = 0.7`) and reproducibility seed (`SEED = 42`) are fixed in `config.py`.

## Commands

Run with `conda run -n tarea3 python src/<script>.py` (or `conda activate tarea3` first) — see Runtime environment above.

```bash
python src/rename_images.py
```
Renames all images in `data/raw/` to the `husky_NNN.ext` scheme. Run once, before any labeling step.

```bash
python src/auto_labeling.py
```
Fase 1: labels up to `config.AUTO_LABELING_LIMIT` images from `data/raw/` with `config.QWEN_LABELER`, writing `.txt` to `data/labels_auto/` and drawn-box visualizations to `data/labels_check/`.

```bash
python src/split_dataset.py
```
Prep step before Fase 2: splits labeled images into train/test and writes `dataset.yaml`. **Check which path block is active at the top of the file first** — it defaults to the fixture paths, not the real ones (see fixture pattern note below).

```bash
python src/train_yolo.py
```
Fase 2: fine-tunes `config.YOLO_BASE` (downloads it via Ultralytics if not cached) on whichever dataset `DATASET_YAML_PATH` (top of the file) points to, using `config.py`'s hyperparameters. Saves to `config.YOLO_TRAINED`. **Check `DATASET_YAML_PATH` and `config.EPOCHS` before running for real** — defaults/leftover values may point at the fixture dataset or a reduced epoch count from testing.

No build, lint, or test tooling is configured (`requirements.txt` is empty, no test suite exists). `*.pt` (model weights) and `runs/` (Ultralytics training outputs) are gitignored — never commit these, they're regenerated by `train_yolo.py`.

## Assignment specification (Tarea #4)

This repo implements a graded assignment ("Pipeline Autónomo de Detección – Auto-etiquetado con Qwen VL, Transfer Learning y Validación"). Full spec: `Especificaciones de Tarea 4_ Pipeline Autónomo con VLM y Transfer Learning.pdf` (repo root). Key constraints future work must respect:

- **Fase 1 (auto_labeling.py):** 100 unlabeled husky images (multiple dogs per image), auto-labeled locally with **Qwen3.5** (0.8B/2B/4B — see model ID note above; PDF says 2B or 4B, but GPU is unavailable here so size choice is memory-constrained). Qwen actually returns boxes as `[x1, y1, x2, y2]` on a 0-1000 scale (native format — see `utils.py` design note, this differs from what the PDF's example prompt asks for); must be converted to YOLO format (`class_id x_center y_center width height`, normalized 0-1) via `convert_to_yolo` — one `.txt` per image, same basename as the image. Bounding boxes must then be visually verified (drawn on images, → `data/labels_check/`).
- **Fase 2 (train_yolo.py):** Fine-tune **YOLOv8s** (Ultralytics, pretrained) on the 70-image train split via `dataset.yaml`; the PDF asks to keep the other pretrained COCO classes active (don't replace the head), but this project deliberately deviates from that — see "Fase 2 design decision" above. Use aggressive augmentation (mosaic, mixup, hsv_h, etc. — already parameterized in `config.py`'s `AUGMENT` dict) since the dataset is small. The remaining 30 images are test-only.
- **Fase 3 (hybrid_inference.py — deployment part):** Optimized Python inference script loading the trained YOLOv8s weights, simulating a production image stream.
- **Fase 4 (hybrid_inference.py — validation part):** Cascade validation — every YOLO detection is cropped (`YOLODetector.crop`) and sent to a Qwen VL validator with a binary Yes/No prompt (`PROMPT_VALIDATION`); "No" discards the detection as a false positive. Must run and compare **both** validator sizes: 0.8B and 2B (`QWEN_VALIDATORS` in `config.py` — pick which one via a config variable, not a CLI flag, per this project's convention). Recommended: keep `CONF_THRESHOLD` low so more raw detections reach the cascade (already set in `config.py`).
- **Metrics (metrics.py):** For each of the 3 pipeline configs (YOLOv8s alone, +0.8B validator, +2B validator), compute mAP@0.5, false positives, false negatives, inference latency (ms), and real FPS — feeds the comparison table and 3 Precision-Recall curves required in the report.

### Required deliverables
- Source: `auto_labeling.py` (done), `train_yolo.py` (done, not yet run for real — see Project status above), `hybrid_inference.py` (still an empty stub).
- Technical report (PDF): filled comparison table (3 configs × mAP/FP/FN/latency/FPS), training curves (bbox loss, objectness loss, precision/recall/mAP@0.5), confusion matrix on the 30-image test set, visual results (successes, false positives mitigated by each validator, failure cases), and answers to the critical-analysis questionnaire (compute cost of 0.8B vs 2B cascade and real-time viability, error propagation from Fase 1 labeling bias into YOLO's bbox regression loss, prompt-engineering sensitivity, and CNN classifier vs VLM validator tradeoffs).
