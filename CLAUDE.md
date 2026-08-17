# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is an early-stage academic project (VLM-assisted auto-labeling + YOLO transfer learning pipeline for husky detection). `src/rename_and_resize_images.py`, `src/utils.py` (Qwen + YOLO utilities, merged into one file — see design note below), `src/auto_labeling.py` (Fase 1, full loop over `data/raw/`), `src/split_dataset.py` (train/test split + `dataset.yaml` generation, prep step before Fase 2), and `src/train_yolo.py` (Fase 2, fine-tunes YOLOv8s) are implemented. `src/hybrid_inference.py`, `src/metrics.py` are still **empty stubs**. `dataset.yaml` (the real one, not `dataset_fixture.yaml`) is still an empty placeholder. `requirements.txt` now lists the real dependencies (`torch`, `torchvision`, `transformers`, `accelerate`, `ultralytics`, `pillow`). Before assuming a function/module exists, check the file is non-empty.

`train_yolo.py` has been run once for real (Kaggle GPU, 100 images, `DATASET_YAML_PATH` already pointing at `config.DATASET_YAML`) but hit the training-instability bug described below (`cls_loss` exploding) — fixed since (`FREEZE`/`OPTIMIZER`/`LR0`), but **that fix hasn't been re-verified against a real full run yet**, only against the fixture. On this CPU-only laptop, 2 fixture epochs take ~1.5 min each — budget accordingly for a real 100-epoch run on Kaggle (faster there, but still budget real time).

`config.py` has `VALIDATION_DIR = DATA / "validation"` (empty, `.gitkeep`) and `MODELS_DIR = RESULTS / "models"` (i.e. `results/models/`) — both added ahead of use, **not referenced by any script yet**. `MODELS_DIR` is a *different* location from `models/` at repo root, which is where `config.YOLO_TRAINED` actually lives (`models/yolov8_finetuned_armando.pt` — see `train_yolo.py` design and gitignore exception below). If `metrics.py`/`hybrid_inference.py` end up using `MODELS_DIR`, decide whether to reconcile these two "models" locations or keep them intentionally separate (e.g. `models/` = final per-person deliverables like `YOLO_TRAINED`, `results/models/` = something else, TBD).

`config.AUTO_LABELING_LIMIT` is `None` (real full run). **Fase 1 is done for real**: all 100 images in `data/raw/` have been labeled (`data/labels_auto/*.txt` + `data/labels_check/*.jpg`), run on Kaggle (2× T4) with `QWEN_LABELER = QWEN_MODELS["4b"]`, `DEVICE`/`DTYPE` set to `"cuda"`/`"float16"` for that session (not reflected in this repo's `config.py`, which still defaults to `"0.8b"`/`"cpu"` for local runs — see Runtime environment below). `src/split_dataset.py` has also been run for real (not just the fixture): `data/train/` and `data/test/` are populated (70/30) and `dataset.yaml` has real content — `split_dataset.py`'s active path block now points to the real `config.*` paths, fixture block commented out.

`config.py`'s `PROMPT_LABELING` was switched to a longer, stricter version (explicitly asks for ALL dogs, tight boxes, no guessing occluded parts, no markdown fences) after observing the shorter original sometimes stopped at the first dog or wandered in output format. The short original is kept commented above it for reference (prompt-sensitivity evidence for the assignment questionnaire).

**Project convention: no CLI arguments.** Scripts (`auto_labeling.py`, and presumably `hybrid_inference.py`/`train_yolo.py` once written) take no `argparse` flags — anything that needs to vary between runs (which Qwen size to use, how many images to process, etc.) is a variable in `config.py` instead, edited by hand. This was an explicit project decision, not an oversight — don't reintroduce CLI args without checking first.

`src/test_box_order.py` is a throwaway verification script (not part of the pipeline) used to visually confirm the coordinate order Qwen actually returns and to sanity-check `convert_to_yolo`'s round-trip; it overwrites `data/labels_check/test_orden_*.jpg` each run.

### Runtime environment (important — no GPU on this machine)

This laptop has **no NVIDIA/CUDA GPU** (AMD integrated graphics only), despite the assignment asking to run Qwen on GPU. Consequences baked into the current code:
- `config.py`: `DEVICE = "cpu"`, `DTYPE = "float32"` (float16 generation isn't well supported on CPU). Switch back to `"cuda"`/`"float16"` if this ever runs on a machine with an NVIDIA GPU.
- `config.QWEN_LABELER` is currently set to `QWEN_MODELS["0.8b"]`, not the 4B the assignment asks for — the 2B model already runs out of RAM (~14GB total, ~7GB free) when loaded in float32 on CPU, so 4B has no chance. `QWEN_MODELS` (in `config.py`) holds four sizes (`0.8b`/`2b`/`4b`/`9b`); change which one `QWEN_LABELER` points to by editing that one line, no code changes needed. On a machine with a real GPU, switch it to `QWEN_MODELS["2b"]`/`["4b"]` per the assignment, or `["9b"]` if available — **and also flip `DEVICE`/`DTYPE` to `"cuda"`/`"float16"`**, both changes have to happen together or it'll try to load a multi-GB model in float32 on CPU and fail.
- `config.AUTO_LABELING_LIMIT` caps how many images `auto_labeling.py` processes per run (`None` = all of `data/raw/`). Set it to a small int for quick local testing.
- Dependencies (`torch`, `transformers`, `accelerate`) live in the **conda env `tarea3`** (`C:\Users\<usuario>\anaconda3\envs\tarea3`), which already had `torch`/`torchvision`/`ultralytics`/`pillow` from a prior assignment. Run scripts with `conda run -n tarea3 python src/<script>.py`, or `conda activate tarea3` first. `requirements.txt` lists these for other environments (e.g. Google Colab), but this laptop's actual env was set up manually, not via `pip install -r requirements.txt`.
- Considered Ollama (quantized GGUF, much lower RAM) as an alternative runtime if memory keeps being a blocker — not adopted yet, would require rewriting `QwenVLM` to talk to Ollama's HTTP API instead of `transformers`.
- **Running on Google Colab (GPU)**: clone the repo (images are committed, no separate download needed), `pip install transformers accelerate ultralytics` (torch/torchvision already preinstalled with CUDA), then flip `DEVICE`/`DTYPE` to `"cuda"`/`"float16"` and `QWEN_LABELER` to a bigger size in `config.py` before running. Colab sessions are ephemeral — anything not committed back to the repo (or copied to Drive) is lost when the session ends; `*.pt`/`runs/` are gitignored on purpose so trained weights won't survive unless explicitly pushed/saved elsewhere.
- **Running on Kaggle (2× T4 GPUs)**: same idea as Colab, but `!pip install -U transformers accelerate ultralytics -q` (Kaggle's preinstalled `transformers` may be too old to recognize the `qwen3_5` architecture — `KeyError: 'qwen3_5'` if so; must `-U` since a bare install won't upgrade an already-present package, and the notebook kernel must be restarted afterward for the new version to load). Kaggle has no built-in file editor — modify `config.py` from a cell with `!sed -i` or `%%writefile`, not by clicking it in the file panel (that only opens a read-only preview). `QwenVLM.load()` passes `device_map="auto"` **only when `torch.cuda.is_available()`** specifically so `accelerate` can shard a model across both T4s when it doesn't fit in float32 on one (e.g. loading the 4B unquantized) — see design note below for why this is conditional, not unconditional.

### Model IDs — real names differ from the PDF spec

The assignment PDF's example code uses `"Qwen/Qwen3.5-4B-VL"`-style IDs and `AutoModelForVision2Seq`. Neither is correct for the actually-published models:
- Real HF repo IDs have **no `-VL` suffix** (Qwen3.5 is natively multimodal): `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B` — already fixed in `config.py`.
- The correct `transformers` class is **`AutoModelForMultimodalLM`**, not `AutoModelForVision2Seq` (deprecated) or `AutoModelForImageTextToText` (also wrong for this model family) — per the official model card usage example.
- `QwenVLM.load()`/`.ask()` follow the model card's official single-call pattern (`processor.apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")`), not the older two-step pattern (`tokenize=False` + separate `processor(text=..., images=...)`) used by some other VL models.

## Pipeline architecture

The intended pipeline runs in phases, each corresponding to a module in `src/`:

1. **Rename** (`src/rename_and_resize_images.py`, implemented) — normalizes raw images in `data/raw/` to `husky_000.jpg ... husky_099.jpg`, then downscales any image over `config.MAX_IMAGE_DIM` (1280px, longest side, aspect preserved) in place. Must run **before** auto-labeling, since re-running the rename part after labels exist breaks the image↔label pairing (it does a two-pass rename through temp names to avoid collisions) — the resize pass alone is idempotent/safe to re-run anytime.
2. **Auto-labeling** (`src/auto_labeling.py`, implemented, orchestrator only) — Phase 1: loops over `data/raw/` (up to `config.AUTO_LABELING_LIMIT` images), asks Qwen (`config.QWEN_LABELER`) for boxes via `PROMPT_LABELING`, writes YOLO-format `.txt` to `data/labels_auto/`, and saves a drawn-box visualization of each image to `data/labels_check/` for manual QA. Model loading, inference, response parsing, and YOLO-format conversion live in `src/utils.py` (see below); `auto_labeling.py` just calls into it.
3. **Training** (`src/train_yolo.py`, implemented) — Phase 2: fine-tunes a YOLOv8 model (`YOLO_BASE`) using the hyperparameters in `config.py` (`EPOCHS`, `IMG_SIZE`, `BATCH`, `PATIENCE`, `SEED`, `OPTIMIZER`, `LR0`, `FREEZE`, `AUGMENT`), on whichever `dataset.yaml` `DATASET_YAML_PATH` (top of the file) points to. Preceded by `src/split_dataset.py` (implemented, separate script — see design note below), which turns `data/raw/` + `data/labels_auto/` into `data/train/`+`data/test/` and `dataset.yaml`.
4. **Hybrid inference** (`src/hybrid_inference.py`, stub) — Phases 3-4: runs YOLO detection (`YOLODetector` in `src/utils.py`) then validates crops with a cascade of small Qwen VLMs (`QWEN_VALIDATORS` in `config.py`) using `PROMPT_VALIDATION`. Per the no-CLI-args convention, which validator size(s) to run should be a `config.py` variable, not a `--validator` flag (the assignment PDF's example suggests a CLI flag, but that contradicts this project's convention — see note above).
5. **Metrics** (`src/metrics.py`, stub) — computes mAP, FP/FN, latency, and P-R curves into `results/metrics/` and `results/figures/`.

All shared configuration — paths, model IDs, prompts, thresholds, hyperparameters — lives centrally in `config.py`. New scripts should import it (`import config`) rather than hardcoding paths or constants. Scripts run from the repo root add the parent dir to `sys.path` to import `config` (see `src/rename_and_resize_images.py`).

### `src/utils.py` design

Holds both Qwen/VLM stuff and the YOLO detector wrapper together in one file — an explicit project decision (previously split across `vlm_utils.py`/`yolo_utils.py`; merged on request into a single general-purpose `utils.py`). Reused by `auto_labeling.py` (Fase 1) and, once written, `hybrid_inference.py` (Fase 3/4).

Qwen section:
- `QwenVLM` class — `__init__(model_id)`, `load()`, `ask(image, prompt) -> str` (generic: runs the model and returns raw text, used both for the detection prompt and the binary Yes/No validation prompt).
- `parse_boxes(response) -> list[[x1, y1, x2, y2]]` — Qwen-specific, parses its raw text response into boxes on the 0-1000 scale. **Important:** despite `PROMPT_LABELING` literally asking for `[ymin, xmin, ymax, xmax]`, Qwen3.5 ignores that and always responds in its native "grounding" schema `[{"bbox_2d": [x1, y1, x2, y2], "label": ...}, ...]`. Verified visually by drawing both interpretations on 3 sample images (see `data/labels_check/test_orden_*.jpg`, generated by `src/test_box_order.py`) — the `bbox_2d` `[x1,y1,x2,y2]` order was the one that actually bounded the dogs correctly. `PROMPT_LABELING` was updated to ask for this native format directly instead of fighting it. Also strips any `<think>...</think>` block before searching for the JSON — see thinking-mode note below.
- `convert_to_yolo(box, class_id) -> str` — generic geometry/format conversion (0-1000 scale `[x1,y1,x2,y2]` → normalized YOLO `class_id x_center y_center width height`), not actually Qwen-specific but kept here anyway per the merge decision above. Round-trip verified by `src/test_box_order.py` (Qwen box → YOLO line → back to pixels → drawn).

**Thinking mode (`enable_thinking=False`):** Qwen3.5-4B/9B operate in "thinking mode" by default — they emit a `<think>...</think>` reasoning block *before* the actual answer (per their model cards; 0.8B defaults to non-thinking but "is more prone to entering thinking loops" per its own card). This broke Fase 1 with the 4B model: `ask()`'s `max_new_tokens=512` was getting consumed by the thinking block, so generation got cut off before ever producing the boxes JSON — looked like "the 4B can't detect dogs" but was actually a truncation/parsing bug, not a capability gap. Fixed by passing `enable_thinking=False` to `apply_chat_template()` in `ask()`, plus stripping `<think>...</think>` in `parse_boxes()` as a defense-in-depth in case a model size ignores the flag. **Confirmed working** on Kaggle (4B, GPU): correctly detected boxes on 26 consecutive images before hitting the CUDA OOM described below (unrelated to thinking mode).

**`device_map` is conditional on CUDA, not unconditionally `"auto"`:** `load()` uses `device_map = "auto" if torch.cuda.is_available() else config.DEVICE`. Originally tried unconditional `device_map="auto"` (to let `accelerate` shard a model across Kaggle's 2 GPUs) — but on this CPU-only laptop, `"auto"` made `accelerate` decide to offload part of even the tiny 0.8B model to **disk** (`"parameters are on the meta device... offloaded to the cpu and disk"`), causing heavy disk I/O; suspected contributor to a Windows BSOD during testing. Guard it so `"auto"` only kicks in when `torch.cuda.is_available()` is true (multi-GPU Kaggle sessions); otherwise falls back to `config.DEVICE` exactly like before, with no disk-offload guessing.

**`del`/`empty_cache()` in `ask()` — kept, but wasn't the real fix:** originally suspected CUDA memory fragmentation from many `ask()` calls in a loop (added `del inputs, output_ids` + `torch.cuda.empty_cache()` after decoding). Re-ran on Kaggle and it crashed at the **exact same image, same box counts** as before — proved it wasn't fragmentation (the error's "allocated by PyTorch" was ~13.7GB, genuinely in use, not reclaimable cache) and that the fix was a no-op for this bug. The `del`/`empty_cache()` call is harmless and stays, but see below for the actual cause.

**Real cause of the OOM: a few source images are enormous.** Some of the 100 raw images are up to 5616×3744px. Qwen's vision encoder turns image pixels into "visual tokens," and self-attention cost scales quadratically with token count — an oversized image can make a single `generate()` prefill call try to allocate 10+GB in one shot (confirmed: `"Tried to allocate 12.79 GiB"` crashing exactly on `husky_029.jpg`, 4700×3135px, in `_prefill`, i.e. *before* any generation even starts). Not a leak, not fragmentation — just one huge image blowing the budget on a GPU that's already mostly full of model weights. Fixed at the source: `config.MAX_IMAGE_DIM = 1280` + `rename_and_resize_images.py` now has a 3rd pass (`redimensionar_si_necesario`) that downscales (never upscales) any image over that limit, saving in place before Fase 1 ever runs. 20 of the 100 raw images needed resizing (`data/raw/` went 56MB → 37MB). This must be re-run (`python src/rename_and_resize_images.py`) on any fresh checkout of the images if it hasn't been already — check by re-running it; it's a no-op (0 resized) if already done.

YOLO section:
- `YOLODetector` class — `__init__(model_path=config.YOLO_TRAINED)`, `load()` (wraps `ultralytics.YOLO`), `detect(image) -> list[dict]` (runs inference with `config.CONF_THRESHOLD`/`IOU_THRESHOLD`, returns `{"box": (x1,y1,x2,y2) in pixels, "conf": float, "class_id": int}` per detection), `crop(image, box, padding=config.CROP_PADDING) -> Image` (crops a detection with margin, clamped to image bounds, for feeding into the Qwen validator in Fase 4).
- Not yet used anywhere — `train_yolo.py` trains via `ultralytics.YOLO` directly (training isn't `YOLODetector`'s job); `YOLODetector` is meant for `hybrid_inference.py`, still a stub.

### `src/split_dataset.py` design

Prep step before Fase 2 (`train_yolo.py`), kept as a separate script rather than folded into training — same reasoning as `rename_and_resize_images.py` being separate from `auto_labeling.py`: it's a one-time (or occasional) data-prep operation, not something that should re-run every time you train.

- `listar_pares(images_dir, labels_dir)` — pairs each image with its `.txt` by matching basename, skips (with a warning) any image missing a label.
- `dividir(pares, train_ratio, seed)` — shuffles with a fixed seed and splits by ratio. Deliberately isolated from the copy/write logic so it's the one place to change if a different splitting strategy is ever needed (e.g. k-fold) — **not implemented**, was explicitly scoped out as over-engineering for what the assignment asks (a single fixed 70/30 split).
- `copiar_pares(pares, dest_dir)` — copies into the `images/`+`labels/` structure Ultralytics expects (it discovers labels by string-replacing `images`→`labels` in the path, so this exact layout matters).
- `escribir_dataset_yaml(...)` — writes the Ultralytics-format `dataset.yaml` (`path`/`train`/`val`/`names`). `path` is written as `.` (relative to the `.yaml` file itself, which Ultralytics resolves against the file's own directory) — **not** an absolute path. Originally wrote `config.ROOT.as_posix()` (an absolute path), which worked locally but broke when the committed `dataset.yaml` (generated on this Windows machine, e.g. `c:/Users/<usuario>/...`) was cloned onto Kaggle (Linux) — training couldn't find the images since that path doesn't exist there. Assumes `yaml_path` lives directly in `config.ROOT` (true for both `DATASET_YAML` and `DATASET_YAML_FIXTURE`).
- `verificar_dataset_yaml(...)` — calls `ultralytics.data.utils.check_det_dataset()` on the generated yaml as a real validation (not just "the file was written"), catching structural problems before a training run would hit them.
- All source/dest paths and `DEST_YAML_PATH` are variables at the top of the file (no CLI args); the real `config.RAW_DIR`/`config.LABELS_AUTO_DIR`/`config.TRAIN_DIR`/`config.TEST_DIR`/`config.DATASET_YAML` block is currently **active** (the fixture block is commented out alongside it) — this has already been run for real (see Project status above). Swap back to the fixture block if testing a change to this script again.

### `src/train_yolo.py` design

Same no-CLI-args/fixture-toggle pattern as `split_dataset.py`:

- `DATASET_YAML_PATH` is a variable at the top of the file, defaulting to `config.DATASET_YAML_FIXTURE` (real one commented out alongside) — swap before running for real.
- All training hyperparameters come straight from `config.py` (`YOLO_BASE`, `EPOCHS`, `IMG_SIZE`, `BATCH`, `PATIENCE`, `SEED`, `DEVICE`, `OPTIMIZER`, `LR0`, `FREEZE`, `AUGMENT` dict unpacked as kwargs) — no hardcoded values in the script itself.
- `project`/`name` passed to `model.train()` are `config.YOLO_RUNS_DIR`/`config.YOLO_RUN_NAME` (`runs/detect/train/`, Ultralytics' own working structure, `exist_ok=True` so it doesn't accumulate `train2/`, `train3/`...). After training, the script explicitly `shutil.copy2`s `YOLO_RUNS_DIR/YOLO_RUN_NAME/weights/best.pt` to `config.YOLO_TRAINED` (`models/yolov8_finetuned_armando.pt`) — the "active" model the rest of the pipeline (`YOLODetector`) actually uses. Split this way because `YOLO_TRAINED` moved into `models/` (a real, committed deliverable folder — see gitignore exception below) which doesn't follow Ultralytics' `<project>/<name>/weights/` layout, unlike the old setup where `YOLO_TRAINED` pointed directly into `runs/`.
- Verified against the fixture dataset (2 epochs, `EPOCHS` temporarily overridden, `YOLO_TRAINED` temporarily redirected to a scratch path to avoid clobbering the real committed model — see Project status note above): downloads `yolov8s.pt`, correctly overrides `nc=80` → `nc=1`, trains, saves `best.pt`/`last.pt` at `runs/detect/train/weights/`, and copies `best.pt` to the target path correctly.

**Training instability on the real 100-epoch run (first real attempt, Kaggle GPU):** epoch 1 looked great (mAP50=0.71), but `cls_loss` then exploded (3.5 → 6.8 → 18 → 36 by epoch 8) and the model never recovered — mAP stayed near 0 for the rest of the run, early-stopping at epoch 21 with epoch 1 as the best checkpoint. Cause: `optimizer="auto"` (Ultralytics default) picked `AdamW(lr=0.002)`, too aggressive for only 70 images / 9 batches per epoch — and passing a custom `lr0` while `optimizer="auto"` is a no-op, Ultralytics logs `"ignoring 'lr0=...' ... determining automatically"` and uses its own value regardless. Fixed by setting `config.OPTIMIZER = "AdamW"` explicitly (so `LR0` actually takes effect) with `config.LR0 = 0.001` (half of what auto picked), plus `config.FREEZE = 10` (freezes the backbone, layers 0-9 per the model summary) for more stable transfer learning with so little data. Verified the params are applied correctly (fixture, 2 epochs, CPU) — `optimizer: AdamW(lr=0.001, ...)` and "Freezing layer..." printed for layers 0-9 as expected — but **not yet re-verified against the real 100-image/100-epoch run** that the instability is actually gone; that still needs to happen on Kaggle.

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
                      # also has Dataset2.zip, a backup archive of the same images;
                      # ignored by the pipeline (es_imagen() filters by extension)
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
python src/rename_and_resize_images.py
```
Renames all images in `data/raw/` to the `husky_NNN.ext` scheme, then downscales any over `config.MAX_IMAGE_DIM`. Run once, before any labeling step.

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

No build, lint, or test tooling is configured (`requirements.txt` has no build/test tooling, just runtime deps; no test suite exists). `*.pt` (model weights) and `runs/` (Ultralytics training outputs) are gitignored — never commit these, they're regenerated by `train_yolo.py`. **Exception:** `models/*.pt` is explicitly un-ignored (`!models/*.pt` in `.gitignore`) — that folder holds the team's finished/final model weights (e.g. `yolov8_finetuned_armando.pt`, `yolov8_finetuned_pedro.pt`), which *are* meant to be committed as deliverables, unlike the ephemeral `runs/detect/train/weights/` output of a given training run.

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
