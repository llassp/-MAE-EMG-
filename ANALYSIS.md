# MAE-EMG Codebase Analysis

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Summary](#2-architecture-summary)
3. [Module-by-Module Analysis](#3-module-by-module-analysis)
4. [Code Quality Assessment](#4-code-quality-assessment)
5. [Potential Issues & Bugs](#5-potential-issues--bugs)
6. [Improvement Recommendations](#6-improvement-recommendations)

---

## 1. Project Overview

This repository implements an **EMG (Electromyography) signal analysis framework** based on **Masked Autoencoders (MAE)**. The pipeline consists of two main stages:

- **Self-supervised pre-training**: An MAE learns to reconstruct masked EMG signal patches, building general-purpose feature representations without labels.
- **Downstream classification**: A two-phase fine-tuning strategy (frozen encoder, then end-to-end) trains an LSTM-based classifier on top of the pre-trained encoder to recognize 16 hand gesture classes.

The project reports **97%+ accuracy** on the [PhysioMio 16-gesture 64-channel EMG dataset](https://www.kaggle.com/datasets/theseus200719/physiomio-16-gestures-64-channel-emg-dataset) from Kaggle.

**Tech stack**: PyTorch, NumPy, SciPy, pandas, scikit-learn, matplotlib.

**License**: MIT

---

## 2. Architecture Summary

### Data Pipeline

```
Raw EMG (786527 samples, 64 channels)
  -> Bandpass filter (1-100 Hz, 4th-order Butterworth)
  -> RMS envelope (window=64, stride=10)
  -> Sliding window (64x64 frames)
  -> Min-Max normalize + Jet colormap -> RGB images (3, 64, 64)
```

### MAE Pre-training Model

| Component | Configuration |
|-----------|--------------|
| Patch size | 8x8 (64 patches per 64x64 image) |
| Encoder | 6-layer Transformer, dim=128, 4 heads, FFN=512 |
| Decoder | 2-layer Transformer Decoder, dim=64, 2 heads |
| Mask ratio | 75% |
| Loss | MSE on masked patches only |

### Downstream Classifier

| Component | Configuration |
|-----------|--------------|
| Input | 7-frame sequences of (3, 64, 64) images |
| Feature extraction | Pre-trained MAE encoder + mean pooling per frame |
| Temporal modeling | 2-layer LSTM, hidden=128, dropout=0.5 |
| Classifier | Dropout(0.5) -> Linear(128,64) -> ReLU -> Dropout(0.3) -> Linear(64,16) |
| Training | Phase 1: frozen encoder (50 epochs), Phase 2: end-to-end (150 epochs) |

---

## 3. Module-by-Module Analysis

### `model.py` -- Model Definitions

**PatchEmbed** (lines 6-20): Splits a 64x64 image into 8x8 patches and projects each patch (8x8x3=192 dims) to `embed_dim` (128) via a linear layer. The reshaping logic is correct but somewhat hard to follow due to multi-step permutation.

**MAEModel** (lines 23-183): Core MAE with encoder-decoder architecture. Key observations:

- The `forward()` method handles masking, encoding visible patches, decoding all patches, and returning predictions vs. targets for loss computation. This is well-structured.
- The `encode()` method provides a clean interface for downstream use (no masking, full patch encoding).
- `reconstruct_from_patches()` handles the inverse of patchification for visualization.
- The decoder receives **all 64 tokens** (visible + masked) assembled via `scatter_`, then uses `nn.TransformerDecoder` with visible embeddings as memory. This is a sound design choice.

**DownstreamLSTM** (lines 185-253): Wraps the pre-trained encoder with temporal modeling.

- Processes each frame independently through the encoder, pools patch features, adds temporal positional embeddings, then runs through LSTM.
- `set_encoder_grad()` and `get_param_groups()` enable the two-phase training strategy with differential learning rates.

### `dataset.py` -- Data Processing

**EMGPretrainDataset** (lines 11-107): Handles the full pipeline from raw parquet files to colormap images. The processing chain (filter -> RMS -> window -> colormap) is implemented as a sequence of private methods, which is clean.

**EMGDownstreamDataset** (lines 140-343): Extends the pipeline with label handling. Notable features:

- `_compute_rms_with_labels()` correctly aligns labels with RMS frames by taking the label at the window start position.
- `_sliding_window_with_labels()` implements a **purity filter** (>70% single-class) to discard transition windows.
- `_build_sequences()` constructs 7-frame sequences requiring **100% label consistency** across all frames.
- `build_splits()` uses stratified train/val/test splitting (70/15/15).

### `train.py` -- MAE Pre-training Script

Implements a standard training loop with:
- Custom `WarmupCosineScheduler` for learning rate warmup + cosine decay.
- Gradient clipping (max_norm=1.0).
- Best model checkpointing based on validation loss.
- Periodic checkpoint saving every 20 epochs.
- Resume from checkpoint support.

### `downstream_train.py` -- Downstream Classification

Implements the two-phase training strategy:
- **Phase 1**: Frozen encoder, train only the classifier head with CosineAnnealingLR.
- **Phase 2**: Unfrozen encoder, end-to-end fine-tuning with differential learning rates (encoder: 1e-5, head: 5e-5) and warmup cosine scheduling.
- Both phases include early stopping.
- Final evaluation on test set with per-class accuracy reporting.
- Saves confusion matrix and training curves.

### `visualize.py` -- Visualization Utilities

Three main functions:
- `reconstruct_visualization()`: Shows original / masked / reconstructed images side by side.
- `tsne_visualization()`: t-SNE plot of encoder features colored by class.
- `confusion_matrix_visualization()` and `training_curve_visualization()`: Standard evaluation plots.

### `test_only.py` -- Standalone Inference

Loads a trained downstream model and runs evaluation on the test set, producing per-class accuracy, confusion matrix, and training curve plots.

---

## 4. Code Quality Assessment

### Strengths

1. **Clear separation of concerns**: Model, data, training, evaluation, and visualization are well-separated into distinct modules.
2. **Comprehensive documentation**: Two detailed Chinese-language technical reports explain the theory, architecture, and design decisions thoroughly.
3. **Reproducibility**: Seeds are set for random operations; checkpoints include full state for resumption.
4. **Training best practices**: Gradient clipping, warmup scheduling, early stopping, differential learning rates, and two-phase fine-tuning are all implemented.
5. **Good README**: Clear usage instructions and file descriptions.

### Weaknesses

1. **No unit tests**: There are zero test files. Core logic (patching, RMS computation, label alignment) would benefit from automated testing.
2. **No configuration file**: All hyperparameters are either hardcoded or scattered across argparse defaults. A centralized config (YAML/JSON) would improve manageability.
3. **Duplicated code**: The bandpass filter, RMS computation, sliding window, and colormap logic are duplicated between `EMGPretrainDataset`, `EMGDownstreamDataset`, and `visualize.py::tsne_visualization()`. These should be shared utility functions.
4. **No type hints**: Function signatures lack type annotations, making the API harder to understand without reading the implementation.
5. **No logging in dataset**: Data processing steps are silent; adding progress indicators or logging would help with debugging and monitoring.
6. **Missing `__all__` exports**: The `__init__.py` is empty, so module-level imports are undefined.

---

## 5. Potential Issues & Bugs

### 5.1 Colormap Normalization Inconsistency

In `visualize.py` line 163, the t-SNE visualization function applies an **extra division by 255**:

```python
images = images / 255.0  # line 163 in tsne_visualization()
```

However, `matplotlib.cm.jet()` already returns values in [0, 1], so this division produces values in [0, 1/255], effectively near-zero. This means the t-SNE features are computed on nearly-black images, which would produce poor feature quality. The pre-training and downstream datasets do **not** apply this division, so there is an inconsistency.

**Severity**: High -- t-SNE visualization will produce misleading results.

### 5.2 `_build_sequences` Uses Overlapping Sequences Despite Documentation

The technical report states "non-overlapping sequences" (non-overlapping step), but `_build_sequences()` iterates with `step=1`:

```python
for i in range(len(images) - sequence_length + 1):  # step = 1
```

This creates heavily overlapping sequences (6 out of 7 frames shared between adjacent sequences). The comment in the technical report ("non-overlapping") contradicts the implementation. While this may work well in practice (more training data), it creates data leakage risk between train/val/test splits since nearby sequences share most of their frames.

**Severity**: Medium -- potential data leakage in evaluation metrics.

### 5.3 Missing `import numpy as np` Placement

In `downstream_train.py`, `numpy` is imported **twice** -- once implicitly via other modules and once explicitly at line 159:

```python
import numpy as np  # line 159, after function definitions
```

This works in Python but is unconventional. The `test()` function at line 113 uses `np.array()` and `np.mean()`, but `numpy` is not imported until line 159. This only works because the function is not called until after line 159 executes. If someone were to call `test()` before `main()`, it would raise a `NameError`.

**Severity**: Low -- works in practice but fragile.

### 5.4 Phase 1 Only Trains Classifier, Not LSTM

In `downstream_train.py` line 236-239, Phase 1 optimizer only includes classifier parameters:

```python
optimizer_p1 = torch.optim.AdamW(
    model.classifier.parameters(),  # Only classifier!
    lr=args.phase1_lr,
    weight_decay=0.01
)
```

But the LSTM and temporal position embeddings also need training. Since the encoder is frozen, the LSTM receives fixed features but its own weights are **not being updated** in Phase 1 because they are not in the optimizer. This means Phase 1 only trains the final linear layers while the LSTM stays at random initialization.

**Severity**: High -- LSTM is not trained in Phase 1, which likely reduces Phase 1 accuracy and makes the Phase 1 "validation check" less meaningful.

### 5.5 `WarmupCosineScheduler` Duplication

The `WarmupCosineScheduler` class is defined in both `train.py` (single learning rate) and `downstream_train.py` (per-group learning rates). These are slightly different implementations of the same concept. This duplication creates maintenance burden.

**Severity**: Low -- code smell, not a bug.

### 5.6 Hardcoded `in_channels=3` in `reconstruct_from_patches`

In `model.py` line 171, the channel count is hardcoded:

```python
C = 3  # Hardcoded
```

While this matches the current dataset (RGB colormapped images), it breaks the flexibility provided by the `in_channels` constructor parameter.

**Severity**: Low -- only affects non-RGB use cases.

### 5.7 No Validation of Data Directory

Neither `EMGPretrainDataset` nor `EMGDownstreamDataset.build_splits()` validates that the data directory exists or contains parquet files before processing. If the directory is empty or missing, the error messages would be cryptic (e.g., empty array stacking).

**Severity**: Low -- user experience issue.

---

## 6. Improvement Recommendations

### Priority 1: Bug Fixes

1. **Fix the `/255.0` bug in `tsne_visualization()`** -- Remove the erroneous normalization on line 163 of `visualize.py`.
2. **Include LSTM + temporal position embed parameters in Phase 1 optimizer** -- Change `model.classifier.parameters()` to include all non-encoder trainable parameters.
3. **Address data leakage in sequence building** -- Either use truly non-overlapping sequences (step=7) or ensure splits are done at a higher level (e.g., per-file or per-session splits) to prevent leakage.

### Priority 2: Code Quality

4. **Extract shared data processing utilities** -- Create a `utils.py` module containing `bandpass_filter()`, `compute_rms()`, `sliding_window()`, and `normalize_and_colormap()` to eliminate duplication across `dataset.py` and `visualize.py`.
5. **Add type hints** -- Annotate all public function signatures with proper types.
6. **Create a centralized config** -- Use a dataclass or YAML file for hyperparameters instead of scattered argparse defaults and inline dicts.
7. **Unify `WarmupCosineScheduler`** -- Merge the two implementations into a single class in a shared module.

### Priority 3: Robustness

8. **Add unit tests** -- Test patch embedding dimensions, RMS computation correctness, label alignment, sequence building purity filtering, and model forward pass shapes.
9. **Add input validation** -- Check data directory existence, parquet file presence, and expected column names early in the pipeline.
10. **Add `num_workers` to DataLoaders** -- Currently all DataLoaders use `num_workers=0` (main process only), which can be a bottleneck.
11. **Use `torch.compile` or mixed precision** -- For larger-scale experiments, adding AMP (automatic mixed precision) support would speed up training significantly.

### Priority 4: Documentation & Usability

12. **Translate technical reports to English** -- Or add English summaries for international accessibility.
13. **Add a `data/` directory with download instructions** -- The README mentions a Kaggle dataset but does not provide download/setup scripts.
14. **Add `.gitignore`** -- Exclude `checkpoints/`, `logs/`, `__pycache__/`, and other generated files.
15. **Populate `__init__.py`** -- Export the key classes (`MAEModel`, `DownstreamLSTM`, `EMGPretrainDataset`, `EMGDownstreamDataset`) for clean imports.

---

*Analysis generated on 2026-03-27. Based on the full codebase at the `main` branch.*
