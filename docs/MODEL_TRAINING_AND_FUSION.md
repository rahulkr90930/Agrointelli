# 🌿 AgroIntelli: Model Training & Multimodal Architecture Fusion

This document details the complete machine learning lifecycle, training protocol, and multimodal fusion mechanisms implemented in **AgroIntelli**.

---

## 1. Dataset Architecture & Pipeline

### A. Fused Dataset Composition
AgroIntelli bridges the gap between controlled laboratory benchmark data and real-world outdoor agricultural environments by fusing two distinct image corpuses:
* **PlantVillage Dataset**: Laboratory-grade images captured under controlled lighting with uniform backgrounds (isolated leaves).
* **PlantDoc Dataset**: In-the-wild agricultural field images featuring varied solar illumination, complex soil/foliage backgrounds, motion blur, and overlapping leaves.
* **Volume & Taxonomy**:
  * **62,669 total verified images** across **14 distinct crop ecosystems** (Tomato, Potato, Corn, Pepper, Apple, Grape, Strawberry, etc.).
  * Unified into a standardized canonical taxonomy of **39 target classes** (disease and healthy states) documented in `backend/models/class_index_map.json`.

### B. High-Throughput Streaming Pipeline
* **Streaming Generator**: An infinite-generator `tf.data.Dataset` stream (`.repeat().batch(16).prefetch(tf.data.AUTOTUNE)`) eliminates generator exhaustion and prevents worker memory starvation during long training runs.
* **In-Graph GPU Data Augmentation**: To resist outdoor weather and camera variations, training batches undergo dynamic in-graph spatial and photometric transformations:
  * `RandomFlip("horizontal_and_vertical")`
  * `RandomRotation(0.15)`
  * `RandomZoom(0.15)`
  * `RandomContrast(0.15)`

---

## 2. Model Architecture & Multimodal Late Fusion

Plant diseases rarely occur in a vacuum; pathogen sporulation and incubation are tightly coupled to environmental microclimates. AgroIntelli implements a **Multimodal Late Fusion Network** combining visual deep learning features with normalized 4D microclimate telemetry.

```
+------------------------------------+       +------------------------------------+
|  Leaf Image Input (224x224x3 RGB)   |       |   Weather Telemetry Vector (4-D)   |
|  [Normalized / Preprocessed]       |       |   [Temp, Humidity, Rain, Wind]     |
+-----------------+------------------+       +-----------------+------------------+
                  |                                            |
                  v                                            v
    Data Augmentation (GPU)                         Dense Layer (32, ReLU)
                  |                                            |
                  v                                            v
  MobileNetV3 / EfficientNet Backbone               Dense Layer (16, ReLU)
                  |                                            |
                  v                                            |
      GlobalAveragePooling2D                                   |
                  |                                            |
                  v                                            |
  Dense (256, ReLU) + BatchNorm + Dropout(0.2)                 |
                  |                                            |
                  +---------------------+----------------------+
                                        |
                                        v
                       Multimodal Concatenation (272-D)
                                        |
                                        v
                            Dense (128, ReLU)
                                        |
                                        v
                              Dropout (0.35)
                                        |
                                        v
                      Dense Softmax (39 Classes)
```

### Fusion Mechanics
1. **Vision Stream**:
   * Pretrained backbones (`MobileNetV3Small` / `EfficientNet-B0`, `weights="imagenet"`).
   * Feature pooling through `GlobalAveragePooling2D` yields spatial embedding.
   * Projected to a 256-dimensional dense embedding with Batch Normalization and 20% Dropout.
2. **Environmental Stream**:
   * Accepts normalized microclimate vectors: $\text{Temperature}$, $\text{Relative Humidity}$, $\text{Precipitation}$, and $\text{Wind Speed}$.
   * Passes through two Dense layers ($32 \rightarrow 16$, ReLU) to learn non-linear microclimate correlations.
3. **Late Fusion**:
   * Visual and weather embeddings are merged via `layers.concatenate([vision_features, weather_features])`, yielding a **272-dimensional joint multimodal representation**.
4. **Classification Head**:
   * Fully-connected Dense layer (128 units, ReLU) followed by 35% Dropout and a 39-unit Softmax classifier.

---

## 3. Two-Stage Transfer Learning Regimen

The training protocol was executed across two distinct phases to preserve pretrained transfer weights while tuning disease-specific representations:

### Stage 1: Classification Head Warmup (5 Epochs)
* **Backbone Status**: Frozen (`trainable = False`).
* **Objective**: Stabilize and train the randomized weights of the vision projection, weather layers, and fusion head.
* **Optimizer**: Adam ($\text{LR} = 1 \times 10^{-3}$).
* **Loss Function**: Categorical Cross-Entropy with **Label Smoothing ($0.1$)** to curb overconfidence on noisy outdoor field annotations.
* **Result**: Rapid convergence to **92.40% train accuracy** and **94.72% validation accuracy** by Epoch 5.

### Stage 2: Backbone Fine-Tuning (15 Epochs)
* **Backbone Status**: Top layers unfrozen (`trainable = True`).
* **Objective**: Adapt higher-order visual feature maps to fine-grained botanical patterns (chlorotic halos, fungal pustules, necrotic spots).
* **Optimizer & Schedule**: Adam with reduced learning rate ($\text{LR} = 1 \times 10^{-4}$), dynamic learning rate reduction (`ReduceLROnPlateau`), and `EarlyStopping` monitoring `val_loss`.
* **Final Metrics**:
  * **Train Accuracy**: **98.67%** (Top-3: **99.79%**)
  * **Validation Accuracy**: **98.31%** (Top-3: **99.69%**)
  * **Validation Loss**: $0.7525$

---

## 4. Test-Time Inference Ensemble

At runtime (`backend/inference.py`), AgroIntelli implements a two-fold merging strategy to ensure high diagnostic reliability:

### A. Dual-Pass Field Illumination Feature Ensemble
To mitigate outdoor lighting extremes (harsh sunlight, leaf glare, shadow):
1. **Pass 1 (Enhanced)**: The image is enhanced in LAB color space using Bilateral Filtering (edge-preserving noise reduction) and CLAHE (Contrast Limited Adaptive Histogram Equalization).
2. **Pass 2 (Raw RGB)**: The native resized RGB image is evaluated concurrently.
3. **Soft-Voting Fusion**:
   $$\mathbf{P}_{\text{final}} = 0.65 \times \mathbf{P}_{\text{enhanced}} + 0.35 \times \mathbf{P}_{\text{raw}}$$
   This weights edge-enhanced lesion patterns at 65% while preserving 35% natural color context.

### B. Plant-Conditioned Logit Normalization
When a user selects or pre-filters a crop species (e.g. *Tomato*), the inference engine dynamically masks cross-species classes and re-normalizes the probability distribution over valid species classes, preventing cross-species diagnostic errors.
