# Mechanical Parts Semantic Segmentation

A deep learning project for **multi-class semantic segmentation of mechanical components** in cluttered industrial images.

The objective is to predict the class of **every pixel** in a \(384\times384\) RGB image containing multiple overlapping mechanical parts.

The project focuses on building the complete segmentation pipeline:

\[
\text{Image}
\rightarrow
\text{Augmentation}
\rightarrow
\text{U-Net++}
\rightarrow
\text{Pixel-wise logits}
\rightarrow
\text{Class prediction}
\rightarrow
\text{RLE submission}
\]

The implementation uses **PyTorch**, **Segmentation Models PyTorch**, and **Albumentations**, with all model weights initialized from scratch.

---

## 1. Problem

This is a **7-class semantic segmentation** problem:

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | `hex_nut` |
| 2 | `washer` |
| 3 | `bolt` |
| 4 | `ball_bearing` |
| 5 | `spring` |
| 6 | `o_ring` |

Each image contains instances of the six foreground classes.

The challenge is not simply recognizing the objects. The model must determine their **exact spatial extent at pixel level**, including thin boundaries, holes, overlapping parts, and visually similar circular structures.

---

## 2. Dataset

The dataset contains:

- **2,000 labelled training images**
- **500 unlabelled test images**
- Image resolution: **384 × 384**
- Image format: **RGB PNG**
- Six foreground classes + background

The segmentation masks contain integer class IDs directly:

```python
mask = np.array(Image.open(mask_path))
```

The masks must not be converted to RGB because the stored pixel values themselves represent the segmentation labels.

---

## 3. Why This Is Difficult

The six object categories have substantially different geometric structures.

### Hex nut

A polygonal outer boundary with a central hole.

### Washer

A thin circular ring with a central hole.

### Bolt

A hexagonal head connected to an elongated threaded shank.

### Ball bearing

A compact approximately spherical object with characteristic highlights.

### Spring

A thin helical structure containing repeated loops.

### O-ring

A circular object with a rounded tube-like cross-section.

This creates several segmentation challenges:

- small foreground regions;
- thin structures;
- holes inside objects;
- similar circular shapes;
- partial occlusion;
- objects touching or overlapping;
- large differences in object geometry.

---

# 4. Evaluation Metric

The primary metric is the **Dice coefficient**.

For a predicted binary mask \(X\) and ground-truth mask \(Y\):

\[
Dice(X,Y)
=
\frac{2|X\cap Y|}
{|X|+|Y|}
\]

The score ranges from:

\[
0 \leq Dice \leq 1
\]

where \(1\) represents perfect overlap.

The final metric is the mean Dice across every image–class pair:

\[
Score =
\frac{1}{6N}
\sum_{i=1}^{N}
\sum_{c=1}^{6}
Dice(X_{i,c},Y_{i,c})
\]

This makes region overlap much more important than simple pixel accuracy.

---

# 5. Model Architecture

## U-Net++

The main segmentation architecture is **U-Net++**.

Two encoder configurations were experimentally trained:

```text
U-Net++ + ResNet-34
U-Net++ + EfficientNet-B1
```

Both models were trained with:

```python
encoder_weights=None
```

so no pretrained encoder weights were used.

The final segmentation head predicts:

\[
7
\]

classes for every pixel.

### High-level architecture

```text
                 Input Image
                384 × 384 × 3
                       │
                       ▼
              ┌─────────────────┐
              │     Encoder     │
              │ ResNet-34 /     │
              │ EfficientNet-B1 │
              └────────┬────────┘
                       │
                Multi-scale
                  features
                       │
                       ▼
              ┌─────────────────┐
              │     U-Net++     │
              │     Decoder     │
              │                 │
              │ Nested Skip     │
              │ Connections     │
              └────────┬────────┘
                       │
                       ▼
                7-channel logits
                       │
                       ▼
              Per-pixel argmax
                       │
                       ▼
               Segmentation Mask
```

---

# 6. Why U-Net++?

Semantic segmentation requires both:

1. **semantic information** — what object is present?
2. **spatial information** — exactly which pixels belong to it?

The encoder progressively learns increasingly abstract representations:

\[
\text{edges}
\rightarrow
\text{textures}
\rightarrow
\text{parts}
\rightarrow
\text{object structure}
\]

while the decoder reconstructs the spatial resolution required for pixel-level prediction.

U-Net++ extends the standard encoder-decoder idea with **nested skip connections**, allowing features from different resolutions to be repeatedly fused.

This is particularly useful for this dataset because objects contain:

- thin structures;
- sharp boundaries;
- holes;
- elongated components;
- small foreground regions.

---

# 7. Training Configuration

The main configuration used in the experiments was:

| Parameter | Value |
|---|---:|
| Image size | \(384\times384\) |
| Batch size | 8 |
| Epochs | 100 |
| Early stopping patience | 15 |
| Initial learning rate | \(1\times10^{-3}\) |
| Optimizer | AdamW |
| Weight decay | \(1\times10^{-4}\) |
| LR scheduler | Cosine Annealing |
| Random seed | 42 |
| Precision | Automatic mixed precision |
| Device | CUDA when available |

The training/validation split uses:

\[
85\% / 15\%
\]

of the labelled images.

A fixed random seed of `42` is used for reproducibility.

---

# 8. Weight Initialization

Because pretrained weights were not used, the convolutional and linear layers are explicitly initialized using **Kaiming Normal initialization**:

```python
nn.init.kaiming_normal_(
    module.weight,
    mode="fan_in",
    nonlinearity="relu"
)
```

The intuition behind Kaiming initialization is to choose the initial weight variance so that activations remain numerically stable through ReLU-based networks.

For a layer with \(n\) input connections, the initialization is approximately based on:

\[
Var(W) \approx \frac{2}{n}
\]

This helps avoid excessively large or vanishing activations at the beginning of training.

---

# 9. Data Augmentation

The training pipeline applies geometric and photometric augmentation.

### Geometric transformations

```text
HorizontalFlip
VerticalFlip
RandomRotate90
ShiftScaleRotate
```

The `ShiftScaleRotate` operation allows the model to see variations in:

- translation;
- scale;
- rotation.

### Appearance transformations

```text
RandomBrightnessContrast
HueSaturationValue
```

These make the model less dependent on a specific illumination or colour configuration.

### Noise / blur

One of:

```text
MotionBlur
GaussianNoise
```

may be applied.

### Coarse dropout

Random regions can also be removed:

```text
1–8 holes
8–32 px height
8–32 px width
```

This encourages the network to use broader contextual information instead of depending on a small local feature.

---

# 10. Normalization

Images are normalized using:

\[
\mu=(0.485,0.456,0.406)
\]

\[
\sigma=(0.229,0.224,0.225)
\]

so each channel is transformed approximately as:

\[
x' = \frac{x-\mu}{\sigma}
\]

The validation pipeline uses normalization but does not apply random augmentation.

---

# 11. Loss Function

The final training loss is a 50/50 combination of multiclass Dice loss and cross-entropy:

\[
\mathcal{L}
=
0.5\mathcal{L}_{Dice}
+
0.5\mathcal{L}_{CE}
\]

### Dice loss

Dice loss directly encourages overlap between predicted and target segmentation regions.

Conceptually:

\[
\mathcal{L}_{Dice}
\approx
1-Dice
\]

### Cross-entropy

For a pixel whose ground-truth class is \(y\), cross-entropy penalizes low probability assigned to the correct class:

\[
\mathcal{L}_{CE}
=
-\log P(y\mid x)
\]

### Why combine them?

The two losses emphasize different properties.

```text
Cross Entropy
     ↓
Better class discrimination
     +
Dice Loss
     ↓
Better region overlap
     =
Combined segmentation objective
```

This is especially useful when foreground regions occupy relatively small portions of the image.

---

# 12. Optimization

AdamW is used for parameter updates.

At a high level, gradient-based optimization follows:

\[
\theta_{t+1}
=
\theta_t
-
\eta_t
\nabla_\theta\mathcal{L}
\]

where:

- \(\theta\) = model parameters
- \(\eta_t\) = learning rate
- \(\mathcal{L}\) = segmentation loss

AdamW additionally applies decoupled weight decay.

The learning rate follows a cosine annealing schedule:

\[
\eta_t
\approx
\eta_{\min}
+
\frac{1}{2}
(\eta_{\max}-\eta_{\min})
\left(
1+\cos\frac{\pi t}{T}
\right)
\]

This provides relatively large updates early in training and increasingly smaller updates as training approaches the end.

---

# 13. Mixed Precision Training

Training uses automatic mixed precision:

```python
with torch.amp.autocast('cuda'):
    outputs = model(images)
    loss = criterion(outputs, masks)
```

and gradient scaling through:

```python
torch.amp.GradScaler('cuda')
```

The purpose is to reduce GPU memory usage and improve computational throughput while maintaining numerical stability through gradient scaling.

---

# 14. Model Selection

Two encoder configurations were trained independently:

```python
ENCODERS = [
    "efficientnet-b1",
    "resnet34"
]
```

Each model has its own:

- optimizer;
- scheduler;
- training history;
- best checkpoint.

A checkpoint is saved whenever validation Dice improves.

Early stopping is triggered after 15 consecutive epochs without improvement.

This prevents unnecessary training once the validation metric stops improving.

---

# 15. Test-Time Augmentation

Validation and inference use **Test-Time Augmentation (TTA)**.

The original image is evaluated together with transformed versions:

```text
Original
Horizontal flip
Vertical flip
Horizontal + vertical flip
90° rotation
270° rotation
```

The predictions are transformed back to the original coordinate system and averaged.

Mathematically, if \(P_i(x)\) is the probability map produced by augmentation \(i\), the TTA prediction is:

\[
P_{TTA}(x)
=
\frac{1}{6}
\sum_{i=1}^{6}P_i(x)
\]

The final segmentation is:

\[
\hat{Y}(x)
=
\arg\max_c P_{TTA,c}(x)
\]

The motivation is that the object identity should remain unchanged under these transformations, while averaging predictions can reduce prediction variance.

---

# 16. Model Ensemble

The two independently trained models are combined.

For validation, the ensemble used:

\[
P_{ensemble}
=
0.4P_{EfficientNet}
+
0.6P_{ResNet}
\]

followed by:

\[
\hat{Y}
=
\arg\max_c P_{ensemble,c}
\]

This gives the ResNet-34 model slightly greater influence in the validation ensemble.

For the final test inference, the two probability maps were averaged equally:

\[
P_{test}
=
\frac{P_{EfficientNet}+P_{ResNet}}{2}
\]

The ensemble idea is based on reducing correlated errors: if two models make different mistakes, averaging their probability distributions can produce a more robust prediction than either model alone.

---

# 17. Small-Class Recovery Heuristic

An additional inference rule was implemented for foreground classes that were completely absent from the initial `argmax` prediction.

For each missing class:

1. inspect the class probability map;
2. select pixels with probability greater than `0.20`;
3. if more than 150 pixels satisfy the threshold, retain the 150 highest-probability pixels;
4. assign those pixels to the missing class.

Conceptually:

\[
\text{if predicted area}_c = 0
\]

then examine:

\[
P(y=c\mid x)>0.20
\]

and recover a small set of high-confidence candidate pixels.

This was motivated by the fact that every foreground class is expected to occur in each image. The heuristic attempts to prevent a class from disappearing entirely due to the hard `argmax` decision.

This is a deliberately task-specific post-processing step and should be evaluated carefully because forcing pixels into a missing class can also introduce false positives.

---

# 18. Observed Training Results

The recorded later-stage training behaviour was:

| Epoch | Loss | Dice | Learning Rate |
|---:|---:|---:|---:|
| 89 | 0.0327 | 0.9795 | 3.0e-5 |
| 90 | 0.0328 | 0.9795 | 2.4e-5 |
| 91 | 0.0324 | 0.9795 | 2.0e-5 |
| 92 | 0.0322 | 0.9795 | 1.6e-5 |
| 93 | 0.0328 | 0.9795 | 1.2e-5 |
| 94 | 0.0323 | 0.9795 | 9.0e-6 |
| 95 | 0.0320 | 0.9795 | 6.0e-6 |
| 96 | 0.0318 | **0.9796** | 4.0e-6 |
| 97 | 0.0319 | 0.9795 | 2.0e-6 |
| 98 | 0.0328 | 0.9795 | 1.0e-6 |
| 99 | 0.0321 | **0.9796** | ~0 |
| 100 | 0.0321 | **0.9796** | ~0 |

### Convergence observation

From epoch 89 onward, the Dice score remained within:

\[
0.9795 \leq Dice \leq 0.9796
\]

while the learning rate decreased toward zero.

This indicates that the optimization had entered a highly stable region by the final stage.

The improvement from:

\[
0.9795 \rightarrow 0.9796
\]

is extremely small, suggesting that the final epochs primarily performed fine adjustment rather than learning a substantially different representation.

---

# 19. Results Interpretation

The strongest value recorded in the training/validation log was:

\[
\boxed{Dice = 0.9796}
\]

with final recorded loss:

\[
\boxed{Loss = 0.0321}
\]

The important point is not only the absolute Dice value, but the behaviour of the optimization:

```text
High Dice
   ↓
Very small improvement
   ↓
Learning rate approaches zero
   ↓
Metric stabilizes
```

This suggests that the selected architecture and training configuration were able to learn a strong segmentation representation from the available labelled data.

**The 0.9796 value should be interpreted as the recorded validation/training-loop Dice, not as an independently verified hidden-test score.**

---

# 20. RLE Encoding

The final masks are converted to **Run Length Encoding (RLE)**.

A binary mask is first flattened using column-major ordering:

```python
px = np.asarray(mask, np.uint8).T.flatten()
```

The resulting sequence is converted into:

```text
start_position run_length
```

pairs.

For example:

```text
2 2
```

represents:

```text
pixel 2
pixel 3
```

The ordering matters because the evaluator reconstructs the 2D mask using Fortran-style ordering.

The implementation also includes an RLE decoder so that the encoding process can be verified.

---

# 21. Submission Generation

For every test image and every foreground class:

```python
mask = predicted_label_map == class_id
```

The binary mask is converted to RLE and written to:

```text
ImageId_ClassId,EncodedPixels
```

The submission therefore contains:

\[
500\times6=3000
\]

prediction rows.

The complete pipeline is:

```text
Test Image
    ↓
EfficientNet-B1
    ↓
TTA probability maps
    │
    ├──────────────┐
    ↓              ↓
ResNet-34       Ensemble
    ↓              │
    └──────────────┘
           ↓
     Pixel-wise argmax
           ↓
   Missing-class recovery
           ↓
     Class segmentation
           ↓
          RLE
           ↓
    submission.csv
```

---

# 22. Experimental Methodology

The experiments were structured around controlled comparisons rather than changing every component simultaneously.

The primary architectural experiment was:

```text
U-Net++ + EfficientNet-B1
vs.
U-Net++ + ResNet-34
```

After training the individual models, their probability outputs were combined through an ensemble.

This provides three levels of comparison:

```text
Single model
     ↓
Single model + TTA
     ↓
Multi-model ensemble + TTA
```

This makes it possible to study whether improvements come from:

- architecture;
- prediction averaging;
- geometric test-time augmentation;
- ensemble diversity;
- task-specific post-processing.

---

# 23. Dataset and GitHub Policy

The original training and test images are **not included in this GitHub repository**.

This is intentional.

The dataset is large, unnecessary for reproducing the source code itself, and was provided specifically for the associated competition/task. The repository therefore contains the **code, experiment configuration, methodology, and results**, while the dataset remains external.

Recommended repository structure:

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── segmentation_experiment.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── inference.py
│   └── rle.py
│
├── checkpoints/
│   └── best_models/
│
├── outputs/
│   ├── predictions/
│   └── submission.csv
│
└── experiments/
    └── training_logs/
```

### Dataset setup

After obtaining the dataset through the original source, configure its local path in the training script:

```python
DATASET_DIR = "/path/to/Dataset"
```

Expected layout:

```text
Dataset/
├── train/
│   ├── images/
│   └── masks/
│
├── test/
│   └── images/
│
├── train.csv
├── metadata.csv
└── sample_submission.csv
```

Do **not** commit the raw dataset, generated masks, or large model checkpoints unless there is a specific reason to distribute them and you have permission to do so.

A `.gitignore` should therefore include entries such as:

```gitignore
# Dataset
Dataset/
data/
train/
test/

# Model checkpoints
*.pth
*.pt
*.ckpt

# Generated outputs
outputs/
submission.csv

# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/
```

The repository remains useful without the data because the objective is to document and reproduce the **methodology and implementation**. Anyone with legitimate access to the dataset can point `DATASET_DIR` to their local copy.

---

# 23. Project Structure

A clean repository layout is:

```text
.
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── segmentation_experiment.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── inference.py
│   └── rle.py
│
├── checkpoints/
│   ├── best_unet_resnet34.pth
│   └── best_unet_efficientnet-b1.pth
│
├── outputs/
│   ├── predictions/
│   └── submission.csv
│
└── experiments/
    └── training_logs/
```

The original dataset should generally remain outside the Git repository because of its size.

---

# 24. Reproducibility

The experiments use:

```python
SEED = 42
```

and explicitly seed:

```text
Python random
NumPy
PyTorch
CUDA
```

when CUDA is available.

The models are initialized from scratch:

```python
encoder_weights=None
```

followed by Kaiming initialization of convolutional and linear layers.

No external training dataset is required by the implementation.

---

# 25. Engineering Lessons

This project demonstrates that segmentation performance depends on much more than the network architecture.

Several parts of the pipeline are equally important:

### Correct mask loading

The mask contains class IDs, not RGB semantic information.

### Correct spatial transformations

Image and mask transformations must remain geometrically synchronized.

### Correct loss

The objective should reflect the segmentation problem rather than relying only on generic classification loss.

### Correct validation

Model selection should use a fixed validation split and the same metric used for the actual task.

### Correct inference

TTA and ensembling operate on probability maps before the final `argmax`.

### Correct RLE

A mathematically correct segmentation can still produce an incorrect submission if the flattening order is wrong.

---

# 26. Future Experiments

The current pipeline provides several natural directions for further experimentation.

## Architecture

Compare:

- U-Net
- U-Net++
- DeepLab
- different U-Net++ encoder sizes

## Loss functions

Evaluate:

\[
\mathcal{L}
=
\lambda_1\mathcal{L}_{Dice}
+
\lambda_2\mathcal{L}_{CE}
\]

for different values of \(\lambda_1,\lambda_2\).

The current experiment uses:

\[
\lambda_1=\lambda_2=0.5
\]

## Augmentation

Perform controlled ablations of:

- geometric augmentation;
- colour augmentation;
- blur/noise;
- coarse dropout.

## TTA

Compare:

```text
No TTA
6-way TTA
Different transformation subsets
```

## Ensemble weights

Instead of fixed weights:

\[
0.4/0.6
\]

evaluate different combinations on the same validation split.

## Post-processing

The missing-class recovery heuristic can be studied independently to determine whether it improves Dice or simply increases false-positive regions.

---

# 27. Key Takeaways

### Task

Multi-class semantic segmentation of six mechanical-part categories.

### Data

\[
2000\ \text{training images}
\]

\[
500\ \text{test images}
\]

\[
384\times384\ \text{RGB}
\]

### Architecture

\[
\boxed{\text{U-Net++}}
\]

with:

```text
ResNet-34
EfficientNet-B1
```

encoder experiments.

### Loss

\[
\boxed{
0.5\,DiceLoss + 0.5\,CrossEntropyLoss
}
\]

### Optimization

```text
AdamW
+ Cosine Annealing
+ Mixed Precision
```

### Inference

```text
TTA
+
Model Ensemble
+
Missing-class recovery
```

### Official leaderboard result

The final submitted solution achieved:

| Metric | Score |
|---|---:|
| **Private leaderboard Dice** | **0.97960** |
| Public leaderboard Dice | 0.98096 |

The **private score is the final score** and is therefore the primary result reported for this project.

### Main lesson

A strong segmentation system is a complete pipeline:

\[
\boxed{
\text{Data}
\rightarrow
\text{Augmentation}
\rightarrow
\text{Architecture}
\rightarrow
\text{Loss}
\rightarrow
\text{Optimization}
\rightarrow
\text{TTA}
\rightarrow
\text{Ensemble}
\rightarrow
\text{Post-processing}
\rightarrow
\text{RLE}
}
\]

---

## Author's Note

This project was developed as an end-to-end study of dense prediction using modern deep learning techniques.

The emphasis is on understanding **why each component is used**, measuring its effect, and maintaining a reproducible experimental pipeline rather than treating the segmentation model as a black box.
