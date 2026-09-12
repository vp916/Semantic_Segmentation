# Mechanical Parts Semantic Segmentation

A deep learning project for **multi-class semantic segmentation of mechanical components** in cluttered industrial images.

The goal is to predict the class of every pixel in a 384 × 384 RGB image containing multiple mechanical parts, including overlapping and visually similar objects.

The final solution achieved a **private leaderboard Dice score of 0.97960**.

---

## Overview

This project implements an end-to-end semantic segmentation pipeline using:

- **PyTorch**
- **Segmentation Models PyTorch**
- **Albumentations**
- **U-Net++**
- **ResNet-34**
- **EfficientNet-B1**
- **AdamW**
- **Cosine Annealing**
- **Mixed Precision Training**
- **Test-Time Augmentation (TTA)**
- **Model Ensembling**

All model weights were initialized from scratch. No pretrained encoder weights were used.

---

## 1. Problem

The task is a 7-class semantic segmentation problem:

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | `hex_nut` |
| 2 | `washer` |
| 3 | `bolt` |
| 4 | `ball_bearing` |
| 5 | `spring` |
| 6 | `o_ring` |

For an input image:

```text
384 × 384 × 3
```

the model produces:

```text
384 × 384 × 7
```

logits.

For each pixel, the class with the highest predicted probability is selected as the final segmentation label.

### Object characteristics

The classes have substantially different geometric structures:

- **Hex nut** — six-sided outer boundary with a central hole
- **Washer** — thin circular ring with a central hole
- **Bolt** — hexagonal head with a threaded shank
- **Ball bearing** — solid spherical component
- **Spring** — thin helical structure with repeating loops
- **O-ring** — circular component with a rounded tube-like cross-section

This makes the problem challenging because the model must simultaneously handle thin structures, holes, boundaries, different object scales, and overlapping components.

---

## 2. Dataset

The dataset contains:

| Split | Images | Resolution |
|---|---:|---|
| Training | 2,000 | 384 × 384 |
| Test | 500 | 384 × 384 |

The training masks store the class IDs directly as integer pixel values from 0 to 6.

The masks are loaded as:

```python
mask = np.array(Image.open(mask_path))
```

They should **not** be converted to RGB, because the underlying pixel values represent the actual segmentation labels.

---

## 3. Evaluation Metric

The primary metric is the **Dice coefficient**.

For a predicted binary mask and ground-truth mask:

```text
Dice = 2 × |Prediction ∩ Ground Truth|
       --------------------------------
       |Prediction| + |Ground Truth|
```

The value ranges from 0 to 1:

```text
0 → no overlap
1 → perfect overlap
```

The final metric is the mean Dice across every image and foreground class.

For 500 test images and 6 foreground classes, this produces:

```text
500 × 6 = 3,000 image-class evaluations
```

Dice is particularly appropriate for this problem because it directly measures segmentation overlap rather than relying only on overall pixel accuracy.

---

## 4. Model Architecture

### U-Net++

The main segmentation architecture is **U-Net++**.

Two encoder configurations were investigated:

```text
U-Net++ + ResNet-34
U-Net++ + EfficientNet-B1
```

The model was created with:

```python
smp.UnetPlusPlus(
    encoder_name=encoder_name,
    encoder_weights=None,
    in_channels=3,
    classes=7
)
```

The important point is:

```python
encoder_weights=None
```

Both encoders were trained from scratch.

### Architecture flow

```text
Input Image
     │
     ▼
Encoder
     │
     ├── Multi-scale feature extraction
     │
     ▼
U-Net++ Decoder
     │
     ├── Nested skip connections
     ├── Feature fusion
     └── Spatial reconstruction
     │
     ▼
7-channel segmentation head
     │
     ▼
Pixel-wise class prediction
```

### Why U-Net++?

Segmentation requires both semantic and spatial information.

The encoder learns increasingly abstract representations:

```text
edges → textures → shapes → object-level features
```

while the decoder reconstructs spatial detail.

The nested skip connections in U-Net++ allow information from different resolutions to be repeatedly combined, which is useful for objects with:

- thin boundaries;
- holes;
- small structures;
- elongated shapes;
- overlapping regions.

---

## 5. Training Configuration

| Parameter | Value |
|---|---:|
| Image size | 384 × 384 |
| Batch size | 8 |
| Maximum epochs | 100 |
| Early stopping patience | 15 |
| Initial learning rate | 0.001 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |
| Scheduler | Cosine Annealing |
| Random seed | 42 |
| Precision | Automatic Mixed Precision |

The labelled data was split into:

```text
85% training
15% validation
```

using a fixed random seed.

---

## 6. Data Augmentation

The training pipeline applies both geometric and appearance-based augmentation.

### Geometric augmentation

```python
HorizontalFlip
VerticalFlip
RandomRotate90
ShiftScaleRotate
```

The `ShiftScaleRotate` transformation introduces variation in:

- position;
- scale;
- rotation.

### Appearance augmentation

```python
RandomBrightnessContrast
HueSaturationValue
```

These reduce dependence on a specific lighting or colour configuration.

### Noise and blur

One of the following may also be applied:

```python
MotionBlur
GaussianNoise
```

### Coarse dropout

Random rectangular regions can be removed using `CoarseDropout`.

This encourages the network to use surrounding context rather than relying on a single small visual feature.

Validation data receives only normalization and tensor conversion.

---

## 7. Normalization

Images are normalized using:

```text
Mean = (0.485, 0.456, 0.406)
Std  = (0.229, 0.224, 0.225)
```

For each channel, normalization follows the standard transformation:

```text
normalized_value = (pixel_value - mean) / std
```

---

## 8. Loss Function

The final training objective combines Dice loss and Cross Entropy:

```text
Loss = 0.5 × Dice Loss + 0.5 × Cross Entropy
```

### Dice Loss

Dice loss encourages the predicted regions to overlap with the ground-truth regions.

Conceptually:

```text
Dice Loss ≈ 1 - Dice
```

### Cross Entropy

Cross entropy encourages the model to assign high probability to the correct class for each pixel.

For a pixel with ground-truth class `y`:

```text
Cross Entropy = -log(P(correct class))
```

### Why combine them?

The two objectives complement each other:

```text
Cross Entropy
     ↓
Pixel-level class discrimination

Dice Loss
     ↓
Region-level overlap

        ↓

Combined segmentation objective
```

---

## 9. Weight Initialization

Because pretrained weights were not used, convolutional and linear layers were initialized using Kaiming Normal initialization.

```python
nn.init.kaiming_normal_(
    module.weight,
    mode="fan_in",
    nonlinearity="relu"
)
```

For ReLU networks, Kaiming initialization is designed to keep activation magnitudes stable during the forward pass.

The approximate variance relationship is:

```text
Var(W) ≈ 2 / fan_in
```

This provides a suitable starting point for training deep ReLU-based networks from random initialization.

---

## 10. Optimization

The models were trained using **AdamW**.

At a high level, gradient-based learning updates the parameters according to:

```text
parameters ← parameters - learning_rate × gradient
```

The gradient indicates how changing each parameter would affect the loss.

AdamW additionally uses decoupled weight decay, which helps regularize the model.

### Learning-rate schedule

A cosine annealing schedule was used.

The learning rate gradually decreases during training, allowing:

```text
Early training
→ larger parameter updates

Later training
→ smaller parameter updates
```

This was particularly visible in the later epochs where the learning rate approached zero and the validation Dice became almost completely stable.

---

## 11. Mixed Precision Training

Training uses automatic mixed precision:

```python
with torch.amp.autocast('cuda'):
    outputs = model(images)
    loss = criterion(outputs, masks)
```

and gradient scaling:

```python
torch.amp.GradScaler('cuda')
```

Mixed precision allows suitable operations to use lower numerical precision while maintaining stability through gradient scaling.

This can reduce memory usage and improve GPU throughput.

---

## 12. Encoder Experiment

Two U-Net++ models were trained independently:

```text
Model A → EfficientNet-B1 encoder
Model B → ResNet-34 encoder
```

Each model had:

- independent training;
- independent optimizer state;
- independent learning-rate schedule;
- independent checkpoint;
- independent validation history.

The purpose was to determine whether different encoder representations produce complementary segmentation predictions.

---

## 13. Test-Time Augmentation

Inference uses six transformations:

```text
1. Original
2. Horizontal flip
3. Vertical flip
4. Horizontal + vertical flip
5. 90° rotation
6. 270° rotation
```

Each transformed image is passed through the model.

The transformed predictions are mapped back to the original coordinate system and averaged.

Conceptually:

```text
             ┌── Original ─────────┐
             ├── Horizontal Flip ──┤
             ├── Vertical Flip ────┤
Image ───────┼── HV Flip ──────────┼── Average
             ├── 90° Rotation ─────┤
             └── 270° Rotation ────┘
                                      │
                                      ▼
                              Final probability map
```

Averaging multiple predictions can reduce sensitivity to a particular orientation or transformation.

---

## 14. Model Ensemble

The two trained models were combined at the probability level.

For validation, the ensemble used:

```text
40% EfficientNet-B1
60% ResNet-34
```

or:

```python
ensemble_probs = (
    0.4 * probs_1 +
    0.6 * probs_2
)
```

For final test inference, equal weighting was used:

```text
50% EfficientNet-B1
50% ResNet-34
```

The final class prediction is obtained with:

```python
preds = torch.argmax(ensemble_probs, dim=1)
```

The reasoning behind the ensemble is that independently trained models can make different errors. Combining their probability distributions can therefore produce a more stable prediction.

---

## 15. Missing-Class Recovery

A task-specific post-processing step was also evaluated.

If a foreground class was completely absent from the initial `argmax` prediction, its probability map was examined.

Pixels satisfying:

```text
probability > 0.20
```

were considered candidates.

If more than 150 candidate pixels existed, only the 150 highest-probability pixels were retained.

The intuition was based on the known structure of the dataset: each image contains instances from all six foreground classes.

This heuristic is intentionally conservative and is applied only when a class completely disappears from the initial prediction.

---

## 16. Training Results

The recorded final portion of training was:

| Epoch | Loss | Dice | Learning Rate |
|---:|---:|---:|---:|
| 89 | 0.0327 | 0.9795 | 0.000030 |
| 90 | 0.0328 | 0.9795 | 0.000024 |
| 91 | 0.0324 | 0.9795 | 0.000020 |
| 92 | 0.0322 | 0.9795 | 0.000016 |
| 93 | 0.0328 | 0.9795 | 0.000012 |
| 94 | 0.0323 | 0.9795 | 0.000009 |
| 95 | 0.0320 | 0.9795 | 0.000006 |
| 96 | 0.0318 | **0.9796** | 0.000004 |
| 97 | 0.0319 | 0.9795 | 0.000002 |
| 98 | 0.0328 | 0.9795 | 0.000001 |
| 99 | 0.0321 | **0.9796** | ~0 |
| 100 | 0.0321 | **0.9796** | ~0 |

### Convergence

During these final epochs:

```text
Dice ≈ 0.9795 – 0.9796
Loss ≈ 0.032
Learning rate → 0
```

The extremely small change in Dice indicates that the model had already reached a stable region of the optimization landscape.

The final epochs therefore behaved more like fine-tuning than substantial representation learning.

---

# 17. Final Leaderboard Result

The final submitted solution achieved:

| Metric | Score |
|---|---:|
| **Private leaderboard Dice** | **0.97960** |
| Public leaderboard Dice | 0.98096 |

The **private score of 0.97960 is the primary final result** because it is the score used for the final evaluation.

The recorded validation/training-loop Dice of approximately 0.9796 was also consistent with the final private result.

---

## 18. RLE Submission

The segmentation masks must be converted into **Run Length Encoding (RLE)** before submission.

The encoder converts a binary mask into:

```text
start_position run_length
```

pairs.

The important detail is that the competition uses **column-major / Fortran ordering**.

For a 4 × 4 mask, pixels are numbered:

```text
1   5   9   13
2   6   10  14
3   7   11  15
4   8   12  16
```

Therefore, a normal row-major flattening would produce an incorrect submission.

The implementation explicitly handles this ordering and also provides an RLE decoder for verification.

---

## 19. Inference Pipeline

The final inference process is:

```text
                 Test Image
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   EfficientNet-B1          ResNet-34
          │                     │
          ▼                     ▼
         TTA                   TTA
          │                     │
          └──────────┬──────────┘
                     ▼
               Probability
                 Ensemble
                     │
                     ▼
             Pixel-wise Argmax
                     │
                     ▼
          Missing-class recovery
                     │
                     ▼
             Final segmentation
                     │
                     ▼
                    RLE
                     │
                     ▼
              submission.csv
```

---

## 20. Reproducibility

A fixed seed is used:

```python
SEED = 42
```

The seed is applied to:

- Python random;
- NumPy;
- PyTorch;
- CUDA when available.

The models are initialized without pretrained weights:

```python
encoder_weights=None
```

and convolutional/linear layers use Kaiming initialization.

The notebook contains the complete experimental workflow, from loading the data through training, validation, inference, and submission generation.

---

## 21. Important Implementation Details

### Mask loading

Correct:

```python
mask = np.array(Image.open(mask_path))
```

Incorrect:

```python
mask = Image.open(mask_path).convert("RGB")
```

The latter converts class IDs into palette colours.

### Image and mask augmentation

Geometric transformations are applied jointly to the image and mask so that pixel correspondence is preserved.

### Validation

The validation split is fixed using the random seed so that different experiments can be compared under the same conditions.

### Submission keys

The submission is generated for every:

```text
image × foreground class
```

pair, producing:

```text
500 × 6 = 3,000 rows
```

---

# 22. Repository

The repository intentionally contains only the two main project artifacts:

```text
.
├── README.md
└── segmentation_experiment.ipynb
```

### `README.md`

Documents:

- problem formulation;
- architecture;
- training methodology;
- experiments;
- inference strategy;
- results;
- implementation decisions.

### `segmentation_experiment.ipynb`

Contains the actual implementation and experimental workflow.

The raw training/test dataset is not included in the repository.

This keeps the repository lightweight while allowing anyone with legitimate access to the dataset to reproduce the experiments by changing the dataset path in the notebook.

---

# 23. Experimental Philosophy

The project was developed as a sequence of measurable experiments rather than treating the final model as a black box.

The general workflow was:

```text
Define problem
      ↓
Prepare masks and augmentations
      ↓
Train segmentation model
      ↓
Evaluate on validation split
      ↓
Compare encoder architectures
      ↓
Apply TTA
      ↓
Evaluate ensemble
      ↓
Generate test predictions
      ↓
Convert masks to RLE
      ↓
Submit
```

This makes it possible to understand where improvements come from rather than changing multiple components without being able to isolate their effects.

---

# 24. Future Experiments

Several controlled experiments could extend this work.

### Architecture

Compare:

```text
U-Net
U-Net++
DeepLab
Different U-Net++ encoders
```

### Loss functions

Experiment with different combinations of:

```text
Dice Loss
Cross Entropy
Focal Loss
```

For example:

```text
Loss = λ × Dice Loss + (1 - λ) × Cross Entropy
```

### Augmentation

Perform ablations to determine the contribution of:

```text
Geometric augmentation
Colour augmentation
Noise / blur
Coarse dropout
```

### TTA

Compare:

```text
No TTA
Flip-only TTA
Rotation TTA
Full TTA
```

### Ensemble weighting

Evaluate different model weights on the same validation set:

```text
50 / 50
40 / 60
60 / 40
```

### Post-processing

The missing-class recovery heuristic can be evaluated independently to determine whether it improves Dice or introduces unnecessary false-positive pixels.

---

# 25. Summary

This project developed a complete semantic segmentation system for mechanical components using U-Net++ and two different encoders.

### Dataset

```text
2,000 training images
500 test images
384 × 384 RGB
6 foreground classes
```

### Architecture

```text
U-Net++
├── ResNet-34
└── EfficientNet-B1
```

### Training

```text
AdamW
+ Dice + Cross Entropy
+ Cosine Annealing
+ Mixed Precision
+ Data Augmentation
```

### Inference

```text
TTA
+ Model Ensemble
+ Missing-Class Recovery
```

### Final result

```text
Private Dice: 0.97960
Public Dice:  0.98096
```

The main takeaway is that strong segmentation performance comes from the interaction of the entire pipeline:

```text
Data
  ↓
Augmentation
  ↓
Architecture
  ↓
Loss
  ↓
Optimization
  ↓
TTA
  ↓
Ensemble
  ↓
Post-processing
  ↓
RLE
```

rather than from the architecture alone.
