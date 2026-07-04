# Leaf Health Classifier

A CNN built from scratch in PyTorch to classify potato leaf images into 
three categories: healthy, early blight, and late blight — a stand-in 
task for growth-stage/health classification in precision agriculture, 
since labeled crop-growth-stage datasets are hard to come by publicly.

## Overview

- **Dataset:** PlantVillage (subset — 450 potato leaf images, 3 classes)
- **Model:** Small CNN built from scratch (4 conv blocks with BatchNorm 
  + ReLU + MaxPool, followed by global average pooling and a linear 
  classifier head)
- **Result:** 97% validation accuracy over 8 epochs

## Why this project

Precision agriculture increasingly relies on computer vision to monitor 
crop health and growth stages from field images — useful for early 
disease detection, yield estimation, and reducing manual field 
inspection. This project explores that pipeline end-to-end: structured 
data loading, augmentation, a from-scratch CNN, training, and evaluation 
beyond raw accuracy (confusion matrix, per-class precision/recall).

## Pipeline

1. **Data loading** — `torchvision.ImageFolder`, with folder structure 
   acting as ground-truth labels
2. **Preprocessing** — resize to 128×128, normalize using ImageNet 
   statistics, augment with random flips/rotation
3. **Model** — 4-block CNN (16→32→64 channels), trained from scratch 
   (no pretrained weights)
4. **Training** — Adam optimizer, cross-entropy loss, 80/20 train/val split
5. **Evaluation** — accuracy, confusion matrix, per-class precision/recall

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Early blight | 0.99 | 0.98 | 0.98 |
| Late blight | 0.96 | 0.98 | 0.97 |
| Healthy | 0.89 | 0.89 | 0.89 |

The two disease classes are occasionally confused with each other (visually 
similar symptoms at different stages of the same disease), while healthy 
leaves are rarely misclassified as diseased. Healthy also has fewer 
validation samples, which likely explains its slightly lower precision/recall.

## Next steps

- Add a transfer-learning variant (pretrained MobileNetV2) to compare 
  against the from-scratch model
- Address class imbalance for the healthy class
- Extend to more classes / crops
