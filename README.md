# Brain Tumor Detection  CNN Classifier

Classifies brain MRI scans into four tumor types using convolutional
neural networks. Built as a comparison study between two architectures
to isolate the effect of regularization on generalization vs training
accuracy.

---

## What this project is

A structured investigation into whether adding batch normalization and
dropout actually improves model generalization. Most tutorials skip this
question, this project runs both models side by side on the same data
to show what actually happens.

| Model | Architecture | Result |
|-------|-------------|--------|
| SimpleCNN | 3 conv layers, no regularization | Baseline |
| ImprovedCNN | Same + batch norm + dropout | Expected to generalize better |

**Real finding: ImprovedCNN generalizes worse.**

### Why this matters

Brain tumor classification is high-stakes. Radiologists need tools that
are both accurate and reliable across different imaging protocols and
institutions. A model that memorizes training data is dangerous in
clinical practice.

---

## Results

| Metric | SimpleCNN | ImprovedCNN |
|--------|-----------|-------------|
| Test Accuracy | **88.6%** | 80.6% |
| Training Accuracy (Epoch 10) | 97.95% | 93.61% |
| Overfitting Gap | 9.35pp | 13.01pp |

SimpleCNN achieves higher test accuracy, but validation curves tell the
deeper story. The baseline plateaus at epoch 2 and never improves 
flat curve, classic overfitting. ImprovedCNN is volatile: dips to 65%
at epoch 4, recovers to 92% by epoch 9, never reaches baseline
stability.

### Per-class performance (SimpleCNN)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.87 | 0.78 | 0.82 |
| Meningioma | 0.92 | 0.76 | 0.83 |
| No tumor | 0.79 | **1.00** | 0.88 |
| Pituitary | 0.94 | 0.95 | 0.94 |

No tumor achieves 100% recall no normal scans are missed. Clinically
critical: false negatives (missing a tumor) carry higher risk than false
positives (flagging for review). Confusion matrices show strong diagonal
dominance for both models. SimpleCNN's matrix is cleaner. ImprovedCNN
shows more cross-class confusion between glioma/meningioma and
meningioma/no tumor.

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Kaggle — Brain Tumor MRI Classification |
| Training images | 5,600 |
| Test images | 1,600 |
| Classes | glioma, meningioma, pituitary, notumor (balanced) |
| Resolution | 128×128 pixels |
| Normalization | Mean 0.5, Std 0.5 (range: −1 to 1) |
| Augmentation | None |

---

## Architecture

Two models. Same structure. Different regularization.

**SimpleCNN**
- 3 conv blocks: 32, 64, 128 filters (kernel=3,padding=1,ReLU)
- Max pooling after each block (2*2)
- Flatten to 32,768 to FC(256) to FC(4)
- No batch norm. No dropout.

**ImprovedCNN**
- Same conv structure + BatchNorm2d after each conv layer
- Dropout(0.5) in first FC layer
- Everything else identical

**Training:** 10 epochs, batch size 32, Adam (lr=0.001),
CrossEntropyLoss. GPU acceleration (CUDA if available, CPU fallback)

---

## Challenges

### 1. Overfitting in SimpleCNN
Training accuracy reached 99.82% by epoch 9. Test accuracy was 88.6%
Validation curve flat after epoch 2 no improvement for 8 more epochs
Model memorized training data instead of learning generalizable features

### 2. Regularization made things worse
Adding batch norm and dropout lowered test accuracy from 88.6% to 80.6%.
Expected: regularization improves generalization. Actual: broke the model
Root cause: dropout rate (0.5) too aggressive for a 256 unit FC layer
Dropping half the neurons prevents stable representation learning
Validation curve shows this wild swings (65% → 92%), never stable

### 3. Architectural bottleneck
After 3 conv layers, spatial dimensions reduce to 16×16. Flattened
feature map: 128 × 16 × 16 = 32,768 elements, compressed to 256 units
 a 128× reduction. Too aggressive for this problem's complexity.

### 4. No validation split during training
Both models were evaluated on the training set during epochs, not a
held out validation set. Reported training accuracy is therefore
optimistic. A proper validation split would have caught SimpleCNN's
overfitting earlier.

---

## Live demo

**[Try it on HuggingFace Spaces](https://huggingface.co/spaces/i-chan/Brain-Tumor-Detection-CNN)**

Upload an MRI scan (JPG or PNG). Get instant predictions with confidence
scores. No installation. No GPU required.

---

## How to run locally

```bash
git clone https://github.com/Ishan-2-0/Brain-Tumor-Detection-CNN.git
cd Brain-Tumor-Detection-CNN
pip install torch torchvision torchaudio
pip install streamlit pillow numpy scikit-learn
streamlit run app.py
```

Open browser to `http://localhost:8501` and upload an MRI scan.

**To retrain:** download the dataset from Kaggle, place in
`Data/Training` and `Data/Testing`, run `tumor-CNN.ipynb` in order.
Model saves to `model_best.pth`, plots save to `Results/`.

---

## Folder structure
Brain-Tumor-Detection-CNN/
├── app.py                        # Streamlit inference app
├── model_best.pth                # Trained ImprovedCNN weights
├── tumor-CNN.ipynb               # Training notebook
├── Data/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
└── Results/
├── baseline_training.png
├── improved_training.png
├── confusion_matrices.png
├── accuracy_comparison.png
└── validation_accuracy_comparison.png

---

## Stack

PyTorch 2.x · CUDA (CPU fallback) · Streamlit

---

## Limitations

- Trained on a single Kaggle dataset generalization to other imaging
  protocols, scanner manufacturers, or patient populations is unknown
- 128×128 resolution is downsampled high-resolution tumor details lost
- Balanced classes don't reflect real clinical distribution
- Test set from same source as training data no external validation
- 88% accuracy is not clinical-grade use only as a screening tool

## Future work

- Retrain ImprovedCNN with lower dropout (0.2-0.3) and wider FC layer
- Add proper train/val/test split during training
- Test on external datasets to measure robustness
- Class activation maps (CAM) to visualize attended MRI regions
- Transfer learning with ResNet or DenseNet pre-trained on ImageNet

---

## Safety disclaimer

This model is a research and educational tool only.

Can help flag potential abnormalities for radiologist review  
Cannot diagnose or substitute for professional medical judgment  

Brain tumor classification requires domain expertise and patient context
that no automated system can provide. Always consult a qualified medical
professional

---
