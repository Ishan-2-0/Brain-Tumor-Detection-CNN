CopyBrain Tumor Detection CNN
Classifies brain MRI scans into four tumor types using convolutional neural networks.
Built as a comparison study between two architectures to isolate the effect of regularization
on generalization vs training accuracy.

What This Project Is
A structured investigation into whether adding batch normalization and dropout actually
improves model generalization. Most tutorials skip this question, this project runs both
models side-by-side on the same data to show what actually happens.

SimpleCNN: 3 conv layers, no regularization, Baseline.
ImprovedCNN: Same architecture + batch norm + dropout. Expected to generalize better
Result: ImprovedCNN generalizes worse. This is the real finding.

Why This Matters
Brain tumor classification is high-stakes. Radiologists need tools that are both accurate
and reliable across different imaging protocols and institutions. A model that memorizes
training data is dangerous in clinical practice

Results
MetricSimpleCNNImprovedCNNTest Accuracy88.6%80.6%Training Accuracy (Epoch 10)97.95%93.61%Overfitting Gap9.35pp13.01pp
SimpleCNN achieves higher test accuracy, but validation curves show different behavior.
Baseline accuracy plateaus at epoch 2 and never improves  flat curve, suggests overfitting.
Improved model is volatile: dips to 65% at epoch 4, recovers to 92% by epoch 9, never
reaches baseline's stability.
The real story: Neither model generalizes well. SimpleCNN memorizes better on this specific
test set. ImprovedCNN's regularization strategy (0.5 dropout on 256 units) was too aggressive.
Per-Class Performance (SimpleCNN)
ClassPrecisionRecallF1-ScoreGlioma0.870.780.82Meningioma0.920.760.83Notumor0.791.000.88Pituitary0.940.950.94
Notumor achieves 100% recall  no normal scans are missed. Clinically critical: false
negatives (missing a tumor) carry higher risk than false positives (flagging for review).
Confusion matrices show strong diagonal dominance for both. SimpleCNN's matrix is cleaner.
ImprovedCNN shows more cross-class confusion between glioma/meningioma and meningioma/notumor.

Dataset
PropertyDetailsSourceKaggle — Brain Tumor MRI ClassificationTraining images5,600Test images1,600Classesglioma, meningioma, pituitary, notumor (balanced)Resolution128×128 pixelsNormalizationMean 0.5, Std 0.5 (range:-1 to 1)AugmentationNone

Architecture
Two models. Same structure. Different regularization.
SimpleCNN
3 conv blocks: 32 → 64 → 128 filters (kernel=3, padding=1, ReLU)
Max pooling after each block (2×2)
Flatten to 32,768 → FC(256) → FC(4)
No batch norm. No dropout.
ImprovedCNN
Same conv structure + BatchNorm2d after each conv layer
Dropout(0.5) in first FC layer
Everything else identical.
Training
10 epochs, batch size 32, Adam (lr=0.001), CrossEntropyLoss.
GPU acceleration (CUDA if available, CPU fallback).

Challenges
1. Overfitting in SimpleCNN
Training accuracy reached 99.82% by epoch 9. Test accuracy was 88.6%.
Validation curve flat after epoch 2 — no improvement for 8 more epochs.
Model memorized training data instead of learning generalizable features.
2. Regularization Made Things Worse
Adding batch norm and dropout lowered test accuracy from 88.6% to 80.6%.
Expected outcome: regularization improves generalization.
Actual outcome: broke the model.
Root cause: dropout rate (0.5) too aggressive for a 256-unit FC layer.
Dropping half the neurons prevents the layer from learning stable representations.
Validation curve shows this: wild swings (65% → 92%), never stable.
3. Architectural Bottleneck
After 3 conv layers, spatial dimensions reduced to 16×16.
Flattened feature map: 128 * 16 * 16 = 32,768 elements.
Compressed to 256 units in FC layer — 128x reduction in feature space.
This compression is too aggressive for the problem complexity.
4. No Validation Split During Training
Both models evaluated on the training set during epochs, not a held-out
validation set. This means reported "training accuracy" is optimistic and
doesn't reflect generalization during training.
A proper validation split would have caught SimpleCNN's overfitting earlier.

Models
Base framework: PyTorch 2.x
Hardware: NVIDIA GPU (CUDA) with CPU fallback
Inference: Streamlit web app (GPU or CPU in Spaces)

How to Run
Live Demo
The app is deployed on Hugging Face Spaces:
Brain Tumor Detection CNN
Upload an MRI scan (JPG or PNG). Get instant predictions with confidence scores.
No installation. No GPU required on your machine.
Locally

Clone and enter the repo:

bashgit clone https://github.com/i-chan/Brain-Tumor-Detection-CNN.git
cd Brain-Tumor-Detection-CNN

Install dependencies:

bashpip install torch torchvision torchaudio
pip install streamlit pillow numpy scikit-learn

Run the app:

bashstreamlit run app.py

Open browser to http://localhost:8501, upload an MRI scan to run locally

Live Demo
Upload MRI scans directly in the deployed app (no setup required)

Retrain
Training pipeline in tumor-CNN.ipynb.

Download dataset from Kaggle place in Data/Training and Data/Testing
Run notebook cells in order.
Model saves to model_best.pth. Plots save to Results/.

Folder Structure
Brain-Tumor-Detection-CNN/
├── app.py                          # Streamlit inference app
├── model_best.pth                  # Trained ImprovedCNN weights
├── tumor-CNN.ipynb                 # Training notebook
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
├── Results/
│   ├── baseline_training.png
│   ├── improved_training.png
│   ├── confusion_matrices.png
│   ├── accuracy_comparison.png
│   └── validation_accuracy_comparison.png
└── README.md

Limitations
Model trained on a single Kaggle dataset. Generalization to other imaging protocols,
scanner manufacturers, or patient populations is unknown.
128×128 resolution is downsampled from original scans. High-resolution tumor details
are lost.
Balanced classes in the dataset (400 images each) don't reflect real clinical distribution.
Test set from same source as training data. No external validation.
Accuracy of 88% is not clinical-grade. Use only as a screening tool, never as a diagnostic tool.
Future Work

Retrain ImprovedCNN with lower dropout (0.2-0.3) and wider FC layer
Add validation set during training (proper train/val/test split)
Test on external datasets to measure robustness
Class activation maps (CAM) to visualize which MRI regions model attends to
Transfer learning: ResNet, DenseNet pre-trained on ImageNet
Safety & Disclaimer
This model is a research and educational tool only.
Can: Help flag potential abnormalities for review by a radiologist
Cannot: Diagnose/Substitute for professional medical judgment.
Always consult qualified medical professionals. Brain tumor classification requires
domain expertise and patient context that no automated system can provide.
