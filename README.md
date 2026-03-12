Brain Tumor MRI Classifier: SimpleCNN vs ImprovedCNN
This project implements two Convolutional Neural Networks (CNNs) for brain tumor classification using MRI images: a SimpleCNN and an ImprovedCNN with dropout and batch normalization. The goal is to compare model performance, investigate overfitting, and leverage GPU acceleration for faster training.
Dataset
Source:Brain Tumor MRI dataset
Classes: ['glioma', 'meningioma', 'notumor', 'pituitary']
Training images: 5600
Testing images: 1600
Preprocessing: Images resized to 128×128, normalized and converted to tensors
Approach
SimpleCNN
3 convolutional layers followed by max pooling
2 fully connected layers
Activation: ReLU
Trained using CrossEntropyLoss and Adam optimizer

ImprovedCNN
3 convolutional layers with Batch Normalization
Max pooling and Dropout (0.5) in fully connected layer
Fully connected layers for classification
Activation: ReLU
Trained using CrossEntropyLoss and Adam optimizer

Training
GPU acceleration used (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
Batch size: 32
Epochs: 10

Metrics
Accuracy on test set
Confusion matrix
Validation accuracy curves

Key Insights
SimpleCNN achieved higher test accuracy (~99%) but shows signs of overfitting
ImprovedCNN achieved slightly lower accuracy (~92%) but generalizes better due to dropout and batch normalization
Overfitting in SimpleCNN is evident in faster memorization of training data, whereas ImprovedCNN balances learning and generalization

Results
All plots are stored in the Results/ folder:
loss_curves.png — Training loss curves for both models
accuracy_curves.png — Validation accuracy comparison
confusion_matrices.png — Side-by-side confusion matrices
Folder Structure
Brain-Tumor-Detection-CNN/
├── Data/
│   ├── Training/
│   └── Testing/
├── Notebooks/
│   └── tumor-CNN.ipynb
├── Results/
│   ├── baseline_training.png
│   ├── improved_training.png
│   ├── confusion_matrices.png
│   ├── accuracy_comparison.png
│   └── validation_accuracy_comparison.png
└── README.md