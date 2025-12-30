# Pneumonia Detection from Chest X-Ray:Resnet18 and Custom CNN

A deep learning project for binary classification of chest X-ray images to detect pneumonia using both a Custom CNN and a fine-tuned ResNet-18 model.

##  Overview

This project implements two different approaches to detect pneumonia from chest X-ray images:

1. **Custom CNN Model** - A baseline convolutional neural network built from scratch
2. **ResNet-18 Model** - A pre-trained ResNet-18 model fine-tuned for pneumonia detection

##  Dataset

The project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle, which contains:

- **Training set**: 5,216 images (1,341 Normal, 3,875 Pneumonia)
- **Validation set**: 16 images
- **Test set**: 624 images

### Class Imbalance Handling

The dataset has significant class imbalance. This is addressed using:
- **Weighted Random Sampling** (Custom CNN)
- **Positive Weight in Loss Function** for BCEWithLogitsLoss

##  Model Architectures

### Custom CNN Model

A 4-block convolutional neural network:

```
Block 1: Conv2d(3→32) → BatchNorm → ReLU → MaxPool
Block 2: Conv2d(32→64) → BatchNorm → ReLU → MaxPool
Block 3: Conv2d(64→128) → BatchNorm → ReLU → MaxPool
Block 4: Conv2d(128→256) → BatchNorm → ReLU → MaxPool
↓
Global Average Pooling
↓
FC(256→128) → ReLU → Dropout(0.7) → FC(128→1)
```

### ResNet-18 (Transfer Learning)

- Pre-trained ResNet-18 with ImageNet weights
- Frozen early layers (only `layer4` and `fc` are trainable)
- Custom classification head:
  ```
  FC(512→512) → BatchNorm → ReLU → Dropout(0.6) → FC(512→1)
  ```

##  Data Preprocessing & Augmentation

Both models use the following preprocessing pipeline:

- **Resize** to 224×224 (or 256 with center crop to 184)
- **Random Affine** transformations (rotation, translation, scaling)
- **Color Jitter** (brightness, contrast adjustments)
- **Random Horizontal Flip**
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) - optional
- **Normalization** using ImageNet statistics `[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`

##  Training Configuration

| Parameter | Custom CNN | ResNet-18 |
|-----------|------------|-----------|
| Optimizer | Adam | Adam (differential LR) |
| Learning Rate | 0.001 | layer4: 1e-5, fc: 1e-4 |
| Batch Size | 32 | 32 |
| Epochs | 20 | 12 |
| Loss Function | BCEWithLogitsLoss | BCEWithLogitsLoss |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Early Stopping | ✗ | ✓ (patience=6) |

##  Evaluation Metrics

The models are evaluated using:

- **Accuracy**
- **Recall** (Sensitivity) - Critical for medical diagnosis
- **F1-Score**
- **AUC-ROC**
- **Confusion Matrix**

### ResNet-18 Performance (Test Set)

| Metric | Score |
|--------|-------|
| Accuracy | 91% |
| Recall | 99% |
| F1-Score | 87% |

## 🛠️ Requirements

```
torch
torchvision
numpy
pandas
opencv-python
Pillow
scikit-learn
matplotlib
seaborn
```

##  Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Download the Dataset

Download the [Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract it.

### 3. Run the Notebooks

- **Custom CNN**: Open and run `Custom-CNN-Model.ipynb`
- **ResNet-18**: Open and run `ResNet-Model.ipynb`

Update the `data_dir` variable in the notebooks to point to your dataset location.

##  Project Structure

```
├── Custom-CNN-Model.ipynb   # Custom CNN implementation
├── ResNet-Model.ipynb       # ResNet-18 transfer learning implementation
├── README.md                # Project documentation
└── best_baseline_model.pth  # Saved model weights (generated after training)
```

##  Key Features

- **CLAHE Preprocessing**: Custom image enhancement for better contrast in X-ray images
- **Class Imbalance Handling**: Weighted sampling and loss functions
- **Transfer Learning**: Leveraging pre-trained ImageNet weights
- **Comprehensive Evaluation**: Multiple metrics including confusion matrices and ROC curves
- **Visualization**: Training curves, confusion matrices, and performance comparisons

##  Notes

- The notebooks are configured for both Google Colab and Kaggle environments
- GPU acceleration (CUDA) is automatically detected and used when available
- Model checkpoints are saved during training for the best validation loss

## License

This project is for educational purposes.

## Acknowledgments

- [Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney
- PyTorch and torchvision teams for the pre-trained models
