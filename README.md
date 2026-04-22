# CNN Image Classification: Dogs vs Cats

Binary image classification using a custom CNN and MobileNetV2 transfer learning on the Dogs vs Cats dataset.

## Results

| Model | Validation Accuracy |
|---|---|
| Custom CNN (3-block, BatchNorm) | ~85% |
| MobileNetV2 (Transfer Learning) | ~95% |

## Architecture

### Custom CNN
- 4 Conv2D blocks with Batch Normalization and MaxPooling (32, 64, 128, 256 filters)
- Dropout(0.5) for regularisation
- Data augmentation: rotation, flip, zoom, shift

### MobileNetV2
- Pre-trained on ImageNet (1.4 million images)
- Frozen base with custom classification head
- Fine-tuned top 30 layers with learning rate 1e-5

## Dataset
- Source: Dogs vs Cats (Kaggle)
- Training samples: 16,000
- Validation samples: 4,000
- Image size: 128x128

## Tech Stack
Python, TensorFlow/Keras, MobileNetV2, scikit-learn, Matplotlib, Seaborn

## How to Run
1. Install dependencies: pip install tensorflow scikit-learn matplotlib seaborn kagglehub
2. Run all cells in CNN_Image_Classification.ipynb
3. Dataset downloads automatically via kagglehub
