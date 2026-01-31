# Skin Lesion Segmentation using Deep Learning

This project implements and compares multiple deep learning models for automatic skin lesion segmentation using the ISIC 2018 dataset.

## Models Implemented
- U-Net  
- Attention U-Net  
- Xception-U-Net Hybrid  

## Key Techniques
- Data augmentation (flip, brightness adjustment)
- L1 and L3 regularization
- Multiple activation functions (ReLU, LeakyReLU, Tanh)
- Hyperparameter tuning
- 80/20 training-testing split
- Dice coefficient and binary accuracy metrics

## Dataset
ISIC 2018 Skin Lesion Segmentation Dataset

## Results
All models were trained and evaluated, and performance was compared using accuracy and Dice coefficient. Training graphs and tuning results are included.

## How to Run
1. Download the ISIC 2018 dataset
2. Place images and masks in:
   - ISIC2018/images
   - ISIC2018/masks
3. Run:
   ```bash
   python training.py
