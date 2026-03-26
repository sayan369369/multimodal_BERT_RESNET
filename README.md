# Multimodal Classification using BERT and ResNet

## Overview
This project implements a multimodal deep learning model that combines text and image features for classification.

- Text features are extracted using BERT
- Image features are extracted using ResNet50
- Features are fused and passed through a classifier

## Architecture
Text → BERT (768)  
Image → ResNet50 (2048)  
Fusion → 2816 features  
Classifier → Prediction  

## Dataset
Custom dataset with:
- Dog images + captions (class 0)
- Car images + captions (class 1)

## Training
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 10

## Results
- Loss decreases across epochs
- Final Accuracy: 100%

## How to Run

```bash
pip install torch torchvision transformers pandas pillow
python train.py
