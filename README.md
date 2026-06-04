# GrainCNN — Microstructure Grain Size Predictor

A lightweight Convolutional Neural Network (CNN) that predicts mean grain size 
from polycrystalline microstructure images.

## Live Demo
Try it here: https://huggingface.co/spaces/YohanNimash1/GrainSizesPrediction

## Overview
- Input: Microstructure image (any format)
- Output: Predicted mean grain size in µm
- Architecture: 3-block CNN with adaptive average pooling and regression head
- Trained on SPPARKS Monte Carlo grain growth simulations

## Model Architecture
- 3 convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool)
- Adaptive average pooling to 2x2
- Fully connected regression head → single grain size output
- Input size: 64x64 grayscale

## Performance
- MAE: [add your value] µm
- RMSE: [add your value] µm
- R²: [add your value]

## Usage
```python
