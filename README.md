# Gait Recognition with Temporal Kolmogorov-Arnold Networks

This project introduces a lightweight gait recognition framework that combines CNN-based spatial encoding with Temporal Kolmogorov-Arnold Networks (TKAN) for sequence modeling.

## Overview

Gait recognition identifies individuals using walking patterns and is useful in surveillance and public safety applications. However, existing temporal models often struggle with noisy sequences, clothing variation, and long-range temporal dependencies.

This work proposes:
- A compact CNN encoder for silhouette feature extraction
- TKAN for efficient temporal sequence modeling
- Robust recognition under challenging gait conditions

## Dataset

- CASIA-B Dataset

Evaluation Conditions:
- NM: Normal Walking
- BG: Carrying Bag
- CL: Clothing Variation

## Architecture

Silhouette Frames → CNN Encoder → TKAN Temporal Modeling → Temporal Pooling → Identity Classification

## Key Contributions

- Temporal modeling using TKAN
- Multi-timescale gait sequence learning
- Controlled comparison with LSTM and Transformer baselines
- Lightweight and efficient architecture

## Results

| Condition | Rank-1 Accuracy |
|---|---|
| NM | 99.52% |
| BG | 99.56% |
| CL | 98.82% |

## Compared Temporal Models

- LSTM
- Transformer
- TKAN (Proposed)

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Deep Learning

## Research Areas

- Computer Vision
- Biometrics
- Gait Recognition
- Temporal Modeling
- Kolmogorov-Arnold Networks
- Deep Learning
