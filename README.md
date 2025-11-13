# Audio Quality Prediction using Wav2Vec2 and Psychoacoustic Features (On-Going)
### A Multi-Task Deep Learning Approach for Perceptual Audio Quality Assessment

This repository implements a research-grade audio quality prediction system using **transformer-based embeddings (Wav2Vec2)** and **psychoacoustic audio features**.  
The system predicts:

1. **Audio quality class** — *low*, *medium*, *high*  
2. **Continuous perceptual quality score** — *pseudo-PESQ (1.0–4.5)*  

All audio data is **synthetically generated**, so no external dataset is required.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Dataset Construction](#dataset-construction)
- [Audio Degradations](#audio-degradations)
- [Feature Extraction](#feature-extraction)
- [Model](#model)
- [Training](#training)
- [Results](#results)
- [Visualizations](#visualizations)
- [How to Run](#how-to-run)
- [References](#references)

---

## Overview

This project explores **perceptual audio quality modeling** using a combination of:

- Self-supervised speech representations from **Wav2Vec2**
- Classical **psychoacoustic features**
- A **multi-task neural network** that jointly predicts:
  - A *categorical* audio quality label
  - A *regression*-based perceptual score

The methodology is inspired by research from:

- Dolby Laboratories  
- Fraunhofer IIS  
- NeuralPESQ (ICASSP 2021)  
- MOSNet (Interspeech 2019)  
- SSL-MOS Benchmark 2022  

---

## Key Features

- 100% synthetic dataset (no downloads needed)
- Realistic clean signal generation (FM, AM, harmonics)
- Five degradation types across three severity levels
- Balanced dataset design
- Pseudo-PESQ score generation
- Wav2Vec2 transformer-based embeddings (768-dim)
- Psychoacoustic feature extraction (19-dim)
- Feature fusion and standardization
- Multi-task neural network (classification + regression)
- Full visualization suite (spectrograms, confusion matrix, scatter, etc.)
- Fully reproducible in Google Colab

---

## Architecture

                Raw Audio (Clean or Degraded)
                              │
            ┌─────────────────┴──────────────────┐
            │                                    │
     Wav2Vec2 Encoder                    Psychoacoustic Features
      (768-dim SSL)                           (19-dim)
            │                                    │
            └──────────────┬─────────────────────┘
                           ▼
                Concatenated Feature Vector
                         (787-dim)
                           │
                   Shared MLP Encoder
                           │
          ┌────────────────┴────────────────┐
          │                                 │
  Classification Head                Regression Head
 (low/medium/high)                  (1.0 to 4.5 score)


---

## Dataset Construction

Each clean audio sample is synthesized using:

- Frequency modulation (vibrato)
- Amplitude modulation (tremolo)
- Multiple harmonics
- Light natural noise floor

Then the following degradations are applied:

- **noise**
- **clip**
- **lowpass**
- **reverb**
- **compression**

Each with severity levels **1, 2, 3**.

### Dataset size


Dataset is **balanced** to avoid model bias.
5 degradation types × 3 severities × N samples per combination
= 300 total samples (default)
---

## Audio Degradations

| Degradation | Description                               |
|-------------|-------------------------------------------|
| noise       | Additive white noise                      |
| clip        | Hard amplitude clipping                   |
| lowpass     | High-frequency suppression                |
| reverb      | Echo / temporal smearing                  |
| compression | Time-stretch based artifact simulation    |

Severity:
- 1 → High Quality  
- 2 → Medium  
- 3 → Low  

---

---

## Model

A **multi-task neural network**:

- **Shared encoder**  
  - Linear(787 → 512)  
  - Linear(512 → 256)  
  - Dropout  

- **Heads**  
  - Classification (3 classes)  
  - Regression (1 value, pseudo-PESQ)

### Loss Function
Total Loss = CE(Classification) + λ * MSE(Regression)

λ = 0.5

---

## Training

- Framework: PyTorch  
- Optimizer: Adam  
- LR: 1e-4  
- Epochs: 40  
- Train/Test Split: 80/20  
- Stratified class sampling  
- StandardScaler for all features  

---

## Results

### Classification  
Quality labels: *low, medium, high*

Example (your exact numbers will differ):

Accuracy: XX%
Low: Precision XX | Recall XX
Medium: Precision XX | Recall XX
High: Precision XX | Recall XX


### Regression (Pseudo-PESQ)
MAE: 0.xx
MSE: 0.xx
RMSE: 0.xx


Regression scatter plot shows a visible correlation between predicted and actual quality scores.

---

## Visualizations

The notebook provides:

- Waveform plots
- STFT spectrograms
- Mel spectrograms
- Clean vs degraded comparisons
- All degradation types grid
- Training loss curves
- Confusion matrix
- Predicted vs true score scatter plot



---

## References

- Maiti & Kim, *Speech Quality Assessment Using Self-Supervised Features*, Interspeech 2021  
- Lo et al., *MOSNet*, Interspeech 2019  
- Avila et al., *NeuralPESQ*, ICASSP 2021  
- Hsu et al., *HuBERT*, 2021  
- ITU-T P.862 (PESQ), P.863 (POLQA)  

---

## Notes

This project is suited for:

- Audio ML portfolios  
- Dolby / Fraunhofer / Meta interviews  
- Master thesis research  
- Codec and perceptual quality experimentation  

Open an issue if you want enhancements or extensions.

