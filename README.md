# 🎧 Audio Quality Classification (Synthetic Dataset)

This project explores a simple and interpretable approach to **classifying audio quality** using:

- **Synthetic audio signals**
- **Psychoacoustic features (MFCC + spectral descriptors)**
- **RandomForest classifier**

The goal is to predict whether an audio sample is:

- **High quality**
- **Medium quality**
- **Low quality**

This project is **only for exploring the idea**, not for real-world or production audio evaluation.

---

## 🔍 Project Overview

The pipeline includes:

1. **Generate synthetic clean tones**  
   (vibrato, tremolo, harmonics, noise floor)

2. **Apply controlled degradations**  
   - Noise  
   - Clipping  
   - Lowpass filtering  
   - Reverb  
   - Compression  

3. **Extract psychoacoustic features**  
   MFCC mean, ZCR, RMS, spectral centroid/bandwidth/rolloff/flatness

4. **Train a RandomForest classifier**

5. **Evaluate model accuracy + diagnostic plots**

---

## 📊 Results

The model achieves:

- **Training Accuracy:** 1.00  
- **Test Accuracy:** ~**99%**  

This result is **expected**, and **not caused by overfitting**.

### ✅ Why this high accuracy is real (not overfitting)

The dataset is:

- **Synthetic**
- **Perfectly balanced**
- **Fully controlled**
- **Has clear, strong degradations**
- **Uses powerful spectral features**

Distortions like clipping, lowpass, reverb, noise, and compression leave **distinct spectral fingerprints** that MFCC and spectral features capture extremely well.

Because the classes are **inherently separable**, a high accuracy is expected and correct.

### 📌 Additional metrics that confirm no overfitting

The following metrics confirm that the classifier generalizes well:

#### **1. Train vs Test Accuracy**
Train Accuracy: 100%
Test Accuracy: 99%

→ If overfitting were happening, the test accuracy would drop significantly.

#### **2. Confusion Matrix**
A near-perfect diagonal matrix indicates clean separation of classes.

#### **3. Classification Report**
Balanced F1-scores across all classes show stable, non-biased performance.

#### **4. No class collapse**
Models that overfit often predict mostly one class.  
Here, all three classes (low/medium/high) are predicted correctly.

### 📌 Why synthetic datasets behave this way

Synthetic distortions are:

- deterministic  
- consistent  
- clean  
- not affected by recording conditions  
- not influenced by human vocal variability  

This makes the machine learning task **much easier** than real-world audio quality prediction.

---

## ⚠️ Important Disclaimer

This project is **not** intended for real-world audio quality assessment.

Real-life audio would require:

- large datasets  
- speech/music content  
- real codec artifacts  
- room acoustics  
- human subjective ratings (MOS/PESQ)  
- more advanced models like Wav2Vec2 or MOSNet  

This project is a **concept exploration**, not a production-quality tool.

---

## ❓ Q&A: Understanding High Accuracy and Limitations

### **Q1: Why does the model reach ~99% accuracy?**
Because the dataset is fully synthetic and distortions are easy to distinguish using spectral features.

### **Q2: Is this overfitting?**
No. Both training and testing accuracies are high, and the confusion matrix + F1 scores confirm clean generalization.

### **Q3: Would this model perform well on real speech or music?**
No. Real audio is much more complex.  
Accuracy would drop significantly without:

- large datasets  
- perceptual MOS labels  
- advanced deep learning models

### **Q4: Why is this project still useful?**
It helps you:
- understand audio feature extraction  
- see the effect of degradations  
- build a complete ML audio pipeline  
- explore ideas before scaling up  
- prepare for real audio research tasks  

---

## 📂 Files

- `audio_quality_project.ipynb` — Main notebook  
- `confusion_matrix.png` — Example output  
- `README.md` — Project documentation  

---

## 🛠 Requirements

Install:

```bash
pip install numpy librosa scikit-learn matplotlib soundfile
