# 🧠 Brain Tumor Detection using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify **MRI brain images** as either having a **tumor** or **no tumor**. It was developed as part of an academic assignment focused on using AI in healthcare.

---

## 📂 Dataset

- **Source**: [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Total Images**: ~3,253
  - `yes/`: 1,522 images with tumor
  - `no/`: 1,731 images without tumor
- **Type**: JPEG MRI scan images
- **Labels**: Binary classification → `Tumor` (1) or `No Tumor` (0)

---

## 🧠 Model Architecture

A custom CNN built using **TensorFlow and Keras**:

- 3 × Conv2D layers with ReLU activation
- 3 × MaxPooling2D layers
- Flatten
- Dense(128) with ReLU
- Dropout(0.5)
- Output Dense(1) with Sigmoid activation

---

## ⚙️ Training Details

- **Image Size**: 150 × 150
- **Batch Size**: 32
- **Epochs**: 10
- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: Binary Crossentropy
- **Validation Split**: 20%

---

## 📊 Performance Metrics

- **Accuracy**: Tracked over training and validation
- **Confusion Matrix**: Visualized using Seaborn
- **Classification Report**: Includes precision, recall, and F1-score

---

## 📈 Results

- Model achieved high accuracy on validation data
- Minor overfitting observed after epoch 7
- High recall for tumor class → crucial in medical diagnostics

---

## 🚧 Challenges

- Data imbalance between tumor and non-tumor images
- Limited dataset diversity
- Slight overfitting with small dataset

---

## 📚 Tools & Libraries

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib / Seaborn
- Scikit-learn

---

## 📎 Folder Structure

```bash
📁 brain_tumor_dataset/
   ├── yes/
   └── no/
📄 brain_tumor_cnn.ipynb
📄 README.md
📄 presentation.pdf
📄 report.pdf
