# HistoPathological Cancer Detection using Deep Learning

This project applies Deep Learning techniques, specifically Convolutional Neural Networks (CNNs), to classify histopathological images of lung and colon tissues. It aims to detect and categorize different types of carcinomas and benign tissues with high accuracy.

---

# Project Overview

The objective of this project is to build an automated classification system capable of distinguishing between five different tissue types associated with lung and colon cancer. The project utilizes PyTorch for building and training the model and includes techniques for model interpretability such as Grad-CAM.

---

# Key Features

- **Automated Data Pipeline:** Downloads and restructures the Kaggle dataset automatically.  
- **Data Preprocessing:** Resizing, normalization, and data augmentation techniques.  
- **Deep Learning Model:** Custom model designed using Convolutional Neural Networks with residual connections.  
- **Evaluation:** Comprehensive performance metrics including Accuracy, Precision, Recall, and F1-Score.  
- **Visualization:** Uses Matplotlib and Seaborn for plotting training curves and Confusion Matrices.  
- **Explainability:** Implements Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize parts of the image influencing the classification.

---

# Dataset

The project uses the **Lung and Colon Cancer Histopathological Images** dataset from Kaggle.

- **Source:** Kaggle – Andrewmvd  
- **Total Images:** 25,000  
- **Image Size:** 768 × 768 (Resized to 224 × 224 for training)

---

# Classes

- **Colon Adenocarcinoma (colon_aca):** Malignant tumor of the colon  
- **Benign Colon Tissue (colon_n):** Healthy colon tissue  
- **Lung Adenocarcinoma (lung_aca):** Malignant tumor of the lung  
- **Benign Lung Tissue (lung_n):** Healthy lung tissue  
- **Lung Squamous Cell Carcinoma (lung_scc):** Malignant tumor of the lung  

---

# Tech Stack

- **Language:** Python  
- **Deep Learning Framework:** PyTorch  

---

# Libraries

- torch, torchvision (Model building & training)  
- numpy, pandas (Data manipulation)  
- matplotlib, seaborn (Visualization)  
- scikit-learn (Metrics & evaluation)  
- kagglehub (Dataset downloading)  
- cv2 (OpenCV for image processing / Grad-CAM)  

---

# Installation & Usage

# Clone the Repository
```bash
git clone https://github.com/yourusername/histopathological-cancer-detection.git
cd histopathological-cancer-detection
```

# Methodology
-Data Loading: The script downloads the dataset via kagglehub and organizes the 5 sub-classes into a unified directory structure for easy loading with ImageFolder.
-Preprocessing: Images are transformed to tensors, resized to 224×224 pixels, and normalized using standard ImageNet mean and standard deviation.
-Model Training: A Convolutional Neural Network is trained on the dataset using Cross-Entropy Loss and an optimizer (Adam).
-Evaluation: The model is tested on a validation set. A Confusion Matrix is plotted to visualize misclassifications, and a Classification Report provides detailed metrics per class.
-Interpretability: The project applies Grad-CAM to overlay heatmaps on the original images, highlighting the regions of the cell tissue that are most indicative of the disease.

# Results
-Accuracy: Test Accuracy = 98.96 %
-Confusion Matrix: See the notebook outputs for a detailed breakdown of True Positives vs. False Negatives.

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# Fork the project.

-Create your Feature Branch (git checkout -b feature/AmazingFeature).
-Commit your Changes (git commit -m 'Add some AmazingFeature').
-Push to the Branch (git push origin feature/AmazingFeature).
-Open a Pull Request.

# 📜License
This project is licensed under the MIT License - see the LICENSE file for details.

# 🙏 Acknowledgements
Dataset provided by Andrewmvd on Kaggle.
Inspiration from medical imaging research community.
