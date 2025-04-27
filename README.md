# Deepfake Detection Using Hybrid CNN-RNN (LSTM) Model
 A deep learning project utilizing a CNN-LSTM model to detect deepfake content with high accuracy. The repository includes a Jupyter Notebook for model implementation and training, along with a responsive website built using HTML/CSS to showcase the results and provide an interactive interface.
 
## 📖 Introduction
Deepfake technology has rapidly advanced, enabling hyper-realistic video fabrications that pose significant risks to personal, financial, and societal security.  
This project tackles the deepfake detection challenge by developing a hybrid deep learning model that combines Convolutional Neural Networks (CNNs) for spatial feature extraction and Recurrent Neural Networks (RNNs) with LSTM units for temporal sequence analysis.

The solution is deployed through a web-based application, allowing users to easily upload videos and determine authenticity with an achieved detection accuracy of **51%** on real-world benchmark datasets.

---

## 🏗️ Project Overview

- **Problem Addressed:** Detection of manipulated videos (deepfakes) by analyzing both frame-level artifacts and temporal inconsistencies.
- **Solution Developed:** A hybrid CNN-LSTM model trained on a curated dataset and integrated into a lightweight web application using Flask.
- **Real-world Impact:** Equips users, organizations, and digital platforms with a tool to validate video authenticity and combat digital misinformation.

---

## 🎯 Key Features

- Deep learning hybrid architecture: Spatial + Temporal video analysis
- Detection based on facial manipulation inconsistencies
- Dataset diversity: Trained on FaceForensics++, DFDC; tested on Celeb-DF (v2)
- Offline video upload and analysis through a custom-built web interface
- Model designed with computational efficiency for deployment in resource-limited environments

---

## 📚 Datasets Referenced

- [FaceForensics++](https://github.com/ondyari/FaceForensics) ([Rossler et al., 2019](https://arxiv.org/abs/1901.08971))
- [Deepfake Detection Challenge Dataset (DFDC)](https://ai.facebook.com/datasets/dfdc) ([Dolhansky et al., 2020](https://arxiv.org/abs/2006.07397))
- [Celeb-DF (v2)](https://github.com/yuezunli/Celeb-DF) ([Li et al., 2020](https://arxiv.org/abs/1909.12962))

---

## 🧠 Model Architecture

- **Feature Extraction:** CNN backbones (ResNet50, EfficientNetV2B0, Xception)
- **Temporal Sequence Modeling:** LSTM networks
- **Classification Layer:** Dense layers with Softmax activation for binary classification (Real or Fake)


















# 🕵️‍♂️ Deepfake Detection in Videos | CNN-RNN (LSTM) Hybrid Model

**Developed by:** Atul Abhishek  
**MSc Cyber Security | City, University of London**

---

## 📜 Abstract

The rapid evolution of deepfake technology poses significant threats to media integrity and public trust.  
This project presents a hybrid deep learning solution combining Convolutional Neural Networks (CNN) for spatial feature extraction and Recurrent Neural Networks (RNN) with LSTM units for temporal analysis.  

A user-friendly web application built with Flask enables users to upload videos and detect deepfakes with an achieved accuracy of **51%** using benchmark datasets.

---

## 🚀 Live Demo

▶️ [Watch Demo Video](#)  
*(Upload your screen recording showing app usage, then update this link.)*

---

## 📷 Screenshots

| Landing Page | Uploading Video | Detection Result |
|:------------:|:----------------:|:----------------:|
| ![Landing Page](static/images/landing_page.png) | ![Upload Page](static/images/upload_page.png) | ![Result](static/images/detection_result.png) |

Additional Screenshots:
- What is Deepfake? Info Page
- Detection in Progress (Loading Spinner)
- Error Handling (Unsupported File Types)

---

## 🏗️ Project Architecture
User Uploads Video ↓ Flask Backend ↓ Frame Extraction (OpenCV + dlib) ↓ Feature Extraction (CNN: ResNet50/EfficientNet/Xception) ↓ Temporal Analysis (RNN - LSTM) ↓ Prediction: Real or Fake ↓ Result Display on Web Interface


---

## 🎯 Features

- Upload short video clips for real/fake classification.
- Hybrid deep learning model (CNN for spatial features + LSTM for temporal dynamics).
- Trained on subsets of **FaceForensics++** and **DFDC**, tested on **Celeb-DF (v2)**.
- Offline processing for enhanced performance.
- Easy-to-use Flask web application.
- Confidence scores for each detection.
- Visualization of detection progress.

---

## ⚙️ Installation Instructions

```bash
# Clone this repository
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py

Make sure you have Python 3.8+ installed.

Download the trained model file (model.h5) if not included (link provided separately).



deepfake-detection/
│
├── app.py
├── model/
│   ├── model.h5          # Pre-trained CNN-LSTM model
│   └── model_utils.py    # Prediction functions
├── static/
│   ├── css/
│   ├── js/
│   └── images/           # Screenshots and evaluation plots
├── templates/
│   ├── index.html
│   ├── upload.html
│   └── result.html
├── requirements.txt
├── README.md
└── LICENSE



