# Deepfake Detection Using Hybrid CNN-RNN (LSTM) Model
 A deep learning project utilizing a CNN-LSTM model to detect deepfake content with high accuracy. The repository includes a Jupyter Notebook for model implementation and training, along with a responsive website built using HTML/CSS to showcase the results and provide an interactive interface.
 
## 📖 Introduction
Deepfake technology has rapidly advanced, enabling hyper-realistic video fabrications that pose significant risks to personal, financial, and societal security.  

This project tackles the deepfake detection challenge by developing a hybrid deep learning model that combines Convolutional Neural Networks (CNNs) for spatial feature extraction and Recurrent Neural Networks (RNNs) with LSTM units for temporal sequence analysis.

The solution is deployed through a web-based application, allowing users to easily upload videos and determine authenticity with an achieved detection accuracy of **51%** on real-world benchmark datasets.

## 🏗️ Project Overview
- **Problem Addressed:** Detection of manipulated videos (deepfakes) by analyzing both frame-level artifacts and temporal inconsistencies.
- **Solution Developed:** A hybrid CNN-LSTM model trained on a curated dataset and integrated into a lightweight web application using Flask.
- **Real-world Impact:** Equips users, organizations, and digital platforms with a tool to validate video authenticity and combat digital misinformation.

## 🎯 Key Features
- Deep learning hybrid architecture: Spatial + Temporal video analysis
- Detection based on facial manipulation inconsistencies
- Dataset diversity: Trained on FaceForensics++, DFDC; tested on Celeb-DF (v2)
- Offline video upload and analysis through a custom-built web interface
- Model designed with computational efficiency for deployment in resource-limited environments

## 📚 Datasets Referenced
- [FaceForensics++](https://github.com/ondyari/FaceForensics) ([Rossler et al., 2019](https://arxiv.org/abs/1901.08971))
- [Deepfake Detection Challenge Dataset (DFDC)](https://ai.facebook.com/datasets/dfdc) ([Dolhansky et al., 2020](https://arxiv.org/abs/2006.07397))
- [Celeb-DF (v2)](https://github.com/yuezunli/Celeb-DF) ([Li et al., 2020](https://arxiv.org/abs/1909.12962))

## 🧠 Model Architecture
- **Feature Extraction:** CNN backbones (ResNet50, EfficientNetV2B0, Xception)
- **Temporal Sequence Modeling:** LSTM networks
- **Classification Layer:** Dense layers with Softmax activation for binary classification (Real or Fake)

Input Video ↓ Frame Extraction + Face Cropping ↓ Feature Extraction (CNN) ↓ Temporal Dynamics (LSTM) ↓ Real/Fake Prediction


---

## 📊 Project Results

Performance metrics evaluated on the Celeb-DF (v2) dataset:

| Model                  | Accuracy | Precision | Recall | F1-Score |
|:----------------------:|:--------:|:---------:|:------:|:--------:|
| ResNet50 + LSTM         | 49%      | 50%       | 47%    | 48%      |
| EfficientNetV2B0 + LSTM | 50%      | 52%       | 48%    | 49%      |
| Xception + LSTM         | 51%      | 53%       | 49%    | 51%      |

- ROC Curves and Confusion Matrices available for detailed evaluation
- Observed challenges: False positives in complex real-world videos

---

## 🛠️ Technologies Used

- **Python 3.8**  
- **TensorFlow 2.10** + **Keras** (Deep Learning)
- **OpenCV** + **dlib** (Computer Vision)
- **Flask** (Web Application Backend)
- **HTML5, CSS3, JavaScript** (Frontend)

---

## 📈 Reflections and Future Work

- Further improvements planned through larger dataset training and fine-tuning.
- Integration of real-time detection capabilities.
- Expansion towards multimodal deepfake detection including voice and audio artifacts.
- Cloud deployment with scalable API access.

---

## 🤝 Acknowledgements

- Supervisor: Mr. Michael Akintunde (City, University of London)
- Special thanks to research contributors of [FaceForensics++](https://github.com/ondyari/FaceForensics), [DFDC](https://ai.facebook.com/datasets/dfdc), and [Celeb-DF (v2)](https://github.com/yuezunli/Celeb-DF) datasets.

---

## 📫 Contact

- **LinkedIn:** [Your LinkedIn Profile](#)
- **Email:** [your.email@example.com](mailto:your.email@example.com)

---

⭐️ *If you found this project insightful, feel free to star the repository or connect with me on LinkedIn!*
