# Intel CNN Image Classification

🌍 Deep Learning Model for Classifying Intel Image Dataset

## **📌 Project Overview**  
This project implements two **Convolutional Neural Networks (CNNs)** for **Intel Image Classification**:  
1️⃣ **CNN Model (Trained from Scratch)**  
2️⃣ **CNN Model with Transfer Learning** (using a pre-trained model)

The goal is to classify images into six categories:  
🏢 **Buildings** | 🌲 **Forest** | 🏔 **Glacier** | ⛰ **Mountain** | 🌊 **Sea** | 🛣 **Street**

---

## **📂 Dataset**  
The dataset used is the **Intel Image Classification Dataset**, which contains:  
- **Train Set:** 14,034 images  
- **Test Set:** 3,000 images  
- **Prediction Set:** 7,301 images  

📥 **Download Dataset:** [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

---

🚀 **Features**

*   📊 **Deep Learning Model:** CNN-based classifier implemented using TensorFlow/Keras.
*   🔥 **Transfer Learning:** Uses pre-trained models (e.g., ResNet, VGG16) for improved accuracy.
*   📈 **Performance Metrics:** Evaluates accuracy, precision, recall, and confusion matrix.
*   🏗️ **Modular Code Structure:** Well-organized for easy modification and experimentation.

---

📂 **Project Structure**

```
Intel-CNN-Image-Classification/
├── Dataset/                 # Intel Image Dataset
│   ├── seg_train/           # Training Images
│   ├── seg_test/            # Testing Images
│   ├── seg_pred/            # Unlabeled Images
├── Models/                  # Saved Models (.h5 files)
│   ├── Model With Transfer Learning.h5
│   ├── Model Without Transfer Learning.h5
├── Notebooks/               # Jupyter Notebooks for training & evaluation
│   ├── Classification With Transfer Learning.ipynb
│   ├── Classification Without Transfer Learning.ipynb
├── requirements.txt         # Required Dependencies
├── README.md                # Project Documentation
└── .gitignore               # Git Ignore File
```

---

## **⚙️ Model Architectures**  

### **1️⃣ CNN Model (Trained from Scratch)**  
- **Architecture:** Conv2D → MaxPooling → Conv2D → MaxPooling → Fully Connected  
- **Activation Functions:** ReLU & Softmax  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  

### **2️⃣ CNN Model with Transfer Learning**  
- **Pre-trained Model:** ResNet50 (or VGG16, MobileNet, etc.)  
- **Fine-tuned Layers:** Last few layers trained on the Intel dataset  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  

---

## **📊 Results & Comparison**  

| Model                  | Accuracy | Training Time | Overfitting Risk |
|------------------------|----------|---------------|------------------|
| CNN (From Scratch)     | XX%      | XX min        | XX%              |
| Transfer Learning CNN  | XX%      | XX min        | XX%              |

📈 *Graphs of Training vs. Validation Loss and Accuracy included in the results section.*

---

## **🚀 Installation & Usage**  

### **🔧 Requirements**  
Ensure you have the required dependencies installed:  
```bash
pip install -r requirements.txt
