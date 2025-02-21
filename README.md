# Intel CNN Image Classification

🌍 Deep Learning Model for Classifying Intel Image Dataset

## **📌 Project Overview**  
This project implements two **Convolutional Neural Networks (CNNs)** for **Intel Image Classification**:  
1️⃣ **CNN Model (Trained from Scratch)**  
2️⃣ **CNN Model with Transfer Learning** (using VGG19 as the pre-trained model)

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

## **🚀 Features**

- **Deep Learning Model:** CNN-based classifier implemented using TensorFlow/Keras.
- **Transfer Learning:** Uses VGG19 as the pre-trained model for improved accuracy.
- **Performance Metrics:** Evaluates accuracy, precision, recall, and confusion matrix.
- **Modular Code Structure:** Well-organized for easy modification and experimentation.

---

## **📂 Project Structure**


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
- **Architecture:**  
  - Conv2D → MaxPooling → Conv2D → MaxPooling → Fully Connected Layers  
- **Activation Functions:** ReLU (hidden layers) & Softmax (output layer)  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  

### **2️⃣ CNN Model with Transfer Learning (VGG19)**  
- **Pre-trained Model:** VGG19 (weights trained on ImageNet)  
- **Fine-tuned Layers:** Last few layers trained on the Intel dataset  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  

---

## **📊 Results & Comparison**  

### **Classification Report for CNN With Transfer Learning**

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| buildings    |    0.92   |  0.89  |   0.91   |   437   |
| forest       |    0.96   |  0.99  |   0.98   |   474   |
| glacier      |    0.83   |  0.80  |   0.82   |   553   |
| mountain     |    0.82   |  0.83  |   0.82   |   525   |
| sea          |    0.93   |  0.91  |   0.92   |   510   |
| street       |    0.89   |  0.93  |   0.91   |   501   |
|              |           |        |          |         |
|              |           |        |          |         |
|              |           |        |          |         |
|              |           |        |          |         |
|              |           |        |          |         |
|              |           |        |          |         |
| accuracy     |           |        |   0.89   |   3000  |
| macro avg    |    0.89   |  0.89  |   0.89   |   3000  |
| weighted avg |    0.89   |  0.89  |   0.89   |   3000  |

---

###  **📈 Graphs of Training Loss and Accuracy**
#### **CNN With Transfer Learning**
![Confusion Matrix](Results/Confusion%20Matrix%20TL.png)

#### **CNN Without Transfer Learning**
![Confusion Matrix](Results/Confusion%20Matrix.png)


###  **Confusion Matrices**
#### **CNN With Transfer Learning**
![Loss and Accuracy Graphs](Results/Plots%20TL.png)

#### **CNN Without Transfer Learning**
![Loss and Accuracy Graphs](Results/Plots.png)


---

## **🚀 Installation & Usage**  

### **🔧 Requirements**  
Ensure you have the required dependencies installed:  
```bash
pip install -r requirements.txt
