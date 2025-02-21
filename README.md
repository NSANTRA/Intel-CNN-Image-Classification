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

|                  | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| **buildings**    |    0.92   |  0.89  |   0.91   |   437   |
| **forest**       |    0.96   |  0.99  |   0.98   |   474   |
| **glacier**      |    0.83   |  0.80  |   0.82   |   553   |
| **mountain**     |    0.82   |  0.83  |   0.82   |   525   |
| **sea**          |    0.93   |  0.91  |   0.92   |   510   |
| **street**       |    0.89   |  0.93  |   0.91   |   501   |
|                  |           |        |          |         |
| **accuracy**     |           |        |   0.89   |   3000  |
| **macro avg**    |    0.89   |  0.89  |   0.89   |   3000  |
| **weighted avg** |    0.89   |  0.89  |   0.89   |   3000  |

#### **🔹 Key Observations:**
- ✅ **High Overall Accuracy:** 89% – The model performs well across all classes.
- ✅ Forest category has the highest accuracy (**Precision: 0.96**, **Recall: 0.99**, **F1-score: 0.98**) – Very few misclassifications.
- ✅ Buildings, Sea, and Street categories also perform well (**F1-score: ~0.91**).
- ✅ Glacier and Mountain have the lowest scores (**F1-score: ~0.82**) – These categories are harder to classify correctly.

#### **Class-Wise Weights**
|     Category     | Precision | Recall | F1-Score | Observations                                               |
|------------------|-----------|--------|----------|------------------------------------------------------------|
| **buildings**    |    0.92   |  0.89  |   0.91   |   Some buildings misclassified as streets.                 |
| **forest**       |    0.96   |  0.99  |   0.98   |   Best performing class – almost perfect classification.   |
| **glacier**      |    0.83   |  0.80  |   0.82   |   Some glaciers misclassified as mountains.                |
| **mountain**     |    0.82   |  0.83  |   0.82   |   Often confused with glaciers.                            |
| **sea**          |    0.93   |  0.91  |   0.92   |   Often confused with glaciers.                            |
| **street**       |    0.89   |  0.93  |   0.91   |   Often confused with glaciers.                            |

#### **🔹 Key Takeaways:**
- 📌 Transfer Learning significantly boosts accuracy, with an **overall F1-score of 0.89**.
- 📌 Forest classification is near-perfect, while glacier and mountain have the most confusion.
- 📌 Further improvements can be made by refining the model’s ability to differentiate glaciers and mountains.
---

###  **Confusion Matrices**
#### **CNN With Transfer Learning**
![Confusion Matrix](Results/Confusion%20Matrix%20TL.png)
<br>

**Key Observations:**
- ✅ High overall accuracy, fewer misclassifications compared to the second model.
- ✅ Forest category is nearly perfect – 468 out of 474 correctly classified.
- ✅ Buildings misclassified mainly as streets (43 cases) – likely due to urban similarities.
- ✅ Glacier vs. Mountain confusion – 69 glacier images misclassified as mountains.
- ✅ Minimal errors in the sea category – 465 out of 510 correctly classified.

**Major Misclassifications:**
- Glacier mistaken as Mountain (69 cases) – Snow-covered landscapes might be confusing.
- Street misclassified as Buildings (43 cases) – Similar structures in urban settings.
- Sea occasionally confused with Glacier & Mountain – Landscape similarities.

**Takeaway:**
- 🔥 Transfer Learning improves classification significantly, but glacier vs. mountain remains a challenge.

#### **CNN From Sratch**
![Confusion Matrix](Results/Confusion%20Matrix.png)
**Key Observations:**
- ❌ Lower overall accuracy – More misclassifications across most categories.
- ❌ Glacier category struggles the most – 99 glaciers classified as mountains (compared to 69 in TL model).
- ❌ Buildings misclassified as streets (67 cases) – Worse than Transfer Learning model (43 cases).
- ✅ Forest category still performs well – 453 correctly classified out of 474.
- ❌ Sea and Mountain confusion is more frequent than in the Transfer Learning model.

**Major Misclassifications:**
- Glacier confused with Mountain (99 cases) – Worse than Transfer Learning model.
- Street misclassified as Buildings (41 cases) – A bit better than the TL model but still a concern.
- Sea misclassified as Glacier (16 cases) – More than in the TL model.

**Takeaway:**
- 🚨 Training from scratch struggles more, particularly with glacier-mountain and sea-glacier distinctions. Transfer Learning is clearly more effective for this task.

#### **Summary**
**Key Observations and Comparison**
|Metric                   |Transfer Learning CNN                               |CNN From Sratch                                     |
|-------------------------|----------------------------------------------------|----------------------------------------------------|
|**Overall Accuracy**     |Higher (Fewer misclassifications)                   |Lower (More misclassifications)                     |
|**Buildings Accuracy**   |391 correctly classified, 43 misclassified as street|351 correctly classified, 67 misclassified as street|
|**Forest Accuracy**      |468 correctly classified, almost no errors          |453 correctly classified, some errors               |
|**Glacier vs. Mountain** |69 glaciers misclassified as mountains              |99 glaciers misclassified as mountains (worse)      |
|**Sea vs. Mountain**     |Few misclassifications                              |More confusion between sea and mountain             |
|**Street vs. Buildings** |Some confusion but better handling                  |More streets misclassified as buildings             |

**Key Takeaways**
- ✅ Transfer Learning performs better overall
- ✅ CNN from Scratch struggles more with Glacier & Mountain misclassifications
- ✅ Transfer Learning model has more confident predictions, fewer mixed-up cases
- ✅ Fine-tuning the CNN from Scratch might improve its performance

---

###  **📈 Graphs of Training Loss and Accuracy**
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
