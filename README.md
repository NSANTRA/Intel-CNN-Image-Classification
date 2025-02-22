# Intel CNN Image Classification

🌍 Deep Learning Model for Classifying Intel Image Dataset

## **📌 Project Overview**  
This project focuses on developing and evaluating **Convolutional Neural Networks (CNNs)** for the classification of images from the **Intel Image Dataset**.

The dataset consists of images categorized into six distinct classes:
<br>
🏢 **Buildings** | 🌲 **Forest** | 🏔 **Glacier** | ⛰ **Mountain** | 🌊 **Sea** | 🛣 **Street**

Two different approaches are implemented to assess performance and effectiveness:

1. **CNN Model with Transfer Learning** – A model leveraging **VGG19**, a pre-trained deep learning architecture, to enhance feature extraction and improve classification accuracy.
2. **CNN Model Trained from Scratch** – A custom-built convolutional neural network trained without any pre-existing weights.

---

## **📂 Dataset**  
The **Intel Image Dataset** consists of images categorized into six natural and man-made scenery classes. It is a widely used benchmark dataset for scene recognition and classification tasks. The dataset is structured into training, validation, and test sets to facilitate model evaluation.

### **Dataset Structure**
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
├── Dataset/                                            # Intel Image Dataset
│   ├── seg_train/                                      # Training Images
│   ├── seg_test/                                       # Testing Images
│   ├── seg_pred/                                       # Unlabeled Images
├── Models/                                             # Saved Models (.h5 files)
│   ├── Model With Transfer Learning.h5
│   ├── Model Without Transfer Learning.h5
├── Notebooks/                                          # Jupyter Notebooks for training & evaluation
│   ├── Classification With Transfer Learning.ipynb
│   ├── Classification Without Transfer Learning.ipynb
├── requirements.txt                                    # Required Dependencies
├── README.md                                           # Project Documentation
└── .gitignore                                          # Git Ignore File
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
| **Buildings**    |    0.92   |  0.89  |   0.91   |   437   |
| **Forest**       |    0.96   |  0.99  |   0.98   |   474   |
| **Glacier**      |    0.83   |  0.80  |   0.82   |   553   |
| **Mountain**     |    0.82   |  0.83  |   0.82   |   525   |
| **Sea**          |    0.93   |  0.91  |   0.92   |   510   |
| **Street**       |    0.89   |  0.93  |   0.91   |   501   |
|                  |           |        |          |         |
| *accuracy*       |           |        |   0.89   |   3000  |
| *macro avg*      |    0.89   |  0.89  |   0.89   |   3000  |
| *weighted avg*   |    0.89   |  0.89  |   0.89   |   3000  |

#### **🔹 Key Observations:**
- ✅ **High Overall Accuracy:** 89% – The model performs well across all classes.
- ✅ Forest category has the highest accuracy (**Precision: 0.96**, **Recall: 0.99**, **F1-score: 0.98**) – Very few misclassifications.
- ✅ Buildings, Sea, and Street categories also perform well (**F1-score: ~0.91**).
- ✅ Glacier and Mountain have the lowest scores (**F1-score: ~0.82**) – These categories are harder to classify correctly.

#### **Class-Wise Weights**
|     Category     | Precision | Recall | F1-Score | Observations                                               |
|------------------|-----------|--------|----------|------------------------------------------------------------|
| **Buildings**    |    0.92   |  0.89  |   0.91   |   Some Buildings misclassified as Streets.                 |
| **Forest**       |    0.96   |  0.99  |   0.98   |   Best performing class – almost perfect classification.   |
| **Glacier**      |    0.83   |  0.80  |   0.82   |   Some Glaciers misclassified as Mountains.                |
| **Mountain**     |    0.82   |  0.83  |   0.82   |   Often confused with Glaciers.                            |
| **Sea**          |    0.93   |  0.91  |   0.92   |   Often confused with Glaciers.                            |
| **Street**       |    0.89   |  0.93  |   0.91   |   Often confused with Glaciers.                            |

#### **🔹 Key Takeaways:**
- 📌 Transfer Learning significantly boosts accuracy, with an **overall F1-score of 0.89**.
- 📌 Forest classification is near-perfect, while Glacier and Mountain have the most confusion.
- 📌 Further improvements can be made by refining the model’s ability to differentiate Glaciers and Mountains.
---

###  **Confusion Matrices**
#### **CNN With Transfer Learning**
![Confusion Matrix](Results/Confusion%20Matrix%20TL.png)
<br>

**Key Observations:**
- ✅ High overall accuracy, fewer misclassifications compared to the second model.
- ✅ Forest category is nearly perfect – 468 out of 474 correctly classified.
- ✅ Buildings misclassified mainly as Streets (43 cases) – likely due to urban similarities.
- ✅ Glacier vs. Mountain confusion – 69 Glacier images misclassified as Mountains.
- ✅ Minimal errors in the Sea category – 465 out of 510 correctly classified.

**Major Misclassifications:**
- Glacier mistaken as Mountain (69 cases) – Snow-covered landscapes might be confusing.
- Street misclassified as Buildings (43 cases) – Similar structures in urban settings.
- Sea occasionally confused with Glacier & Mountain – Landscape similarities.

**Takeaway:**
- 🔥 Transfer Learning improves classification significantly, but Glacier vs. Mountain remains a challenge.

#### **CNN From Sratch**
![Confusion Matrix](Results/Confusion%20Matrix.png)
<br>

**Key Observations:**
- ❌ Lower overall accuracy – More misclassifications across most categories.
- ❌ Glacier category struggles the most – 99 Glaciers classified as Mountains (compared to 69 in TL model).
- ❌ Buildings misclassified as Streets (67 cases) – Worse than Transfer Learning model (43 cases).
- ✅ Forest category still performs well – 453 correctly classified out of 474.
- ❌ Sea and Mountain confusion is more frequent than in the Transfer Learning model.

**Major Misclassifications:**
- Glacier confused with Mountain (99 cases) – Worse than Transfer Learning model.
- Street misclassified as Buildings (41 cases) – A bit better than the TL model but still a concern.
- Sea misclassified as Glacier (16 cases) – More than in the TL model.

**Takeaway:**
- 🚨 Training from scratch struggles more, particularly with Glacier-Mountain and Sea-Glacier distinctions. Transfer Learning is clearly more effective for this task.

#### **Summary**
**Key Observations and Comparison**
|Metric                   |Transfer Learning CNN                               |CNN From Sratch                                     |
|-------------------------|----------------------------------------------------|----------------------------------------------------|
|**Overall Accuracy**     |Higher (Fewer misclassifications)                   |Lower (More misclassifications)                     |
|**Buildings Accuracy**   |391 correctly classified, 43 misclassified as Street|351 correctly classified, 67 misclassified as Street|
|**Forest Accuracy**      |468 correctly classified, almost no errors          |453 correctly classified, some errors               |
|**Glacier vs. Mountain** |69 Glaciers misclassified as Mountains              |99 Glaciers misclassified as Mountains (worse)      |
|**Sea vs. Mountain**     |Few misclassifications                              |More confusion between Sea and Mountain             |
|**Street vs. Buildings** |Some confusion but better handling                  |More Streets misclassified as Buildings             |

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

## **Technologies and Tools Used**
- **IDE:** Jupyter Lab
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow/Keras
- **Data Processing:** OpenCV, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Hardware Acceleration:** GPU (CUDA-enabled for TensorFlow)

---

## **🚀 Installation & Usage**  

### **🔧 Requirements**  
Ensure Anaconda is installed, if not you can download from [Anaconda](https://www.anaconda.com/download/success)

Once Anaconda is installed, use the **Anaconda Prompt** to run the following commands:

Create a conda environment:
```bash
conda env create -f "Tensorflow.yml"
```

Activate the newly created environment:
```bash
conda activate Tensorflow
```