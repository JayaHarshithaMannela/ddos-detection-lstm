# 🚀 Multiclass DDoS Attack Detection using LSTM

## 📌 Overview

This project implements a **Multiclass Network Intrusion Detection System** using a **Long Short-Term Memory (LSTM)** deep learning model.

It classifies network traffic into multiple categories such as:

* Benign traffic
* Various DDoS attack types

The system includes:

* Model training
* Random sample prediction
* Comprehensive evaluation
* Performance metrics analysis

---

## 🎯 Features

* ✅ Multiclass classification using LSTM
* ✅ Automated preprocessing and scaling
* ✅ Model saving and reuse
* ✅ Random sample prediction
* ✅ Detailed evaluation metrics
* ✅ Class-wise performance analysis
* ✅ Modular code structure

---

## 📊 Dataset

This project uses the **CIC-DDoS2019 Dataset**.

🔗 https://cicresearch.ca//CICDataset/CICDDoS2019/

### 📌 Description

* Real-world network traffic
* Includes multiple DDoS attacks:

  * SYN Flood
  * UDP Flood
  * LDAP
  * DNS
  * NetBIOS
  * SNMP
  * MSSQL
* Includes **benign traffic**

---

## ⚙️ Preprocessing

* Cleaned dataset
* Removed inconsistencies
* Converted into multiclass format

Saved as:

```
data/multiclass_dataset.csv
```

---

## 📂 Project Structure

```
ddos-detection-lstm/
│
├── data/
│   ├── multiclass_dataset.csv
│   └── test_dataset.csv
│
├── model/
│   ├── multiclass_model.keras
│   └── multiclass_scaler.pkl
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
│
├── train.py
├── prediction.py
├── evaluation.py
├── evaluation_vis.py
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/JayaHarshithaMannela/ddos-detection-lstm.git
cd ddos-detection-lstm
```

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow joblib matplotlib
```

---

## 🧠 Model Architecture

* LSTM (64 units, return_sequences=True)
* Dropout (0.2)
* LSTM (32 units)
* Dropout (0.2)
* Dense (Softmax output layer)

---

## ▶️ Usage

### 🔹 Train Model

```bash
python train.py
```

This will:

* Train the LSTM model
* Save model & scaler
* Generate test dataset

---

### 🔹 Run Prediction

```bash
python prediction.py
```

Outputs:

* Actual class
* Predicted class
* Confidence score
* Class probabilities

---

### 🔹 Evaluate Model

```bash
python evaluation.py
```

Outputs:

* Accuracy
* Precision (Macro & Weighted)
* Recall (Macro & Weighted)
* F1 Score (Macro & Weighted)
* ROC-AUC
* Confusion Matrix
* Cohen Kappa Score
* MCC (Matthews Correlation Coefficient)
* Log Loss
* Class-wise Accuracy
* Saved graphs in `results/`

---

### 📊 Visualization Outputs

The evaluation process includes graphical analysis to better understand model performance.

The following graphs are generated:

* Confusion Matrix Heatmap
* ROC Curve (One-vs-Rest for multiclass)
* Precision-Recall Curve

📁 All graphs are automatically saved in the `results/` folder.

These visualizations are especially useful for analyzing performance on imbalanced datasets.

---

### 📊 Visualization Script

A separate script is provided for generating only visual outputs:

```bash
python evaluation_vis.py
```

This script:

* Loads the trained model and test dataset
* Generates performance graphs
* Saves them in the `results/` directory

---

## 📈 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Cohen Kappa Score
* Matthews Correlation Coefficient
* Log Loss

---

## 📊 Results

*(Update after running evaluation)*

| Metric    | Value |
| --------- | ----- |
| Accuracy  | XX    |
| Precision | XX    |
| Recall    | XX    |
| F1 Score  | XX    |
| ROC-AUC   | XX    |

---

## ⚠️ Important Note

The dataset may be **imbalanced**, which can lead to:

* High overall accuracy
* Poor performance on minority classes

👉 Always check:

* Class-wise accuracy
* Precision & Recall per class

---

## 🚀 Future Improvements

* Apply SMOTE for class balancing
* Hyperparameter tuning
* Real-time intrusion detection system
* Deployment using Flask / FastAPI
* Integration with IoT systems

---

## 🧑‍💻 Author

**Jaya Harshitha Mannela**
B.Tech CSE (AIML)

---

## ⭐ Acknowledgment

Developed for academic and research purposes in:

* Cybersecurity
* Machine Learning

---

## 📌 License

Free to use for educational purposes.

