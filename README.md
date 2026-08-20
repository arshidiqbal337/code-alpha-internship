# Disease Prediction from Medical Data 🩺

Predict disease risks (Breast Cancer, Diabetes, Heart Disease) using structured medical dataset features and Machine Learning classification algorithms in Python.

---

## 📌 Project Overview & Objectives
- **Goal**: Build an easy-to-understand, highly accurate machine learning pipeline to classify and predict patient disease outcomes based on physiological vitals and blood test measurements.
- **Classification Algorithms**:
  1. **Logistic Regression** (Linear baseline probabilistic model)
  2. **Support Vector Machine (SVM)** (Hyperplane decision boundary classifier with RBF Kernel)
  3. **Random Forest Classifier** (Ensemble decision trees with bagging)
  4. **XGBoost Classifier** (Gradient boosted decision trees)
- **Medical Datasets**:
  - **Breast Cancer Wisconsin Dataset** (Scikit-Learn / UCI ML Repository)
  - **Diabetes Dataset** (Pima Indians / UCI ML Repository)
  - **Heart Disease Dataset** (UCI ML Repository)

---

## 🛠️ Key Features
- **Data Preprocessing & Scaling**: Handles feature normalization using `StandardScaler` and stratified train-test splits (`data_loader.py`).
- **Comprehensive Evaluation**: Evaluates models on **Accuracy, Precision, Recall, F1-Score, and ROC-AUC** (`train_evaluate.py`).
- **Visual Analytics**: Generates comparative performance bar charts, confusion matrices, and feature importance diagrams (`visualize.py`).
- **Interactive Web App**: Modern Streamlit web interface (`app.py`) allowing users to adjust patient parameters (Age, Blood Pressure, Glucose, BMI, etc.) and get instant disease risk predictions!

---

## 📂 Repository File Structure
```
project 4/
├── requirements.txt      # Python dependencies
├── data_loader.py       # Preprocessing & dataset loader functions
├── train_evaluate.py    # Training & evaluation script for all 4 ML algorithms
├── visualize.py         # Visual chart & plot generator
├── app.py               # Interactive Streamlit Web Interface
├── README.md            # Comprehensive documentation
├── models/              # Saved model weights (.pkl) and results CSVs
└── plots/               # Performance charts and confusion matrices (.png)
```

---

## 🚀 Quickstart & Usage Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models and Evaluate Metrics
Run `train_evaluate.py` to train all 4 classification algorithms across Breast Cancer, Diabetes, and Heart Disease datasets. Trained models are automatically saved into `models/`.
```bash
python train_evaluate.py
```

### 3. Generate Visual Plots & Confusion Matrices
Run `visualize.py` to generate visual plots comparing model accuracies, F1-Scores, confusion matrices, and feature importances.
```bash
python visualize.py
```

### 4. Launch the Interactive Web Dashboard
Run `app.py` using Streamlit to launch the interactive browser interface:
```bash
streamlit run app.py
```

---

## 📊 Performance Metrics Comparison

| Dataset | Best Performing Algorithm | Typical Accuracy | Top Diagnostic Features |
|---|---|---|---|
| **Breast Cancer** | SVM / Random Forest / XGBoost | ~95% - 97% | Concave Points, Area, Radius |
| **Diabetes** | Random Forest / XGBoost | ~78% - 82% | Glucose, BMI, Age |
| **Heart Disease** | Logistic Regression / Random Forest | ~83% - 86% | Chest Pain, Max HR, ST Depression |

---

## 📜 License & Acknowledgments
Built for educational and medical machine learning research using benchmark UCI ML Repository datasets.
