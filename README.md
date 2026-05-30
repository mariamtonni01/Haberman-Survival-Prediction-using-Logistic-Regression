# 🏥 Haberman Survival Prediction using Logistic Regression

A machine learning project that predicts the **5-year survival outcome of breast cancer patients** following surgery, using the Haberman's Survival dataset and Logistic Regression classification.

---

## 📌 Project Overview

This project applies binary classification to determine whether a patient survived **5 years or longer** after breast cancer surgery. It covers the full ML pipeline — from data loading and exploration to model training, evaluation, and prediction.

---

## 📂 Dataset

- **File:** `synthetic_haberman_dataset.csv`
- **Source:** Based on the [Haberman's Survival Dataset](https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival) (UCI ML Repository)
- **Features:**

| Column | Description |
|--------|-------------|
| Age | Age of patient at time of operation |
| Year | Year of operation (1958–1969, stored as e.g. 2002 for offset) |
| Nodes | Number of positive axillary nodes detected |
| Survival | **Target** — 1: survived ≥ 5 years, 2: survived < 5 years |

---

## 🛠️ Tech Stack

- **Python 3.x**
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — model training and evaluation

---

## 🔁 Workflow

```
Data Loading → Feature Selection → Train/Test Split (80/20)
      ↓
Logistic Regression Training
      ↓
Evaluation: Accuracy, Precision, Recall, Confusion Matrix, ROC-AUC
      ↓
New Patient Prediction
```

---

## 📊 Model Evaluation

The model is evaluated using the following metrics:

- ✅ **Accuracy Score**
- 📋 **Classification Report** (Precision, Recall, F1-score)
- 🟦 **Confusion Matrix** (heatmap via Seaborn)
- 📈 **ROC-AUC Curve** (saved as `Log_ROC.png`)

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/haberman-survival-prediction.git
cd haberman-survival-prediction
```

### 2. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Add the Dataset
Place `synthetic_haberman_dataset.csv` in the project root directory.

### 4. Run the Script
```bash
python haberman_data.py
```

> **Note:** If running in Google Colab, the script uses `google.colab.files.upload()` for file upload. Remove that block and use a local `pd.read_csv()` call when running locally.

---

## 🔮 Sample Prediction

```python
New_obs = pd.DataFrame([[32, 2002, 5]], columns=['Age', 'Year', 'Nodes'])
lg.predict(New_obs)
# Output: [1] → Patient likely survives ≥ 5 years
```

---

## 📁 Project Structure

```
haberman-survival-prediction/
│
├── haberman_data.py              # Main script
├── synthetic_haberman_dataset.csv # Dataset (add manually)
├── Log_ROC.png                   # ROC curve output (auto-generated)
└── README.md
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
