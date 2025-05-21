### 1. `README.md`

```markdown
# 🧬 Breast Cancer Prediction using Logistic Regression

This project uses the **Wisconsin Breast Cancer Diagnostic Dataset** from Kaggle to train a machine learning model that predicts whether a tumor is **Benign (B)** or **Malignant (M)** using **Logistic Regression**.

---

## 📌 Dataset

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Format:** CSV
- **Target Column:** `diagnosis` (M = Malignant, B = Benign)

---

## 📊 Features

- Radius, Texture, Perimeter, Area, Smoothness (mean, worst, standard error)
- Compactness, Concavity, Symmetry, Fractal Dimension
- Total: 30 numeric features

---

## 🔧 Project Structure

```

breast-cancer-prediction/
│
├── breast\_cancer\_data.csv         # Dataset file
├── breast\_cancer\_model.ipynb      # Jupyter Notebook (full code)
├── README.md                      # This file


```

---

## 🧠 Model Used

- **Algorithm:** Logistic Regression
- **Library:** `sklearn.linear_model.LogisticRegression`

---

## 🧪 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Example Output:
```

Accuracy: 97.36%
Precision, Recall, F1-score for both classes

````

---

## 🚀 How to Run

1. **Clone this repository** or download the files.
2. Install dependencies:

```bash
pip install -r requirements.txt
````

3. Open and run the Jupyter notebook:

```bash
jupyter notebook breast_cancer_model.ipynb
```

4. The notebook will:

   * Load and preprocess the data
   * Train/test split
   * Train Logistic Regression
   * Predict and evaluate
   * Let you test with new inputs

---

## 🧪 Predict on New Input

Inside the notebook, you can add:

```python
# Example input data (30 features)
input_data = [14.5, 20.5, 96.4, 658.8, 0.103, 0.128, 0.075, 0.040, 0.185, 0.068,
              0.300, 1.230, 2.050, 27.5, 0.006, 0.018, 0.017, 0.010, 0.017, 0.003,
              16.6, 30.5, 113.3, 826.4, 0.132, 0.211, 0.144, 0.065, 0.284, 0.085]

# Convert and predict
import numpy as np
input_np = np.array(input_data).reshape(1, -1)
result = model.predict(input_np)
print("Prediction:", "Malignant" if result[0] == 1 else "Benign")
```

---

## ✅ Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```


---

## 🧑‍💻 Author

* **Name:** Neha Reddy
* **Tech Stack:** Python, Jupyter, scikit-learn
* **Goal:** Predict breast cancer type (Malignant or Benign)

---

## 📜 License

This project is open-source and free to use for educational purposes.

```

---

### 2. `requirements.txt`

```

pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter

---


