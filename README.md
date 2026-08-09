<<<<<<< HEAD
# 🏠 House Price Prediction AI

<p align="center">
  <img src="https://img.shields.io/badge/AI%2FML-Regression-blue?style=for-the-badge" alt="AI/ML">
  <img src="https://img.shields.io/badge/Python-3.x-yellow?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-Models-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/CatBoost-Boosting-red?style=for-the-badge" alt="CatBoost">
  <img src="https://img.shields.io/badge/Streamlit-App-ff4b4b?style=for-the-badge&logo=streamlit" alt="Streamlit">
</p>

<p align="center">
  <b>An end-to-end machine learning project for intelligent house price prediction.</b>
</p>

<p align="center">
  Data Preparation → EDA → Model Comparison → Tuning → Evaluation → Deployment
</p>

---

## 📌 About the Project

**House Price Prediction AI** is a complete supervised machine learning regression project that predicts house prices using property characteristics.

Instead of training only one algorithm, the project follows a structured model-selection process:

```text
Raw Dataset
    ↓
Data Cleaning
    ↓
Feature Selection
    ↓
EDA & Visualization
    ↓
Train / Test Split
    ↓
Multiple Regression Models
    ↓
Model Comparison
    ↓
Best Candidate Models
    ↓
Hyperparameter Tuning
    ↓
Cross Validation
    ↓
Final Model
    ↓
Model Evaluation
    ↓
best_model.pkl
    ↓
Streamlit Application
```

---

## 🎯 Project Goals

- Clean and preprocess the house-price dataset.
- Select useful predictive features.
- Perform exploratory data analysis.
- Visualize feature relationships and distributions.
- Compare multiple regression algorithms.
- Evaluate models using **MAE, RMSE, and R²**.
- Tune promising models.
- Apply cross-validation.
- Explore ensemble learning.
- Analyze feature importance and prediction errors.
- Save the final trained model.
- Deploy the model through Streamlit.

---

# 📊 Dataset & Features

### 🎯 Target Variable

```text
price
```

### 🔑 Final Features

| Feature | Description |
|:---|:---|
| `sqft_living` | Interior living area in square feet |
| `sqft_lot` | Total lot area in square feet |
| `floors` | Number of floors |
| `condition` | Overall condition rating |
| `grade` | Construction/design quality rating |
| `yr_built` | Year the property was built |
| `lat` | Geographic latitude |
| `long` | Geographic longitude |

---

# 🔍 Exploratory Data Analysis

EDA is performed before model training to understand the dataset and identify useful patterns.

### Analysis includes

- Dataset structure
- Data types
- Missing values
- Numerical distributions
- Target distribution
- Feature relationships
- Correlation analysis
- Outlier inspection

### 📈 Visualizations

- Histograms
- Box plots
- Scatter plots
- Correlation heatmap
- Distribution plots
- Actual vs Predicted plots
- Residual plots
- Feature importance plots

---

# 🤖 Machine Learning Models

The project compares multiple regression algorithms.

### 1️⃣ Linear & Regularized Models

```text
Linear Regression
Ridge Regression
Lasso Regression
ElasticNet Regression
```

### 2️⃣ Distance / Kernel-Based Models

```text
KNN Regressor
Support Vector Regressor
```

### 3️⃣ Tree-Based Models

```text
Decision Tree
Random Forest
Extra Trees
Gradient Boosting
AdaBoost
```

### 4️⃣ Advanced Boosting

```text
XGBoost
LightGBM
CatBoost
```

This broad comparison helps identify which family of algorithms works best for the dataset.

---

# ⚙️ Preprocessing & Feature Scaling

Models that are sensitive to feature magnitude use `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted **only on training data** to avoid data leakage.

Tree-based algorithms generally do not require feature scaling.

---

# 🔗 Machine Learning Pipeline

The project uses Scikit-Learn pipelines to combine preprocessing and model training.

```text
Input Features
      ↓
Preprocessing
      ↓
ML Model
      ↓
Prediction
      ↓
Evaluation
```

A pipeline also makes deployment safer because the preprocessing steps used during training can be kept together with the trained estimator.

---

# 📏 Model Evaluation

Three main regression metrics are used.

### MAE — Mean Absolute Error

Measures the average absolute prediction error.

```text
Lower = Better
```

### RMSE — Root Mean Squared Error

Penalizes larger errors more strongly.

```text
Lower = Better
```

### R² Score

Measures the proportion of target variance explained by the model.

```text
Higher = Better
```

> **Note:** This is a regression problem, so **R² Score is used instead of classification accuracy**.

---

# 🏆 Model Selection

The performance of all models is stored in a results table:

```text
Model
R² Score
MAE
RMSE
```

Example:

```python
results_df = (
    pd.DataFrame(results)
    .sort_values(by="R2 Score", ascending=False)
    .reset_index(drop=True)
)
```

The strongest model is selected by considering:

```text
High R²
Low MAE
Low RMSE
Good validation performance
Good generalization
```

---

# 🐱 CatBoost Regression

CatBoost is one of the advanced boosting models evaluated in this project.

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    random_state=42,
    verbose=0
)
```

CatBoost builds an ensemble of decision trees sequentially, where later trees focus on improving previous errors.

It is particularly useful for structured/tabular datasets.

---

# 🎛️ Hyperparameter Tuning

Promising models can be optimized using `RandomizedSearchCV`.

### CatBoost search space

```python
param_grid = {
    "iterations": [200, 500, 800],
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "l2_leaf_reg": [1, 3, 5, 10]
}
```

Example:

```python
search = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1
)
```

If tuning gives nearly identical validation performance, additional tuning may not provide meaningful improvement.

---

# 🔄 Cross Validation

Cross-validation evaluates the model over multiple train/validation folds.

```text
              Training Data
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
      Fold 1      Fold 2      Fold 3
     Validate    Validate    Validate
        └───────────┼───────────┘
                    ↓
             Average Score
```

This helps provide a more reliable estimate of model performance.

---

# 🧩 Ensemble Learning

Ensemble learning combines multiple models to potentially improve prediction performance.

## Voting Regressor

```text
Model A ──┐
Model B ──┼──→ Combined Prediction
Model C ──┘
```

The individual predictions can be averaged to obtain the final prediction.

## Stacking Regressor

```text
Model A ──┐
Model B ──┼──→ Meta Model ──→ Final Prediction
Model C ──┘
```

Ensembling is useful only when it improves validation/generalization performance over the individual models.

---

# 📈 Final Model Analysis

## Actual vs Predicted

Shows how closely predicted prices follow actual prices.

## Residual Analysis

```text
Residual = Actual Price - Predicted Price
```

Residual plots can help identify systematic errors and unusual prediction behavior.

## Feature Importance

Tree-based models can be used to analyze which input features contributed most strongly to predictions.

> Feature importance describes model behavior; it does not prove causal relationships.

---

# 💾 Model Serialization

The final trained model is saved as:

```text
models/best_model.pkl
```

Example:

```python
import joblib

joblib.dump(
    best_model,
    "models/best_model.pkl"
)
```

The saved model can then be loaded by the application:

```python
model = joblib.load("models/best_model.pkl")
```

### Important

The Streamlit application **does not retrain the model**.

It simply:

```text
Load trained model
       ↓
Receive user input
       ↓
Predict
       ↓
Display result
```

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit interface designed for user-friendly house-price prediction.

### ✨ UI Features

- 🏠 Modern house-price prediction interface
- 📐 Living-area input
- 🌳 Lot-area input
- 🏢 Floors
- 🛠️ Condition
- ⭐ Grade
- 📅 Year built
- 📍 Latitude
- 🧭 Longitude
- 🚀 Interactive prediction button
- 💰 Animated prediction result
- 📋 Input summary
- 📚 Feature explanations
- ✨ Animated visual design

### User Flow

```text
User enters property details
            ↓
       Streamlit UI
            ↓
    best_model.pkl
            ↓
   Preprocessing + Model
            ↓
        Prediction
            ↓
   Estimated House Price
```

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   HOUSE PRICE DATA   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   DATA PROCESSING    │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  FEATURE SELECTION   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    EDA & VISUALS     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   TRAIN / TEST SPLIT │
                    └──────────┬───────────┘
                               ↓
              ┌────────────────┴────────────────┐
              ↓                                 ↓
       Linear Models                      Tree / Boosting
              │                                 │
              └────────────────┬────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │  MODEL COMPARISON    │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  MODEL TUNING + CV   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    FINAL MODEL       │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   best_model.pkl     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  STREAMLIT APP       │
                    └──────────┬───────────┘
                               ↓
                    🏠 PRICE PREDICTION
```

---

# 📁 Project Structure

```text
HOUSE PRICE PREDICTION/
│
├── 📂 data/
│   ├── 📂 raw/
│   │   └── house_data.csv
│   │
│   └── 📂 processed/
│       └── preprocessed.csv
│
├── 📂 models/
│   └── best_model.pkl
│
├── 📂 notebook/
│   ├── best_model.ipynb
│   ├── data_exploration.ipynb
│   ├── data_preprocessing.ipynb
│   ├── models_train_pipeline.ipynb
│   └── models_train.ipynb
│
├── 📂 reports/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### 🚫 Ignored Generated Files

```text
__pycache__/
catboost_info/
.ipynb_checkpoints/
```

These files are generated during development/training and are not required by the deployed application.

---

# ⚡ Installation & Setup

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd "House Price Prediction"
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

## 3. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

# 🧰 Tech Stack

| Technology | Role |
|:---|:---|
| 🐍 Python | Core programming |
| 🐼 Pandas | Data manipulation |
| 🔢 NumPy | Numerical computation |
| 🤖 Scikit-Learn | ML algorithms & pipelines |
| 🐱 CatBoost | Gradient boosting |
| ⚡ XGBoost | Gradient boosting |
| 💡 LightGBM | Gradient boosting |
| 📊 Matplotlib | Visualization |
| 🎨 Seaborn | Statistical visualization |
| 💾 Joblib | Model serialization |
| 🌐 Streamlit | Web application |

---

# 📚 Concepts Demonstrated

```text
✓ Supervised Learning
✓ Regression
✓ Data Cleaning
✓ Feature Selection
✓ EDA
✓ Data Visualization
✓ Feature Scaling
✓ Train/Test Split
✓ Machine Learning Pipelines
✓ Regularization
✓ KNN Regression
✓ Support Vector Regression
✓ Decision Trees
✓ Random Forest
✓ Extra Trees
✓ Gradient Boosting
✓ AdaBoost
✓ XGBoost
✓ LightGBM
✓ CatBoost
✓ Ensemble Learning
✓ Voting Regression
✓ Stacking
✓ Cross Validation
✓ Hyperparameter Tuning
✓ RandomizedSearchCV
✓ MAE
✓ RMSE
✓ R² Score
✓ Residual Analysis
✓ Feature Importance
✓ Model Serialization
✓ Streamlit Deployment
```

---

# 🚀 Future Improvements

- [ ] FastAPI REST API
- [ ] Docker deployment
- [ ] Cloud deployment
- [ ] SHAP explainability
- [ ] Prediction history
- [ ] Automated retraining
- [ ] Model monitoring
- [ ] CI/CD pipeline
- [ ] MLOps integration
- [ ] Database integration

---

# 👨‍💻 Author

## Vinay Kumar

**B.Tech — Artificial Intelligence & Machine Learning**

**Focus:** AI/ML • Machine Learning • Intelligent Applications

### Project

**🏠 House Price Prediction AI**

An end-to-end machine learning project covering the complete journey:

```text
Raw Data
   ↓
Data Analysis
   ↓
Machine Learning
   ↓
Model Selection
   ↓
Optimization
   ↓
Deployment
```

---

<p align="center">
  <b>🏠 From Raw Data → 🤖 Machine Learning → ⚙️ Optimization → 🚀 Deployment</b>
</p>

<p align="center">
  Built with Python • Scikit-Learn • CatBoost • Streamlit
</p>
=======
    
>>>>>>> 6d9a3ac62d4903a00c14061b8ea60197a4383e6b
