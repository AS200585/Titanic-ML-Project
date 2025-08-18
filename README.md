# Titanic Passenger Age Prediction

This project is a machine learning exploration focused on predicting the age of passengers from the Titanic dataset. It serves as a practical exercise in data cleaning, feature engineering, and model evaluation.

This repository documents the end-to-end process, from the data preprocessing to the evaluation of model performance.

---

## Dataset

The project utilizes the `tested.csv` file from the Titanic dataset, which contains various details for each passenger, including their class, fare, and family size.

---

## 🔧 How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AS200585/Titanic-ML-Project.git](https://github.com/AS200585/Titanic-ML-Project.git)
    cd Titanic-ML-Project
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```

3.  **Run the script:**
    ```bash
    python titanic-prediction.py
    ```

---

## Results

An initial set of models was built using the available features. This approach produced extremely high R-squared scores, suggesting near-perfect prediction capabilities.

| Model | Test R² Score (Phase 1) |
| :--- | :--- |
| **Linear Regression** | `0.830` |
| **Lasso Regression** | `0.844` |
| **Gradient Boosting** | `0.999` |

### Analysis: Uncovering Data Leakage

Upon review, it was discovered that the near-perfect scores were a result of **data leakage**. The target variable, `Age`, was accidentally included in the feature set used for training. This meant the model wasn't *predicting* the age; it was simply looking up the answer from the input data, leading to an inflated and misleading performance metric.

---

## 🚀 Future Improvements

The following steps are planned to improve the model's predictive power:

1.  **Smarter Imputation:** Replace the simple `fillna(0)` strategy for missing `Age` values with a more robust method, such as filling with the **median age**.
2.  **Feature Engineering:** Convert categorical columns like `Sex` and `Embarked` into numerical format using one-hot encoding.
3.  **Feature Selection:** Remove non-informative features like `PassengerId` that only add noise to the model.
4.  **Hyperparameter Tuning:** Once the features are improved, fine-tune model parameters to optimize performance and reduce overfitting.
