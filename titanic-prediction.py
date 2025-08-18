import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

# %%
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

# Read the CSV file using pandas
df = pd.read_csv("tested.csv")
print(df)
print(df.columns)

# Check for missing values before fillna
print("\nMissing values before fillna:")
print(df.isna().sum())

df.fillna(0, inplace=True)
# Verify changes
print("\nDataFrame after fillna:")
print(df)

print("\nMissing values after fillna:")
print(df.isna().sum())


# %%
# Define features (X) and target (y)
# Replace 'target_column' with the name of your target column
X = df.drop(columns=["Sex", "Name", "Embarked", "Cabin", "Ticket"])  # Features
Y = df["Age"]  # Target

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Verify the shapes of the splits
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", Y_train.shape)
print("Testing target shape:", Y_test.shape)

# Log transform Age
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()
X_train_encoded["Age"] = np.log(X_train_encoded["Age"] + 1)
X_test_encoded["Age"] = np.log(X_test_encoded["Age"] + 1)

# %%
print(f"X_train_encoded shape: {X_train_encoded.shape}")
print(f"X_test_encoded shape: {X_test_encoded.shape}")

# %%
# Adding Age-to-Pclass ratio
X_train_encoded["Age_Pclass_ratio"] = X_train_encoded["Age"] / (
    X_train_encoded["Pclass"] + 1
)
X_test_encoded["Age_Pclass_ratio"] = X_test_encoded["Age"] / (
    X_test_encoded["Pclass"] + 1
)

# %%
print(f"NaN in X_train_encoded: {X_train_encoded.isna().sum()}")
print(f"Infinite in X_train_encoded: {np.isinf(X_train_encoded).sum()}")

from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Scaling features
standard_scaler = StandardScaler()
X_train_scaled = standard_scaler.fit_transform(X_train_encoded)
X_test_scaled = standard_scaler.transform(X_test_encoded)

# %%
x_train = X_train_scaled
x_test = X_test_scaled
y_train = Y_train
y_test = Y_test

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# %%
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor

# Model training using Linear Regression
reg = LinearRegression()
lr = Lasso()
gbr = GradientBoostingRegressor()
reg.fit(x_train, y_train)
lr.fit(x_train, y_train)
gbr.fit(x_train, y_train)

# %%
# Evaluation
score = reg.score(x_test, y_test)
score_2 = lr.score(x_test, y_test)
score_3 = gbr.score(x_test, y_test)
print(f"R squared Score for log(Age) - LinearRegression: {score}\n")
print(f"R squared Score for log(Age) - Lasso: {score_2}\n")
print(f"R squared Score for log(Age) - GradientBoostingRegressor: {score_3}\n")

# %%
print(f"R² Score (train): {reg.score(x_train, y_train)}")
print(f"R² Score (test): {reg.score(x_test, y_test)}\n")
print(f"R² Score 2(train): {lr.score(x_train, y_train)}")
print(f"R² Score 2(test): {lr.score(x_test, y_test)}\n")
print(f"R² Score 3(train): {gbr.score(x_train, y_train)}")
print(f"R² Score 3(test): {gbr.score(x_test, y_test)}\n")


# %%
# Predictions
y_pred = reg.predict(x_test)
y_pred_2 = lr.predict(x_test)
y_pred_3 = gbr.predict(x_test)
# Display predictions
print(f"Predictions: {y_pred[:10]}\n")  # Display first 10 predictions
print(f"Predictions 2: {y_pred_2[:10]}\n")  # Display first 10 predictions
print(f"Predictions 3: {y_pred_3[:10]}\n")  # Display first 10 predictions


# %%
evaluation_df = pd.DataFrame({"Actual Age": y_test, "Predicted Age": y_pred})
print("\nEvaluation DataFrame:")
print(evaluation_df.head(10))  # Display first 10 rows of the evaluation DataFrame
# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs Predicted Age 1")
plt.grid()
plt.show()

evaluation_df_2 = pd.DataFrame({"Actual Age": y_test, "Predicted Age": y_pred_2})
print("\nEvaluation DataFrame 2:")
print(evaluation_df_2.head(10))  # Display first 10 rows of the evaluation DataFrame
# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_2, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs Predicted Age 2")
plt.grid()
plt.show()

evaluation_df_3 = pd.DataFrame({"Actual Age": y_test, "Predicted Age": y_pred_3})
print("\nEvaluation DataFrame 3:")
print(evaluation_df_3.head(10))  # Display first 10 rows of the evaluation DataFrame
# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_3, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs Predicted Age 3")
plt.grid()
plt.show()


# %%
# Evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse_2 = mean_squared_error(y_test, y_pred_2)
r2_2 = r2_score(y_test, y_pred_2)
mse_3 = mean_squared_error(y_test, y_pred_3)
r2_3 = r2_score(y_test, y_pred_3)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}\n")
print(f"Mean Squared Error 2: {mse_2}")
print(f"R² Score 2: {r2_2}\n")
print(f"Mean Squared Error 3: {mse_3}")
print(f"R² Score 3: {r2_3}\n")
