import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

for feature in features:
    if feature not in train_data.columns:
        raise ValueError(f"Feature column '{feature}' not found in training data.")
if target not in train_data.columns:
    raise ValueError(f"Target column '{target}' not found in training data.")

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train_split)

y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

print("Training Data:")
print("Mean Squared Error:", mean_squared_error(y_train_split, y_train_pred))
print("R2 Score:", r2_score(y_train_split, y_train_pred))

print("\nValidation Data:")
print("Mean Squared Error:", mean_squared_error(y_val_split, y_val_pred))
print("R2 Score:", r2_score(y_val_split, y_val_pred))

submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': y_test_pred
})

try:
    submission.to_csv('submission.csv', index=False)
    print("Predictions have been saved to 'submission.csv'.")
except Exception as e:
    print(f"An error occurred while saving the submission file: {e}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train_split, y_train_pred, color='blue', alpha=0.5)
plt.plot([y_train_split.min(), y_train_split.max()], [y_train_split.min(), y_train_split.max()], color='red', linestyle='--')
plt.title('Training Set: True vs Predicted')
plt.xlabel('True SalePrice')
plt.ylabel('Predicted SalePrice')

plt.subplot(1, 2, 2)
plt.scatter(y_val_split, y_val_pred, color='green', alpha=0.5)
plt.plot([y_val_split.min(), y_val_split.max()], [y_val_split.min(), y_val_split.max()], color='red', linestyle='--')
plt.title('Validation Set: True vs Predicted')
plt.xlabel('True SalePrice')
plt.ylabel('Predicted SalePrice')

plt.tight_layout()
plt.show()
