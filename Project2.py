import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = 'WineQT.csv'
wine_data = pd.read_csv(file_path)

# Display basic information about the dataset
print(wine_data.info())
print(wine_data.describe())
print(wine_data.head())

# Check for missing values
print(wine_data.isnull().sum())

# Separate features and target variable
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Exploratory Data Analysis (EDA)

# Distribution of wine quality
sns.countplot(x='quality', data=wine_data, palette='pastel')
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

# Correlation matrix (simplified heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(wine_data.corr(), annot=False, cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()

# Model Building

# Initialize models
rf = RandomForestClassifier(random_state=42)
sgd = SGDClassifier(random_state=42)
svc = SVC(random_state=42)

# Train models
rf.fit(X_train, y_train)
sgd.fit(X_train, y_train)
svc.fit(X_train, y_train)

# Predict and evaluate
models = {'Random Forest': rf, 'SGD': sgd, 'SVC': svc}
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f'{name} Classification Report:\n')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('\n')

# Hyperparameter Tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
print(f'Best parameters for Random Forest: {grid_rf.best_params_}')

# Feature Importance from Random Forest
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center', color='skyblue')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()

# Confusion Matrix for the best Random Forest model
best_rf = grid_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred_best_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
