# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

from imblearn.over_sampling import SMOTE  # To handle class imbalance

import requests
import zipfile
from io import BytesIO

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# 1. Download and Extract Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
r = requests.get(url)
z = zipfile.ZipFile(BytesIO(r.content))
z.extractall()

# 2. Load Data (Math student performance)
df = pd.read_csv('student-mat.csv', sep=';')

# 3. Exploratory Data Analysis (EDA)
print("Dataset shape:", df.shape)
print("\nData sample:\n", df.head())

print("\nData info:")
print(df.info())

print("\nTarget variable (G3) distribution:")
sns.histplot(df['G3'], bins=20, kde=True)
plt.title('Distribution of Final Grade (G3)')
plt.show()

# Define dropout target: 1 if final grade ≤ 10, else 0
df['dropout'] = (df['G3'] <= 10).astype(int)

print("\nDropout class distribution:")
print(df['dropout'].value_counts())
sns.countplot(x='dropout', data=df)
plt.title('Dropout Class Distribution')
plt.show()

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 4. Encode Categorical Features
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 5. Feature/Target split
X = df.drop(['G3', 'dropout'], axis=1)
y = df['dropout']

# 6. Handle Class Imbalance using SMOTE
print(f"Before SMOTE: {y.value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

# 7. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res)

# 9. Model Definition and Hyperparameter Tuning with Cross-Validation

# Logistic Regression hyperparameter grid
lr_params = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']}

lr = LogisticRegression(random_state=42)
lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='f1', n_jobs=-1)
lr_grid.fit(X_train, y_train)

print("\nBest Logistic Regression parameters:", lr_grid.best_params_)

# Random Forest hyperparameter grid
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("\nBest Random Forest parameters:", rf_grid.best_params_)

# 10. Evaluate best models on test set

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"{model_name} Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Dropout', 'Dropout'], yticklabels=['No Dropout', 'Dropout'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

    # ROC Curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

# Evaluate Logistic Regression
evaluate_model(lr_grid.best_estimator_, X_test, y_test, "Logistic Regression")

# Evaluate Random Forest
evaluate_model(rf_grid.best_estimator_, X_test, y_test, "Random Forest")

# 11. Feature Importance (Random Forest)
feature_names = X.columns
importances = rf_grid.best_estimator_.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=feat_imp[:15], y=feat_imp.index[:15])
plt.title("Top 15 Feature Importances - Random Forest")
plt.show()

# --- End of Project ---
