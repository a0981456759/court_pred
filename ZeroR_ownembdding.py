import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time

def print_progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

class MajorityClassPredictor:
    def __init__(self):
        self.majority_class = None

    def fit(self, y):
        self.majority_class = np.argmax(np.bincount(y))

    def predict(self, X):
        return np.full(X.shape[0], self.majority_class)

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], 2))
        probas[:, self.majority_class] = 1
        return probas


# 1. Loading data

df = pd.read_json("train.jsonl", lines=True)


# Loading embdding file
embeddings = np.load("court_hearing_embeddings.npy")


# 2. Feature engineering
df['argument_date'] = pd.to_datetime(df['argument_date'])
df['year'] = df['argument_date'].dt.year
df['month'] = df['argument_date'].dt.month
df['day'] = df['argument_date'].dt.day
df['dayofweek'] = df['argument_date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['case_duration'] = df['argument_date'].dt.year - df['year']

# 3. Feature definition
numerical_features = ['year', 'court_hearing_length', 'utterances_number', 'case_duration', 'month', 'day', 'dayofweek', 'is_weekend']
# data stacking
X = np.hstack([df[numerical_features].values, embeddings])
y = df['successful_appeal'].values

# 4. Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Training

majority_predictor = MajorityClassPredictor()
majority_predictor.fit(y_train)

y_pred = majority_predictor.predict(X_test)
y_pred_proba = majority_predictor.predict_proba(X_test)[:, 1]
# 6. Eva

print("Majority Class Prediction Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. vis

# cm
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Zero R Prediction with Embeddings)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('./graph/confusion_matrix_zeror_with_own_embedding.png')
plt.close()

# ROC 
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Zero R Prediction)')
plt.legend(loc="lower right")
plt.savefig('./graph/roc_curve_zeror_with_own_embedding.png')
plt.close()

# Precision-Recall 
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Zero R Prediction)')
plt.legend(loc="lower left")
plt.savefig('./graph/precision_recall_curve_zeror_with_own_embedding.png')
plt.close()
