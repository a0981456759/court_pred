import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import time
from joblib import dump

def print_progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


# 1. Data Loading

df = pd.read_json("train.jsonl", lines=True)

# 2. Feature Engineering

df['argument_date'] = pd.to_datetime(df['argument_date'])
df['year'] = df['argument_date'].dt.year
df['month'] = df['argument_date'].dt.month
df['day'] = df['argument_date'].dt.day
df['dayofweek'] = df['argument_date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['case_duration'] = df['argument_date'].dt.year - df['year']

# 3. Define Features
categorical_features = ['petitioner', 'respondent', 'petitioner_category', 'respondent_category', 'issue_area', 'petitioner_state', 'respondent_state']
numerical_features = ['year', 'court_hearing_length', 'utterances_number', 'case_duration', 'month', 'day', 'dayofweek', 'is_weekend']

# 4. Create Preprocessing and Model Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k=200)),
    ('classifier', MLPClassifier(random_state=42, max_iter=1000))
])

# 5. Prepare Feature Matrix and Target Variable

X = df.drop('successful_appeal', axis=1)
y = df['successful_appeal'].values

# 6. Split the Dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Define Parameter Grid
param_grid = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'classifier__activation': ['relu', 'tanh'],
    'classifier__alpha': [0.0001, 0.001, 0.01],
    'classifier__learning_rate': ['constant', 'adaptive'],
}

# 8. Hyperparameter Tuning with GridSearchCV

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1')
grid_search.fit(X_train, y_train)

# 9. Get Best Model
best_model = grid_search.best_estimator_
print_progress(f"params: {grid_search.best_params_}")

# 10. Visualization
print_progress("vis...")
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("MLP Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (MLP)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('./graph/confusion_matrix_mlp.png')
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
plt.title('ROC Curve (MLP)')
plt.legend(loc="lower right")
plt.savefig('./graph/roc_curve_mlp.png')
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
plt.title('Precision-Recall Curve (MLP)')
plt.legend(loc="lower left")
plt.savefig('./graph/precision_recall_curve_mlp.png')

dump(best_model, './model/best_mlp_model.joblib')

