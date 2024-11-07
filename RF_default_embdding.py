import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
from joblib import dump

def print_progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

print_progress("Starting script")

# 1. Load data
print_progress("Loading data...")
df = pd.read_json("../train.jsonl", lines=True)
print_progress(f"Amount of features: {len(df)}")

# Load embedding file
print_progress("Loading embedding file...")
embeddings = np.load("../sembed/train.npy")
print_progress(f"Shape: {embeddings.shape}")

# 2. Feature engineering
print_progress("Feature engineering...")
df['argument_date'] = pd.to_datetime(df['argument_date'])
df['year'] = df['argument_date'].dt.year
df['month'] = df['argument_date'].dt.month
df['day'] = df['argument_date'].dt.day
df['dayofweek'] = df['argument_date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['case_duration'] = df['argument_date'].dt.year - df['year']

# 3. Define features
categorical_features = ['petitioner', 'respondent', 'petitioner_category', 'respondent_category', 'issue_area', 'petitioner_state', 'respondent_state']
numerical_features = ['year', 'court_hearing_length', 'utterances_number', 'case_duration', 'month', 'day', 'dayofweek', 'is_weekend']

# 4. Data preprocessing
print_progress("Data Preprocessing...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# 5. Prepare feature matrix and target variable
print_progress("Feature matrix...")
X_processed = preprocessor.fit_transform(df)
X = np.hstack([X_processed.toarray(), embeddings])  # Merge processed features with embeddings
y = df['successful_appeal'].values
print_progress(f"Amount of matrix: {X.shape[1]}")

# 6. Split
print_progress("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print_progress(f"Training set: {X_train.shape[0]}, Testing set: {X_test.shape[0]}")

# 7. SMOTE
print_progress("SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print_progress(f"Amount of new dataset: {X_train_resampled.shape[0]}")

# 8. Feature selection
print_progress("Feature selection...")
selector = SelectKBest(score_func=f_classif, k=200)
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector.transform(X_test)
print_progress(f"Amount of selection: {X_train_selected.shape[1]}")

# 9. Define Random Forest model and parameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# 10. Use RandomizedSearchCV for hyperparameter tuning
print_progress("Tuning...")
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                   n_iter=20, cv=5, verbose=2, random_state=42, n_jobs=-1, scoring='f1')
random_search.fit(X_train_selected, y_train_resampled)
print_progress("Finished")

# 11. Get the best model
best_rf = random_search.best_estimator_
print_progress(f"Params: {random_search.best_params_}")

# 12. Model evaluation and visualization
print_progress("Visualization...")
y_pred = best_rf.predict(X_test_selected)
y_pred_proba = best_rf.predict_proba(X_test_selected)[:, 1]

print("Random Forest Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Default Text Embedding (Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('./graph/confusion_matrix_rf_with_default_embedding.png')
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve Default Text Embedding (Random Forest)')
plt.legend(loc="lower right")
plt.savefig('./graph/roc_curve_rf_with_default_embedding.png')
plt.close()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Default Text Embedding (Random Forest)')
plt.legend(loc="lower left")
plt.savefig('./graph/precision_recall_curve_rf_with_default_embedding.png')
plt.close()

# 13. Save the model
print_progress("Save model...")
dump(best_rf, './model/best_random_forest_model_with_default_embedding.joblib')
dump(selector, './model/feature_selector_with_default_embedding.joblib')
dump(preprocessor, './model/preprocessor_rf_with_default_embedding.joblib')
print_progress("Saved")
