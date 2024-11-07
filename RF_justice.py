import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time
from joblib import dump

def print_progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def extract_justice_features(justices):
    avg_born_year = np.mean([j['born_year'] for j in justices])
    gender_ratio = sum(1 for j in justices if j['gender'] == 'male') / len(justices)
    liberal_ratio = sum(1 for j in justices if j['political_direction'] == 'Liberal') / len(justices)
    return avg_born_year, gender_ratio, liberal_ratio

print_progress("Script execution started")

# 1. Data loading
print_progress("Loading data...")
df = pd.read_json("train.jsonl", lines=True)
print_progress(f"Data loading completed. Number of samples: {len(df)}")

# 2. Feature engineering
print_progress("Performing feature engineering...")
df['argument_date'] = pd.to_datetime(df['argument_date'])
df['year'] = df['argument_date'].dt.year
df['month'] = df['argument_date'].dt.month
df['day'] = df['argument_date'].dt.day
df['dayofweek'] = df['argument_date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['case_duration'] = df['argument_date'].dt.year - df['year']

# Processing new features
df['avg_justice_born_year'], df['justice_gender_ratio'], df['justice_liberal_ratio'] = zip(*df['justices'].map(extract_justice_features))

# 3. Feature definition
categorical_features = ['petitioner', 'respondent', 'petitioner_category', 'respondent_category', 'issue_area', 'petitioner_state', 'respondent_state', 'chief_justice']
numerical_features = ['year', 'court_hearing_length', 'utterances_number', 'case_duration', 'month', 'day', 'dayofweek', 'is_weekend', 
                      'avg_justice_born_year', 'justice_gender_ratio', 'justice_liberal_ratio']

# 4. Creating preprocessing and model pipeline
print_progress("Creating preprocessing and model pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. Preparing feature matrix and target variable
print_progress("Preparing feature matrix and target variable...")
X = df.drop(['successful_appeal', 'argument_date', 'justices'], axis=1)
y = df['successful_appeal'].values

# 6. Splitting the dataset
print_progress("Splitting the dataset into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print_progress(f"Dataset split completed. Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# 7. Defining parameter distributions for RandomizedSearchCV
param_distributions = {
    'classifier__n_estimators': [100, 200, 300, 400, 500],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None]
}

# 8. Performing hyperparameter tuning using RandomizedSearchCV
print_progress("Starting hyperparameter tuning...")
random_search = RandomizedSearchCV(full_pipeline, param_distributions, n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=42, scoring='f1')
random_search.fit(X_train, y_train)
print_progress("Hyperparameter tuning completed")

# 9. Getting the best model
best_model = random_search.best_estimator_
print_progress(f"Best model parameters: {random_search.best_params_}")

# 10. Model evaluation and visualization
print_progress("Evaluating model performance and generating visualizations...")
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("Random Forest Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix with Justice (Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('./graph/confusion_matrix_rf_with_Justice.png')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Justice (Random Forest)')
plt.legend(loc="lower right")
plt.savefig('./graph/roc_curve_rf_ withJustice.png')
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Justice (Random)')
plt.legend(loc="lower left")
plt.savefig('./graph/precision_recall_curve_rf_withJustice.png')
plt.close()

print_progress("Visualizations completed and images saved")

# 11. Feature Importance
feature_importance = best_model.named_steps['classifier'].feature_importances_
feature_names = (best_model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names(categorical_features).tolist() + numerical_features)

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Top 20 Feature Importances (Random Forest without Embeddings)')
plt.tight_layout()
plt.savefig('./graph/feature_importance_rfwithJustice .png')
plt.close()

# 12. Saving the model
print_progress("Saving the model...")
dump(best_model, './model/best_rf_model_with_Justice.joblib')
print_progress("Model saved")

print_progress("Script execution completed")