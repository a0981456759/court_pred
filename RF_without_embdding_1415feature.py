import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import time
from joblib import dump

def print_progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# 1. Load data
df = pd.read_json("train.jsonl", lines=True)

# 2. Feature engineering

df['argument_date'] = pd.to_datetime(df['argument_date'])
df['year'] = df['argument_date'].dt.year
df['month'] = df['argument_date'].dt.month
df['day'] = df['argument_date'].dt.day
df['dayofweek'] = df['argument_date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['case_duration'] = df['argument_date'].dt.year - df['year']

# Process new features
df['decision_date'] = pd.to_datetime(df['decision_date'])
df['decision_year'] = df['decision_date'].dt.year
df['decision_month'] = df['decision_date'].dt.month
df['decision_day'] = df['decision_date'].dt.day
df['decision_to_argument_days'] = (df['decision_date'] - df['argument_date']).dt.days

# 3. Define features
categorical_features = ['petitioner', 'respondent', 'petitioner_category', 'respondent_category', 'issue_area', 'petitioner_state', 'respondent_state']
numerical_features = ['year', 'court_hearing_length', 'utterances_number', 'case_duration', 'month', 'day', 'dayofweek', 'is_weekend', 
                      'majority_ratio', 'decision_year', 'decision_month', 'decision_day', 'decision_to_argument_days']

# 4. Create preprocessing and model pipeline

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
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. Prepare feature matrix and target variable

X = df.drop(['successful_appeal', 'argument_date', 'decision_date'], axis=1)
y = df['successful_appeal'].values

# 6. Split the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Define parameter grid
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# 8. Perform hyperparameter tuning using GridSearchCV

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1')
grid_search.fit(X_train, y_train)

# 9. Get the best model
best_model = grid_search.best_estimator_
print_progress(f"params: {grid_search.best_params_}")

# 10. Model evaluation and visualization
print_progress("vis...")
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("Random Forest Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix plus more features (Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('./graph/confusion_matrix_rf_without_embeddings_1415feature.png')
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
plt.title('ROC Curve plus more features (Random Forest)')
plt.legend(loc="lower right")
plt.savefig('./graph/roc_curve_rf_without_embeddings_1415feature.png')
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
plt.title('Precision-Recall Curve plus more features (Random Forest)')
plt.legend(loc="lower left")
plt.savefig('./graph/precision_recall_curve_rf_without_embeddings_1415feature.png')
plt.close()

print_progress("saved")

# 11. Feature importance
# feature_importance = best_model.named_steps['classifier'].feature_importances_
# feature_names = (best_model.named_steps['preprocessor']
#                  .named_transformers_['cat']
#                  .named_steps['onehot']
#                  .get_feature_names(categorical_features).tolist() + numerical_features)

# feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
# feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(20)

# plt.figure(figsize=(12, 8))
# sns.barplot(x='importance', y='feature', data=feature_importance_df)
# plt.title('Top 20 Feature Importances (Random Forest without Embeddings)')
# plt.tight_layout()
# plt.savefig('./graph/feature_importance_rf_without_embeddings_1415feature.png')
# plt.close()

# 12. Save the model
print("saving...")
dump(best_model, './model/best_rf_model_without_embeddings_1415feature.joblib')
print("saved")
