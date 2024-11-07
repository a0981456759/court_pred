import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import time

def print_progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

print_progress("Script execution started")

# 1. loading model and preprocessor
print_progress("Loading model and preprocessor...")
best_rf = load('./model/best_rf_model_with_ownembeddings_randomized.joblib')
preprocessor = load('./model/preprocessor_rf_with_ownembeddings_randomized.joblib')

# 2. loading data
print_progress("Loading test data...")
df_test = pd.read_json("test.jsonl", lines=True)

# 3. loading embeddings
print_progress("Loading test embeddings...")
all_embeddings = np.load("court_hearing_embeddings.npy")  # 假设这包含所有嵌入

# 4. features engineering
print_progress("Performing feature engineering...")
df_test['argument_date'] = pd.to_datetime(df_test['argument_date'])
df_test['year'] = df_test['argument_date'].dt.year
df_test['month'] = df_test['argument_date'].dt.month
df_test['day'] = df_test['argument_date'].dt.day
df_test['dayofweek'] = df_test['argument_date'].dt.dayofweek
df_test['is_weekend'] = df_test['dayofweek'].isin([5, 6]).astype(int)
df_test['case_duration'] = df_test['argument_date'].dt.year - df_test['year']

# 5. features
categorical_features = ['petitioner', 'respondent', 'petitioner_category', 'respondent_category', 'issue_area', 'petitioner_state', 'respondent_state']
numerical_features = ['year', 'court_hearing_length', 'utterances_number', 'case_duration', 'month', 'day', 'dayofweek', 'is_weekend']

X_test = df_test.drop(['case_id'], axis=1)

# 6. applying preprocessor
print_progress("Applying preprocessor...")
X_test_processed = preprocessor.transform(X_test)

print(f"Shape of X_test_processed: {X_test_processed.shape}")
print(f"Shape of all_embeddings: {all_embeddings.shape}")

# choose embeddings
test_embeddings = all_embeddings[-len(df_test):]

print(f"Shape of test_embeddings: {test_embeddings.shape}")

# 7. merging processed features and embeddings
X_test_with_embeddings = np.hstack([X_test_processed.toarray(), test_embeddings])

print(f"Shape of X_test_with_embeddings: {X_test_with_embeddings.shape}")

# 8. applying feature selection
print_progress("Applying feature selection...")
selector = SelectKBest(score_func=f_classif, k=200)
X_test_selected = selector.fit_transform(X_test_with_embeddings, np.zeros(X_test_with_embeddings.shape[0]))  # 使用虚拟标签

# 9. pred
print_progress("Making predictions...")
y_pred = best_rf.predict(X_test_selected)
y_pred_proba = best_rf.predict_proba(X_test_selected)[:, 1]

# 10. outcome DataFrame
results_df = pd.DataFrame({
    'case_id': df_test['case_id'],
    'prediction': y_pred
})
results_df['case_number'] = results_df['case_id'].str.extract('(\d+)').astype(int)
single_model_df = results_df.sort_values('case_number')
single_model_df = single_model_df.drop('case_number', axis=1)

# 9. Saving to CSV
print_progress("Saving results to CSV...")
single_model_df.to_csv('RF_own_embdding.csv', index=False)

