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

# 1. loading model, feature selector, and preprocessor
print_progress("Loading model, feature selector, and preprocessor...")
best_rf = load('./model/best_random_forest_model_with_default_embedding.joblib')
selector = load('./model/feature_selector_with_default_embedding.joblib')
preprocessor = load('./model/preprocessor_rf_with_default_embedding.joblib')

# 2. loading data
print_progress("Loading test data...")
df_test = pd.read_json("test.jsonl", lines=True)

# 3. loading embddings
print_progress("Loading test embeddings...")
test_embeddings = np.load("./sembed/test.npy")

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

# 7. merging processed features and embeddings
X_test_with_embeddings = np.hstack([X_test_processed.toarray(), test_embeddings])

# 8. applying feature selection
print_progress("Applying feature selection...")
X_test_selected = selector.transform(X_test_with_embeddings)

# 9. pred
print_progress("Making predictions...")
y_pred = best_rf.predict(X_test_selected)

results_df = pd.DataFrame({
    'case_id': df_test['case_id'],
    'prediction': y_pred
})
results_df['case_number'] = results_df['case_id'].str.extract('(\d+)').astype(int)
single_model_df = results_df.sort_values('case_number')
single_model_df = single_model_df.drop('case_number', axis=1)

# 9. Saving to CSV
print_progress("Saving results to CSV...")
single_model_df.to_csv('RF_default_embdding.csv', index=False)

