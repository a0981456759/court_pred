import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import time

def print_progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def extract_justice_features(justices):
    avg_born_year = np.mean([j['born_year'] for j in justices])
    gender_ratio = sum(1 for j in justices if j['gender'] == 'male') / len(justices)
    liberal_ratio = sum(1 for j in justices if j['political_direction'] == 'Liberal') / len(justices)
    return avg_born_year, gender_ratio, liberal_ratio

print_progress("Script execution started")

# 1. loading model
print_progress("Loading the model...")
best_model = load('./model/best_rf_model_with_Justice.joblib')

# 2. loading data
print_progress("Loading and preprocessing test data...")
df_test = pd.read_json("test.jsonl", lines=True)

# 3. features engineering
print_progress("Performing feature engineering...")
df_test['argument_date'] = pd.to_datetime(df_test['argument_date'])
df_test['year'] = df_test['argument_date'].dt.year
df_test['month'] = df_test['argument_date'].dt.month
df_test['day'] = df_test['argument_date'].dt.day
df_test['dayofweek'] = df_test['argument_date'].dt.dayofweek
df_test['is_weekend'] = df_test['dayofweek'].isin([5, 6]).astype(int)
df_test['case_duration'] = df_test['argument_date'].dt.year - df_test['year']

# jsutice features
df_test['avg_justice_born_year'], df_test['justice_gender_ratio'], df_test['justice_liberal_ratio'] = zip(*df_test['justices'].map(extract_justice_features))

# 4. features
categorical_features = ['petitioner', 'respondent', 'petitioner_category', 'respondent_category', 'issue_area', 'petitioner_state', 'respondent_state', 'chief_justice']
numerical_features = ['year', 'court_hearing_length', 'utterances_number', 'case_duration', 'month', 'day', 'dayofweek', 'is_weekend', 
                      'avg_justice_born_year', 'justice_gender_ratio', 'justice_liberal_ratio']

X_test = df_test.drop(['case_id', 'argument_date', 'justices'], axis=1)

# 5. pred
print_progress("Making predictions...")
y_pred = best_model.predict(X_test)

results_df = pd.DataFrame({
    'case_id': df_test['case_id'],
    'prediction': y_pred
})
results_df['case_number'] = results_df['case_id'].str.extract('(\d+)').astype(int)
single_model_df = results_df.sort_values('case_number')
single_model_df = single_model_df.drop('case_number', axis=1)

# 9. Saving to CSV
print_progress("Saving results to CSV...")
single_model_df.to_csv('RF_justice.csv', index=False)
