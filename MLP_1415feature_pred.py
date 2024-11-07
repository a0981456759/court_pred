import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from joblib import load
import time

def print_progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# features
categorical_features = ['petitioner', 'respondent', 'petitioner_category', 'respondent_category', 'issue_area', 'petitioner_state', 'respondent_state']
numerical_features = ['year', 'court_hearing_length', 'utterances_number', 'case_duration', 'month', 'day', 'dayofweek', 'is_weekend', 
                      'majority_ratio', 'decision_year', 'decision_month', 'decision_day', 'decision_to_argument_days']

def preprocess_data(df):
    df['argument_date'] = pd.to_datetime(df['argument_date'])
    df['year'] = df['argument_date'].dt.year
    df['month'] = df['argument_date'].dt.month
    df['day'] = df['argument_date'].dt.day
    df['dayofweek'] = df['argument_date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['case_duration'] = df['argument_date'].dt.year - df['year']

    df['decision_date'] = pd.to_datetime(df['decision_date'])
    df['decision_year'] = df['decision_date'].dt.year
    df['decision_month'] = df['decision_date'].dt.month
    df['decision_day'] = df['decision_date'].dt.day
    df['decision_to_argument_days'] = (df['decision_date'] - df['argument_date']).dt.days
    
    return df

# loading model
print_progress("Loading the model...")
best_model = load('./model/best_mlp_model_1415feature.joblib')

# loading embdding
print_progress("Loading and preprocessing test data...")
df_test = pd.read_json("test.jsonl", lines=True)
df_test = preprocess_data(df_test)

# drop outcome variable
X_test = df_test.drop(['case_id'], axis=1)

# pred
print_progress("Making predictions...")
y_pred = best_model.predict(X_test)

# outcome DataFrame
results_df = pd.DataFrame({
    'case_id': df_test['case_id'],
    'prediction': y_pred
})
results_df['case_number'] = results_df['case_id'].str.extract('(\d+)').astype(int)
single_model_df = results_df.sort_values('case_number')
single_model_df = single_model_df.drop('case_number', axis=1)

# 9. Saving to CSV
print_progress("Saving results to CSV...")
single_model_df.to_csv('mlp_predictions_1415feature.csv', index=False)
