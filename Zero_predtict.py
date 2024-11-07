import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# 1. Loading training data
print_progress("Loading training data...")
df_train = pd.read_json("train.jsonl", lines=True)

# 2. Feature engineering for training data
print_progress("Performing feature engineering on training data...")
df_train['argument_date'] = pd.to_datetime(df_train['argument_date'])
df_train['year'] = df_train['argument_date'].dt.year
df_train['month'] = df_train['argument_date'].dt.month
df_train['day'] = df_train['argument_date'].dt.day
df_train['dayofweek'] = df_train['argument_date'].dt.dayofweek
df_train['is_weekend'] = df_train['dayofweek'].isin([5, 6]).astype(int)
df_train['case_duration'] = df_train['argument_date'].dt.year - df_train['year']

# 3. Feature definition
numerical_features = ['year', 'court_hearing_length', 'utterances_number', 'case_duration', 'month', 'day', 'dayofweek', 'is_weekend']
categorical_features = ['petitioner', 'respondent', 'petitioner_category', 'respondent_category', 'issue_area', 'petitioner_state', 'respondent_state']

# Prepare training data
X_train = pd.get_dummies(df_train[numerical_features + categorical_features], columns=categorical_features)
y_train = df_train['successful_appeal'].values

# 4. Training
print_progress("Training the model...")
majority_predictor = MajorityClassPredictor()
majority_predictor.fit(y_train)

# 5. Loading and preparing test data
print_progress("Loading and preparing test data...")
df_test = pd.read_json("test.jsonl", lines=True)

# Perform the same feature engineering on test data
df_test['argument_date'] = pd.to_datetime(df_test['argument_date'])
df_test['year'] = df_test['argument_date'].dt.year
df_test['month'] = df_test['argument_date'].dt.month
df_test['day'] = df_test['argument_date'].dt.day
df_test['dayofweek'] = df_test['argument_date'].dt.dayofweek
df_test['is_weekend'] = df_test['dayofweek'].isin([5, 6]).astype(int)
df_test['case_duration'] = df_test['argument_date'].dt.year - df_test['year']

# Prepare test features
X_test = pd.get_dummies(df_test[numerical_features + categorical_features], columns=categorical_features)

# Ensure X_test has the same columns as X_train
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X_train.columns]

# 6. Predicting
print_progress("Making predictions...")
y_pred = majority_predictor.predict(X_test)

# 7. Creating results DataFrame
results_df = pd.DataFrame({
    'case_id': df_test['case_id'],
        'prediction': y_pred
})
results_df['case_number'] = results_df['case_id'].str.extract('(\d+)').astype(int)
single_model_df = results_df.sort_values('case_number')
single_model_df = single_model_df.drop('case_number', axis=1)

# 9. Saving to CSV
print_progress("Saving results to CSV...")
single_model_df.to_csv('zero_r_predictions.csv', index=False)

