import pandas as pd
import numpy as np
from joblib import load
import json

# loading model and preprocessor
model = load('./model/improved_mlp_model_with_ownembedding_kbest.joblib')
preprocessor = load('./model/improved_preprocessor_mlp_with_ownembedding_kbest.joblib')

# loading data
df_test = pd.read_json("test.jsonl", lines=True)

#  embeddings
test_embeddings = np.load("court_hearing_embeddings.npy")

# features engineering
df_test['argument_date'] = pd.to_datetime(df_test['argument_date'])
df_test['year'] = df_test['argument_date'].dt.year
df_test['month'] = df_test['argument_date'].dt.month
df_test['day'] = df_test['argument_date'].dt.day
df_test['dayofweek'] = df_test['argument_date'].dt.dayofweek
df_test['is_weekend'] = df_test['dayofweek'].isin([5, 6]).astype(int)
df_test['case_duration'] = df_test['argument_date'].dt.year - df_test['year']

# preprocess
X_test_processed = preprocessor.transform(df_test)

print("Shape of X_test_processed:", X_test_processed.shape)
print("Shape of test_embeddings:", test_embeddings.shape)

# make sure two arrays have same rows
min_rows = min(X_test_processed.shape[0], test_embeddings.shape[0])
X_test_processed = X_test_processed[:min_rows]
test_embeddings = test_embeddings[:min_rows]

# stacking
X_test = np.hstack([X_test_processed.toarray(), test_embeddings])


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

df_test['predicted_successful_appeal'] = y_pred
df_test['predicted_probability'] = y_pred_proba

# csv pre and order
results_df = pd.DataFrame({
    'case_id': df_test['case_id'],
    'prediction': y_pred
})
results_df['case_number'] = results_df['case_id'].str.extract('(\d+)').astype(int)
single_model_df = results_df.sort_values('case_number')
single_model_df = single_model_df.drop('case_number', axis=1)

single_model_df.to_csv('MLP_own_embedding.csv', index=False)


