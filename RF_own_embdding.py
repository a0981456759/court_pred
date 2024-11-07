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

# 1. 数据加载
print_progress("Loading data...")
df = pd.read_json("../train.jsonl", lines=True)
print_progress(f"amount of features: {len(df)}")

# 加载 embedding 文件
print_progress("Loading embdding file...")
embeddings = np.load("../court_hearing_embeddings.npy")
print_progress(f"Shape: {embeddings.shape}")

# 2. 特征工程
print_progress("Feature engineering...")
df['argument_date'] = pd.to_datetime(df['argument_date'])
df['year'] = df['argument_date'].dt.year
df['month'] = df['argument_date'].dt.month
df['day'] = df['argument_date'].dt.day
df['dayofweek'] = df['argument_date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['case_duration'] = df['argument_date'].dt.year - df['year']

# 3. 特征定义
categorical_features = ['petitioner', 'respondent', 'petitioner_category', 'respondent_category', 'issue_area', 'petitioner_state', 'respondent_state']
numerical_features = ['year', 'court_hearing_length', 'utterances_number', 'case_duration', 'month', 'day', 'dayofweek', 'is_weekend']

# 4. 数据预处理
print_progress("Data Preprocessing...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# 5. 准备特征矩阵和目标变量
print_progress("Feature matrix...")
X_processed = preprocessor.fit_transform(df)
X = np.hstack([X_processed.toarray(), embeddings])  # 将处理后的特征和 embeddings 合并
y = df['successful_appeal'].values
print_progress(f"Amount of matrix: {X.shape[1]}")

# 6. 分割数据集
print_progress("Spliting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print_progress(f"Training set: {X_train.shape[0]}, Testing set: {X_test.shape[0]}")

# 7. 应用SMOTE来处理类别不平衡
print_progress("SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print_progress(f"Amount of new dataset: {X_train_resampled.shape[0]}")

# 8. 特征选择
print_progress("Feature selection...")
selector = SelectKBest(score_func=f_classif, k=200)
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector.transform(X_test)
print_progress(f"Amount of selection: {X_train_selected.shape[1]}")

# 9. 定义随机森林模型和参数网格
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# 10. 使用RandomizedSearchCV进行超参数调优
print_progress("tunning...")
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                   n_iter=20, cv=5, verbose=2, random_state=42, n_jobs=-1, scoring='f1')
random_search.fit(X_train_selected, y_train_resampled)
print_progress("finished")

# 11. 获取最佳模型
best_rf = random_search.best_estimator_
print_progress(f"params: {random_search.best_params_}")

# 12. 模型评估和可视化
print_progress("vis...")
y_pred = best_rf.predict(X_test_selected)
y_pred_proba = best_rf.predict_proba(X_test_selected)[:, 1]
print("Random Forest Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Re Text Embdding (Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('./graph/confusion_matrix_rf_with_ownembeddings.png')
plt.close()

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Re Text Embdding (Random Forest)')
plt.legend(loc="lower right")
plt.savefig('./graph/roc_curve_rf_with_ownembeddings.png')
plt.close()

# Precision-Recall曲线
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Re Text Embdding(Random Forest)')
plt.legend(loc="lower left")
plt.savefig('./graph/precision_recall_curve_rf_with_ownembeddings.png')
plt.close()

print_progress("saved")

# 11. 保存模型
print_progress("saving...")
dump(best_rf, './model/best_rf_model_with_ownembeddings_randomized.joblib')
dump(preprocessor, './model/preprocessor_rf_with_ownembeddings_randomized.joblib')
print_progress("saved")
