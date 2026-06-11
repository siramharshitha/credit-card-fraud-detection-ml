import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
 
# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
n_samples = 10000
n_fraud = 200  # ~2% fraud rate
 
# Legitimate transactions
legit = pd.DataFrame({
    'Amount':     np.random.exponential(scale=80, size=n_samples - n_fraud),
    'Time':       np.random.uniform(0, 172800, n_samples - n_fraud),
    'V1':         np.random.normal(0, 1, n_samples - n_fraud),
    'V2':         np.random.normal(0, 1, n_samples - n_fraud),
    'V3':         np.random.normal(0, 1, n_samples - n_fraud),
    'V4':         np.random.normal(0, 1, n_samples - n_fraud),
    'Class':      0
})
 
# Fraudulent transactions (higher amount, different distribution)
fraud = pd.DataFrame({
    'Amount':     np.random.exponential(scale=200, size=n_fraud),
    'Time':       np.random.uniform(0, 172800, n_fraud),
    'V1':         np.random.normal(-3, 2, n_fraud),
    'V2':         np.random.normal(2, 2, n_fraud),
    'V3':         np.random.normal(-2, 2, n_fraud),
    'V4':         np.random.normal(3, 2, n_fraud),
    'Class':      1
})
 
df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
 
print('Dataset shape:', df.shape)
print('Class distribution:')
print(df['Class'].value_counts())
print(f'Fraud percentage: {df["Class"].mean()*100:.2f}%')
 
# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
# Class distribution
df['Class'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue','tomato'])
axes[0].set_title('Class Distribution (0=Legit, 1=Fraud)')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)
 
# Amount distribution by class
df[df['Class']==0]['Amount'].hist(bins=50, ax=axes[1], alpha=0.6, label='Legit', color='steelblue')
df[df['Class']==1]['Amount'].hist(bins=50, ax=axes[1], alpha=0.6, label='Fraud', color='tomato')
axes[1].set_title('Transaction Amount Distribution')
axes[1].set_xlabel('Amount ($)')
axes[1].legend()
 
plt.tight_layout()
plt.savefig('eda_plots.png', dpi=120)
plt.show()
print('EDA plots saved as eda_plots.png')
 
# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
features = ['Amount', 'Time', 'V1', 'V2', 'V3', 'V4']
X = df[features]
y = df['Class']
 
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
 
# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f'After SMOTE - Train size: {X_train_res.shape[0]}, Fraud samples: {y_train_res.sum()}')
 
# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}
 
results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    results[name] = {'model': model, 'y_pred': y_pred, 'y_proba': y_proba, 'auc': auc}
    print(f'\n=== {name} ===')
    print(f'ROC-AUC Score: {auc:.4f}')
    print(classification_report(y_test, y_pred, target_names=['Legit','Fraud']))
 
# ─────────────────────────────────────────────
# 5. VISUALISE RESULTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Confusion matrix for best model (Random Forest)
best_name = max(results, key=lambda n: results[n]['auc'])
best      = results[best_name]
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'])
axes[0].set_title(f'Confusion Matrix – {best_name}')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
 
# ROC curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
axes[1].plot([0,1],[0,1],'k--', label='Random')
axes[1].set_title('ROC Curves')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()
 
plt.tight_layout()
plt.savefig('model_results.png', dpi=120)
plt.show()
print('Model result plots saved as model_results.png')
 
# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
 
plt.figure(figsize=(8, 4))
importances.plot(kind='bar', color='steelblue')
plt.title('Feature Importances – Random Forest')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=120)
plt.show()
print('Feature importance plot saved as feature_importance.png')
 
# ─────────────────────────────────────────────
# 7. PREDICT ON NEW TRANSACTION
# ─────────────────────────────────────────────
new_transaction = pd.DataFrame([{
    'Amount': 450.0, 'Time': 50000,
    'V1': -2.5, 'V2': 1.8, 'V3': -1.9, 'V4': 2.7
}])
 
new_scaled = scaler.transform(new_transaction)
prob = rf_model.predict_proba(new_scaled)[0][1]
label = 'FRAUD' if prob > 0.5 else 'LEGITIMATE'
print(f'\nNew transaction prediction: {label} (fraud probability: {prob:.2%})')