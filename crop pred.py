import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r"B:\Crop_recommendation.csv")
data.head(5)                
crop_summary = pd.pivot_table(data, index=['label'], aggfunc='mean')

import plotly.express as px


for feature in ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall', 'ph']:
    fig = px.box(data, y=feature, points="all")

# Outlier removal
df_boston = data.copy()
columns_to_process = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

print(f"Original shape: {df_boston.shape}")

mask = pd.Series(True, index=df_boston.index)

for col in columns_to_process:
    print(f"\nProcessing: {col}")
    
    q1 = np.percentile(df_boston[col], 25, interpolation='midpoint')
    q3 = np.percentile(df_boston[col], 75, interpolation='midpoint')
    IQR = q3 - q1
    
    mask = mask & (df_boston[col] >= (q1 - 1.5 * IQR)) & (df_boston[col] <= (q3 + 1.5 * IQR))
    
    print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}")
    print(f"  Rows kept so far: {mask.sum()}")

df_boston_clean = df_boston[mask].copy()

print(f"\nFinal shape: {df_boston_clean.shape}")
print(f"Rows removed: {len(df_boston) - len(df_boston_clean)}")

data = df_boston_clean
x = data.drop('label', axis=1)
y = data['label']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True, random_state=0)


import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print(f"Training set: {x_train.shape}")
print(f"Testing set: {x_test.shape}")
print(f"Number of unique crops: {len(le.classes_)}")
print(f"Crop classes: {le.classes_}")



model = xgb.XGBClassifier(
    objective='multi:softprob',  
    num_class=len(le.classes_),  
    n_estimators=150,            
    max_depth=4,                
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,          
    gamma=0.1,                   
    reg_alpha=0.1,               # L1 regularization
    reg_lambda=1.0,              # L2 regularization
    eval_metric='mlogloss',      
    random_state=42
)



model.fit(x_train, y_train_encoded)


y_pred_proba = model.predict_proba(x_test)
print(f"\nPrediction probability shape: {y_pred_proba.shape}")
print(f"(samples, classes) = ({y_pred_proba.shape[0]}, {y_pred_proba.shape[1]})")




def get_top_k_predictions(probabilities, label_encoder, k=3):
    
    top_k_results = []
    
    for i, proba in enumerate(probabilities):
       
        top_k_indices = np.argsort(proba)[::-1][:k]
        
        
        top_k_probs = proba[top_k_indices]
        
        
        top_k_classes = label_encoder.inverse_transform(top_k_indices)
       
        result = {
            'sample_id': i,
            'predictions': [
                {
                    'rank': rank + 1,
                    'class': cls,
                    'probability': prob
                }
                for rank, (cls, prob) in enumerate(zip(top_k_classes, top_k_probs))
            ]
        }
        
        top_k_results.append(result)
    
    return top_k_results



top_3_predictions = get_top_k_predictions(y_pred_proba, le, k=3)


for i in range(min(10, len(top_3_predictions))):
    result = top_3_predictions[i]
    true_label = y_test.iloc[i]
    
    print(f"\nSample {i+1} | True Label: {true_label}")
    print("-" * 50)
    
    for pred in result['predictions']:
        rank_marker = "✓" if pred['class'] == true_label else " "
        print(f"  {rank_marker} Rank {pred['rank']}: {pred['class']:<15} | Probability: {pred['probability']:.4f} ({pred['probability']*100:.2f}%)")



y_pred_top1 = le.inverse_transform(np.argmax(y_pred_proba, axis=1))
top1_accuracy = accuracy_score(y_test, y_pred_top1)
print(f"\nTop-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")

top3_accuracy = top_k_accuracy_score(y_test_encoded, y_pred_proba, k=3, labels=range(len(le.classes_)))
print(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
print(classification_report(y_test, y_pred_top1))


def predict_top_k_crops(model, label_encoder, features, k=3):
    probabilities = model.predict_proba(features)
    
    results = []
    for i, proba in enumerate(probabilities):
        
        top_k_indices = np.argsort(proba)[::-1][:k]
        top_k_probs = proba[top_k_indices]
        top_k_classes = label_encoder.inverse_transform(top_k_indices)
        
        
        for rank, (cls, prob) in enumerate(zip(top_k_classes, top_k_probs), 1):
            results.append({
                'sample_id': i,
                'rank': rank,
                'predicted_crop': cls,
                'probability': prob,
                'confidence_pct': prob * 100
            })
    
    return pd.DataFrame(results)


sample_features = x_test.iloc[[0]]

print(sample_features.T)

predictions_df = predict_top_k_crops(model, le, sample_features, k=3)



y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
print(f"{'='*70}")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Overfitting Gap:   {(train_accuracy - test_accuracy):.4f} ({(train_accuracy - test_accuracy)*100:.2f}%)")
if train_accuracy - test_accuracy > 0.02:
    print(" some overfitting")
else:
    print(" good generalization (gap ≤ 2%)")
print(f"{'='*70}")
