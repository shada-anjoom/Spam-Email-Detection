import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import preprocess_pipeline

def evaluate():
    MODEL_PATH = 'models/spam_classifier.pkl'
    VEC_PATH = 'models/tfidf_vectorizer.pkl'
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        print("Model or vectorizer not found. Please run train_model.py first.")
        sys.exit(1)
        
    print("Loading model and vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    
    print("Preparing testing data...")
    df = preprocess_pipeline(None)
    X = df['clean_message']
    y = df['label'] # 'spam' or 'not spam'
    
    # Map 'spam' to 1, 'not spam' to 0 for ROC curve
    y_binary = y.map({'spam': 1, 'not spam': 0})
    
    _, X_test, _, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    print("Vectorizing test data...")
    X_test_vec = vectorizer.transform(X_test)
    
    print("Evaluating...")
    y_pred = model.predict(X_test_vec)
    
    # For ROC, model.predict_proba must be used and we need the probability of the positive class
    # The model was trained on string labels 'spam'/'not spam', so its classes_ attribute is typically ['not spam', 'spam']
    pos_class_idx = list(model.classes_).index('spam')
    y_pred_proba = model.predict_proba(X_test_vec)[:, pos_class_idx]
    
    # To match y_test which is 0/1, we map the text predictions to 0/1
    y_pred_binary = [1 if p == 'spam' else 0 for p in y_pred]
    
    cm = confusion_matrix(y_test, y_pred_binary)
    acc = accuracy_score(y_test, y_pred_binary)
    
    print("="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_binary, target_names=['not spam', 'spam']))
    print("Confusion Matrix:")
    print(f"                 Predicted Not Spam     Predicted Spam")
    print(f"Actual Not Spam  {cm[0][0]:<28} {cm[0][1]}")
    print(f"Actual Spam      {cm[1][0]:<28} {cm[1][1]}")
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print("\nROC Curve Performance:")
    print(f"AUC (Area Under the Curve): {roc_auc:.4f}")
    
    # Plot and save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['not spam', 'spam'], yticklabels=['not spam', 'spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot and save ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    print("\nSaved performance plots: 'confusion_matrix.png' and 'roc_curve.png'")

if __name__ == "__main__":
    evaluate()