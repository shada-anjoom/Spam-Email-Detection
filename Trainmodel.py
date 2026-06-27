import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from data_preprocessing import preprocess_pipeline

def train():
    print("Starting preprocessing...")
    df = preprocess_pipeline(None)
    
    X = df['clean_message']
    y = df['label']  # 'spam' or 'not spam'
    
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Saving model and vectorizer...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/spam_classifier.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    print("Training complete! Model successfully saved in 'models/' directory.")

if __name__ == "__main__":
    train()