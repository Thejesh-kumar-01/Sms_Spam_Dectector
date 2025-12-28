"""
Fraud / Scam Message Detector - Model Training Script
This script downloads the SMS Spam Collection dataset, preprocesses the text,
trains a Multinomial Naive Bayes classifier, and saves the model.
"""

import pandas as pd
import numpy as np
import pickle
import os
import requests
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data if not already present
print("Checking NLTK data...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def download_dataset():
    """
    Download the SMS Spam Collection dataset if it doesn't exist.
    Uses the UCI Machine Learning Repository dataset.
    """
    if os.path.exists('spam.csv'):
        print("Dataset file 'spam.csv' already exists. Skipping download.")
        return
    
    print("Downloading SMS Spam Collection dataset...")
    # UCI ML Repository SMS Spam Collection dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    try:
        # Download the zip file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save zip file temporarily
        with open('smsspamcollection.zip', 'wb') as f:
            f.write(response.content)
        
        # Extract the CSV file from zip
        import zipfile
        with zipfile.ZipFile('smsspamcollection.zip', 'r') as zip_ref:
            # Extract SMSSpamCollection file
            zip_ref.extractall('.')
        
        # The extracted file might be named differently, let's check
        if os.path.exists('SMSSpamCollection'):
            # Read the TSV file and convert to CSV
            df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
            df.to_csv('spam.csv', index=False)
            os.remove('SMSSpamCollection')
        
        # Clean up zip file
        if os.path.exists('smsspamcollection.zip'):
            os.remove('smsspamcollection.zip')
        
        print("Dataset downloaded and saved as 'spam.csv'")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Please download the dataset manually:")
        print("1. Visit: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
        print("2. Download the dataset")
        print("3. Save it as 'spam.csv' with columns: label, message")
        raise


def preprocess_text(text):
    """
    Clean and preprocess text data.
    Steps:
    1. Convert to lowercase
    2. Remove punctuation and special characters
    3. Remove stopwords
    4. Apply stemming
    """
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Step 3: Split into words and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Step 4: Apply stemming
    words = [stemmer.stem(word) for word in words]
    
    # Join words back into a single string
    processed_text = ' '.join(words)
    
    return processed_text


def main():
    """
    Main function to train the spam detection model.
    """
    print("=" * 60)
    print("Fraud / Scam Message Detector - Model Training")
    print("=" * 60)
    
    # Step 1: Download dataset if needed
    download_dataset()
    
    # Step 2: Load the dataset
    print("\nLoading dataset...")
    df = pd.read_csv('spam.csv')
    print(f"Dataset loaded: {len(df)} messages")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle different column name formats
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    
    # Display class distribution
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    
    # Step 3: Preprocess the text data
    print("\nPreprocessing text data...")
    print("This may take a few moments...")
    df['processed_message'] = df['message'].apply(preprocess_text)
    print("Text preprocessing completed!")
    
    # Step 4: Prepare features and labels
    X = df['processed_message']  # Features (processed text)
    y = df['label']  # Labels (spam/ham)
    
    # Convert labels to binary (spam = 1, ham = 0)
    y = y.map({'spam': 1, 'ham': 0})
    
    # Step 5: Split data into training and testing sets
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Step 6: Convert text to numbers using TF-IDF
    print("\nConverting text to numbers using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("TF-IDF vectorization completed!")
    
    # Step 7: Train the Multinomial Naive Bayes classifier
    print("\nTraining Multinomial Naive Bayes classifier...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    print("Model training completed!")
    
    # Step 8: Evaluate the model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 60}")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"{'=' * 60}")
    
    # Step 9: Save the model and vectorizer
    print("\nSaving model and vectorizer...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Model saved as 'model.pkl'")
    print("Vectorizer saved as 'vectorizer.pkl'")
    print("\nTraining completed successfully!")
    print("\nYou can now run 'streamlit run app.py' to use the web application.")


if __name__ == "__main__":
    main()

