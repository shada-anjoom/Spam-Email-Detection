import pandas as pd
import string
import pandas as pd
import string

# Minimal custom stopwords list since nltk is causing installation issues on py3.14
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", 
    "once", "here", "there", "when", "where", "why", "how", "all", "any", 
    "both", "each", "few", "more", "most", "other", "some", "such", "no", 
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
    "t", "can", "will", "just", "don", "should", "now"
])

def load_data(filepath=None):
    import kagglehub
    import os
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("jackksoncsie/spam-email-dataset")
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    filepath = os.path.join(path, csv_file)
    print("Dataset path:", filepath)
    
    df = pd.read_csv(filepath)
    df = df[['spam', 'text']]
    df.columns = ['label', 'message']
    
    df['label'] = df['label'].replace({1: 'spam', 0: 'not spam'})
    return df

def clean_text(text):
    """Cleans text: lowercase, remove punctuation, remove stopwords, and stems."""
    text = str(text).lower()
    
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    
    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]
    
    # Skipping stemming since nltk is removed
    
    return " ".join(filtered_words)

def preprocess_pipeline(filepath):
    print("Loading data...")
    df = load_data(filepath)
    print(f"Data loaded: {len(df)} records.")
    
    # Clean missing values if any
    df.dropna(inplace=True)
    
    print("Cleaning text (this may take a moment)...")
    df['clean_message'] = df['message'].apply(clean_text)
    
    return df

if __name__ == "__main__":
    import sys
    import os
    
    filepath = "data/spam.csv"
    if not os.path.exists(filepath):
        filepath = "data/SMSSpamCollection"
    
    if not os.path.exists(filepath):
        print("Data file not found. Ensure download_data.py has been run.")
        sys.exit(1)
        
    df = preprocess_pipeline(filepath)
    print(df.head())
    print("Preprocessing successfully tested.")