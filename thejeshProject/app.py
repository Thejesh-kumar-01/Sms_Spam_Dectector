"""
Fraud / Scam Message Detector - Streamlit Web Application
A modern web app to detect fraudulent/scam messages using machine learning.
"""

import streamlit as st
import pickle
import re
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download the stopwords package
nltk.download('stopwords')

# If you are also doing tokenization, you might need 'punkt' as well:
nltk.download('punkt') 
nltk.download('punkt_tab') # Occasionally required for newer NLTK versions

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Page configuration
st.set_page_config(
    page_title="Fraud Message Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Clean and preprocess text data (same as training).
    Steps:
    1. Convert to lowercase
    2. Remove punctuation and special characters
    3. Remove stopwords
    4. Apply stemming
    """
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Step 3: Split into words and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Step 4: Apply stemming
    words = [stemmer.stem(word) for word in words]
    
    # Join words back into a single string
    processed_text = ' '.join(words)
    
    return processed_text

def load_model():
    """
    Load the trained model and vectorizer from pickle files using relative paths.
    """
    # Get the directory where app.py is located
    current_dir = os.path.dirname(__file__)
    
    # Create absolute paths to the files
    model_path = os.path.join(current_dir, 'model.pkl')
    vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"Model files not found at {current_dir}! Please ensure model.pkl and vectorizer.pkl are in the same folder as app.py.")
        st.stop()



# Custom CSS for modern glassmorphism UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main Container */
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Title Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.8s ease-in;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        animation: fadeIn 1s ease-in;
    }
    
    /* Input Area */
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        color: white;
        font-size: 1rem;
        padding: 1rem;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.5);
        outline: none;
        box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1);
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Result Container */
    .result-container {
        margin-top: 2rem;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    .result-legitimate {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(56, 142, 60, 0.2) 100%);
        border: 2px solid rgba(76, 175, 80, 0.5);
        color: #4caf50;
    }
    
    .result-fraud {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.2) 0%, rgba(211, 47, 47, 0.2) 100%);
        border: 2px solid rgba(244, 67, 54, 0.5);
        color: #f44336;
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: scaleIn 0.5s ease-out;
    }
    
    @keyframes scaleIn {
        from {
            transform: scale(0);
        }
        to {
            transform: scale(1);
        }
    }
    
    .result-text {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .result-description {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Info Box */
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeIn 1.2s ease-in;
    }
    
    .info-title {
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .info-text {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """
    Main function to run the Streamlit app.
    """
    # Load model and vectorizer
    model, vectorizer = load_model()
    
    # Main title and subtitle
    st.markdown('<h1 class="main-title">🛡️ Fraud Message Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Protect yourself from scams and fraudulent messages</p>', unsafe_allow_html=True)
    
    # Glassmorphism card container
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Text input area
    st.markdown("### Enter Message to Check")
    user_message = st.text_area(
        "",
        placeholder="Paste or type the message you want to check here...",
        height=150,
        label_visibility="collapsed"
    )
    
    # Check button
    check_button = st.button("🔍 Check Message", type="primary")
    
    # Process and display result
    if check_button:
        if not user_message or user_message.strip() == "":
            st.warning("⚠️ Please enter a message to check.")
        else:
            # Preprocess the input message
            processed_message = preprocess_text(user_message)
            
            # Convert to TF-IDF vector
            message_vector = vectorizer.transform([processed_message])
            
            # Make prediction
            prediction = model.predict(message_vector)[0]
            prediction_proba = model.predict_proba(message_vector)[0]
            
            # Display result
            if prediction == 0:  # Legitimate (ham)
                confidence = prediction_proba[0] * 100
                st.markdown(f"""
                <div class="result-container result-legitimate">
                    <div class="result-icon">✅</div>
                    <div class="result-text">Legitimate Message</div>
                    <div class="result-description">This message appears to be safe and legitimate.</div>
                    <div class="result-description" style="margin-top: 1rem; font-size: 0.9rem;">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:  # Fraud/Scam (spam)
                confidence = prediction_proba[1] * 100
                st.markdown(f"""
                <div class="result-container result-fraud">
                    <div class="result-icon">🚨</div>
                    <div class="result-text">Fraud / Scam Message</div>
                    <div class="result-description">Warning: This message appears to be fraudulent or a scam.</div>
                    <div class="result-description" style="margin-top: 1rem; font-size: 0.9rem;">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Close glass card
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <div class="info-title">ℹ️ How It Works</div>
        <div class="info-text">
            This detector uses Machine Learning (Multinomial Naive Bayes) trained on thousands of SMS messages 
            to identify fraudulent and scam messages. The model analyzes text patterns, keywords, and message 
            structure to determine if a message is legitimate or potentially fraudulent.
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Download NLTK data if needed
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    main()


