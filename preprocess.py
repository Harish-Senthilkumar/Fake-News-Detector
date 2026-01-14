import re 
from matplotlib import text
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Downloading my NLTK data so I don't have to keep doing it every time
print("Downloading NLTK data...")

#Backup plan for code failure
try:
    nltk.download('stopwords', quiet=True)
    print("Done!")
except:
    print("Failed to download NLTK data. Please check your internet connection.")


#Making a clean display
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters and keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

#Removing stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

#Stemming/Lemmatization
def stem_text(text):
    stemmer = PorterStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

