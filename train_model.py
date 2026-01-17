import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("Training Fake News Detection Model")
print("=" * 60)

#Loading the prepared data 
print("\nLoading prepared data...")
with open('prepared_data.pkl', 'rb') as file:
    X_train, X_test, y_train, y_test = pickle.load(file)


X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"Loaded {len(X_train)} training samples")
print(f"Loaded {len(X_test)} testing samples")

'''
Converting text data to numerical features

TF-IDF
'''
print("\n" + "=" * 60)
print("STEP 1: CONVERTING TEXT TO NUMBERS")
print("=" * 60)
print("\nUsing TF-IDF (Term Frequency-Inverse Document Frequency)")
print("This converts text into numerical features that the model can understand")

vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\nCreated {X_train_vec.shape[1]} features from the text")

