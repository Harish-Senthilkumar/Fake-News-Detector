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
print("\n📂 Loading prepared data...")
with open('prepared_data.pkl', 'rb') as file:
    X_train, X_test, y_train, y_test = pickle.load(file)


X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"✅ Loaded {len(X_train)} training samples")
print(f"✅ Loaded {len(X_test)} testing samples")


