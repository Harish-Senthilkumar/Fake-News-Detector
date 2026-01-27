import pandas as pd
from preprocess import preprocess_text
from sklearn.model_selection import train_test_split
import pickle

print("=" * 60)
print("PREPARING DATASET FOR TRAINING")
print("=" * 60)

# Load data
print("\nLoading data...")
df = pd.read_csv('news_data.csv')

# Remove any rows with missing text
df = df.dropna(subset=['text', 'label'])
print(f"Loaded {len(df)} articles")

# For learning purposes, we'll use a sample (you can increase this later)
# Using 10,000 articles for faster training
print("\nSampling 10,000 articles for training...")
df = df.sample(n=min(10000, len(df)), random_state=42)
print(f"Using {len(df)} articles")

# Preprocess all text
print("\nPreprocessing text (this may take 1-2 minutes)...")
print("Cleaning, removing stopwords, and stemming...")

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Show before/after example
print("\nExample transformation:")
print("\nBEFORE:")
print(df['text'].iloc[0][:200] + "...")
print("\nAFTER:")
print(df['cleaned_text'].iloc[0][:200] + "...")

# Prepare features (X) and labels (y)
X = df['cleaned_text']  # The cleaned articles
y = df['label']          # FAKE or REAL

# Convert labels to numbers: FAKE=0, REAL=1
label_mapping = {'FAKE': 0, 'REAL': 1}
y = y.map(label_mapping)

print(f"\nLabel distribution:")
print(f"FAKE (0): {(y == 0).sum()}")
print(f"REAL (1): {(y == 1).sum()}")

# Split into training (80%) and testing (20%) sets
print("\nSplitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} articles")
print(f"Testing set: {len(X_test)} articles")

# Save the prepared data
print("\nSaving prepared data...")
data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
}

with open('prepared_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Saved to 'prepared_data.pkl'")
print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE!")
print("=" * 60)
print("\nNext step: Run 'python train_model.py' to train the model")

