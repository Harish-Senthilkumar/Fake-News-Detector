import pandas as pd
from preprocess import preprocess_text
from sklearn.model_selection import train_test_split
import pickle 

print("=" * 60)
print("Preparing Dataset for training and testing")
print("=" * 60)

#: Load the dataset
print("\n📂 Loading data...")
df = pd.read_csv('news_data.csv')

#Remove any rows with missing text
df = df.dropna(subset=['text', 'label'])
print(f"Data loaded. Total records: {len(df)}")

