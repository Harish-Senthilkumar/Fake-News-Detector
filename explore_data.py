import pandas as pd
import matplotlib.pyplot as plt

#Load my data set
df = pd.read_csv('news_data.csv')

print("=" * 50)
print("EXPLORING OUR DATASET")
print("=" * 50)

#How many articles are there?
print(f"Total number of articles: {len(df)}")

#What are the columns in the dataset?
print(f"Columns in the dataset: {df.columns.tolist()}")

#First article:
print("First article:")
print(f"Text: {df['text'].iloc[0][:300]}...")  # Print first 300 characters
print(f"Label: {df['label'].iloc[0]}")

#Distribution of Fake vs Real News...
print("\n🏷️ Label Distribution:")
print(df['label'].value_counts())

