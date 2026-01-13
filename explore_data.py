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

#Are there any missing values?
print(f"\n❓ Missing Values:")
print(df.isnull().sum())

# Let's see article lengths
df['text_length'] = df['text'].apply(len)
print(f"\n📏 Article length statistics:")
print(f"Average length: {df['text_length'].mean():.0f} characters")
print(f"Shortest article: {df['text_length'].min()} characters")
print(f"Longest article: {df['text_length'].max()} characters")