import pandas as pd

print("Checking if news_data.csv is ready...\n")

df = pd.read_csv('news_data.csv')

print("File loaded successfully!")
print(f"Total articles: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

print(f"\n Label distribution:")
print(df['label'].value_counts())

print(f"\n Example article:")
print(f"Text: {df['text'].iloc[0][:200]}...")
print(f"Label: {df['label'].iloc[0]}")

print("\n Everything looks good! Ready to continue to Step 5.")