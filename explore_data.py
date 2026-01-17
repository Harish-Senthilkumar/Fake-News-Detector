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
print("\nLabel Distribution:")
print(df['label'].value_counts())

#Are there any missing values?
print(f"\nMissing Values:")
print(df.isnull().sum())

# Let's see article lengths
df['text_length'] = df['text'].apply(len)
print(f"\nArticle length statistics:")
print(f"Average length: {df['text_length'].mean():.0f} characters")
print(f"Shortest article: {df['text_length'].min()} characters")
print(f"Longest article: {df['text_length'].max()} characters")

#try to identify and compare the lengths of fake vs real news articles
print(f"\nAverage article lengths by label:")
print(df.groupby('label')['text_length'].mean())

#Visualize the data through graphs, figures, etc.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#Label distr = plot 1
df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
axes[0].set_title('Distribution of Fake vs Real News', fontsize = 15, fontweight='bold')
axes[0].set_xlabel('Label', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=0)

#Label distr = plot 2
df.boxplot(column='text_length', by='label', ax=axes[1])
axes[1].set_title('Article Lengths by Label', fontsize = 15, fontweight='bold')
axes[1].set_xlabel('Label', fontsize=12)
axes[1].set_ylabel('Text Length (characters)', fontsize=12)

# To remove the default title
plt.suptitle('')
#Other layout adjustments
plt.tight_layout()
plt.savefig('data_exploration_plots.png', dpi=300, bbox_inches='tight')
#show the actual graph
plt.show()

print("\n" + "=" * 50)
print("DATA EXPLORATION COMPLETE")
print("=" * 50)