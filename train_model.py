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


'''
Training the Logistic Regression model
'''
print("\n" + "=" * 60)
print("STEP 2: TRAINING THE LOGISTIC REGRESSION MODEL")
print("=" * 60)

model = LogisticRegression(max_iter=1000, random_state - 42)
model.fit(X_train_vec, y_train)
print("\nModel training complete!")

#Evaluating the model
print("\n" + "=" * 60)
print("STEP 3: EVALUATING THE MODEL")
print("=" * 60)

print("\n Testing on unseen articles and text...")
y_pred = model.predict(X_test_vec)

#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
#All Detailed rest of classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

#Confusion Matrix
print("\nConfusion Matrix - Creating Visualization: ")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues',
            xticklabels=['FAKE', 'REAL'],
            yticklabels=['FAKE', 'REAL']
            cbar_kws={'label': 'Count'})
plt.titleImportance('Confusion Matrix\n', fontsize=16, fontweight='bold')
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)

plt.text(1, -0.3, f'Overall Accuracy: {accuracy * 100:.2f}%', 
         ha='center', fontsize=12, fontweight='bold')

#Dynamic and compact layout
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

#Show some example predictions
print("\nSome Example Predictions:")
for i in range(5):
    actual = "Fake" if y_test.iloc[i] == 0 else "Real"
    predicted = "Fake" if y_pred[i] == 0 else "Real"
    correct = "Correct" if actual == predicted else "Incorrect"
    print(f"\n{correct} Article {i+1}:")
    print(f"   Text: {X_test.iloc[i][:100]}...")
    print(f"   Actual: {actual} | Predicted: {predicted}")

#Save the trained model and vectorizer
print("\n" + "=" * 60)
print("STEP 4: SAVING THE MODEL AND VECTORIZER")
print("=" * 60)

with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved as 'fake_news_model.pkl'")
print("Vectorizer saved as 'tfidf_vectorizer.pkl'")

#Accuracy summary and last statements
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")
print("\nNext step: Run 'python predict.py' to test predictions!")