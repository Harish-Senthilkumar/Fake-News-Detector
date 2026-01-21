import pickle
from preprocess import preprocess_text

print("=" * 60)
print("FAKE NEWS DETECTOR - PREDICTION MODE")
print("=" * 60)

# Load the trained model
print("\nLoading model...")
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model loaded!\n")

def predict_news(article):
    """Predicts if a news article is FAKE or REAL"""
    cleaned = preprocess_text(article)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    label = "FAKE" if prediction == 0 else "REAL"
    confidence = probability[prediction] * 100
    
    return label, confidence

# Test examples
test_articles = [
    """
    Scientists discovered eating chocolate daily makes you live forever!
    Researchers claim this miracle finding. Big pharma doesn't want you to know!
    """,
    
    """
    The Federal Reserve announced interest rates will remain unchanged
    following their monthly meeting based on economic indicators.
    """,
    
    """
    BREAKING: Aliens landed in New York! Government cover-up!
    Share before they delete it!
    """,
]

print("=" * 60)
print("TESTING THE DETECTOR")
print("=" * 60)

for i, article in enumerate(test_articles, 1):
    print(f"\nArticle {i}:")
    print(article.strip()[:150] + "...")
    
    label, confidence = predict_news(article)
    
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 60)

# Interactive mode
print("\n" + "=" * 60)
print("TRY YOUR OWN ARTICLE!")
print("=" * 60)
print("\nPaste an article and press Enter twice:\n")

try:
    user_article = []
    empty_count = 0
    
    while True:
        line = input()
        if line == "":
            empty_count += 1
            if empty_count >= 2 or (user_article and empty_count >= 1):
                break
        else:
            empty_count = 0
            user_article.append(line)
    
    if user_article:
        full_article = "\n".join(user_article)
        label, confidence = predict_news(full_article)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
        
except KeyboardInterrupt:
    print("\n\nGoodbye!")