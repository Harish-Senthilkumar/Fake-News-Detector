# Fake News Detector

A machine learning project that classifies news articles as FAKE or REAL using Natural Language Processing.

## Project Goal

Detect fake news by analyzing text patterns and linguistic features using machine learning.

## Tech Stack

- Python 3.x
- pandas - Data manipulation
- scikit-learn - Machine learning
- NLTK - Natural language processing
- matplotlib & seaborn - Visualization

## Project Structure
```
fake-news-detector/
├── explore_data.py       # Data exploration
├── preprocess.py         # Text preprocessing
├── prepare_dataset.py    # Dataset preparation
├── train_model.py        # Model training
├── predict.py            # Predictions
└── README.md
```

## Installation
```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/fake-news-detector.git
cd fake-news-detector

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"
```

## Dataset

Uses 40,000+ news articles labeled as FAKE or REAL.

*Dataset not included due to size.*

## Learning Goals

- Text preprocessing techniques
- TF-IDF feature extraction
- Binary classification
- Model evaluation
- NLP applications

## Progress

- [x] Project setup
- [x] Data exploration
- [x] Text preprocessing
- [ ] Model training
- [ ] Predictions
- [ ] Deployment

## Author
Harish Senthilkumar