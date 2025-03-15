import sys
import os

# Add the 'src' directory to the import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# Verify the PYTHONPATH
print("PYTHONPATH:", sys.path)

from data_loader import DataLoader
from text_cleaner import TextCleaner
from sentiment_analyzer import SentimentAnalyzer
from model_trainer import ModelTrainer

# Load the data
data_loader = DataLoader('raw_data/news.csv')
df = data_loader.load_data()
if df is not None:
    print('Data loaded successfully...')
else:
    raise ValueError('There is an error with loading data...')

# Clean the text
text_cleaner = TextCleaner()
df_cleaned = text_cleaner.apply_cleaning(df, 'news')
if df_cleaned is not None:
    print('Data cleaned successfully...')
else:
    raise ValueError('There is an error with cleaning data...')

# Apply sentiment analysis
sentiment_analyzer = SentimentAnalyzer()
df_with_sentiment = sentiment_analyzer.apply_sentiment_analysis(df_cleaned, 'news')
if df_with_sentiment is not None:
    print('Sentiment analysis applied successfully...')
else:
    raise ValueError('There is an error with sentiment analysis...')

# Drop unnecessary columns
df_with_sentiment = df_with_sentiment.drop(columns=['news', 'date'])

# Prepare features and labels
X = df_with_sentiment.drop(columns='sentiment')
y = df_with_sentiment['sentiment']

# Train the model
model_trainer = ModelTrainer()
accuracy, report = model_trainer.train_model(X, y)
if accuracy is not None and report is not None:
    print('Model trained successfully...')
else:
    raise ValueError('There is an error with model training...')

# Save the model
model_trainer.save_model()

# to run the program: python src/main.py
