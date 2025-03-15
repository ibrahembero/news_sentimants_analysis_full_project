# src/web_scraper/news_scraper.py
# src/web_scraper/news_scraper.py
import sys
import os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib  # To load the saved model
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Add the 'src' directory to the import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from text_cleaner import TextCleaner  # Now this should work


class NewsScraper:
    def __init__(self, num_articles=10):
        self.num_articles = num_articles
        self.text_cleaner = TextCleaner()  # Initialize the TextCleaner
        self.analyzer = SentimentIntensityAnalyzer()  # Initialize the sentiment analyzer
        # Dynamically get the absolute path to the model
        model_path = os.path.join(os.path.dirname(__file__), '../../models/random_forest_news_sentiment_model.pkl')

        # Now use this path to load the model
        self.model = joblib.load(model_path) # Load the saved model (update the path)

    def fetch_news(self, url="https://news.ycombinator.com/"):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Ensure the request was successful
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all news links
            news_items = soup.find_all('tr', class_='athing', limit=self.num_articles)

            news_data = []

            for item in news_items:
                title_tag = item.find('span', class_='titleline').find('a')  # Corrected to find title inside the titleline span
                link = title_tag['href'] if title_tag else ''
                title = title_tag.get_text(strip=True) if title_tag else 'No title'
                
                # Debug: Print the title to see if we are extracting it correctly
                print(f"Title: {title}")  # Check the title being extracted

                # Get the rank number (optional)
                rank = item.find('span', class_='rank').get_text(strip=True) if item.find('span', class_='rank') else 'No rank'

                # Clean the text (title)
                cleaned_title = self.text_cleaner.clean_text(title)
                
                # Perform sentiment analysis using VADER and TextBlob
                polarity, subjectivity = self.get_textblob_sentiment(cleaned_title)
                vader_scores = self.analyzer.polarity_scores(cleaned_title)
                
                sentiment_features = {
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'neg': vader_scores['neg'],
                    'neu': vader_scores['neu'],
                    'pos': vader_scores['pos'],
                    'compound': vader_scores['compound']
                }

                # Add the sentiment features to the data
                news_data.append({
                    'title': cleaned_title,
                    'link': link,
                    'rank': rank,
                    **sentiment_features
                })

            # Convert the list to a DataFrame for easy processing
            df = pd.DataFrame(news_data)

            # We only need the features for prediction
            X = df[['neg', 'neu', 'pos', 'compound','polarity', 'subjectivity']]

            # Use the saved model to classify sentiment (positive/negative)
            df['sentiment_class'] = self.predict_sentiment(X)

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error with the request: {e}")
        except Exception as e:
            print(f"Error during scraping: {e}")
        
        return None

    def get_textblob_sentiment(self, text):
        """Calculate polarity and subjectivity using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        return polarity, subjectivity

    def predict_sentiment(self, X):
        """Predict sentiment (positive/negative) using the saved model"""
        predictions = self.model.predict(X)
        return predictions
