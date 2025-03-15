from textblob import TextBlob
import pandas as pd

class SentimentAnalyzer:
    @staticmethod
    def get_sentiment_features(text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        return polarity, subjectivity

    def apply_sentiment_analysis(self, df, text_column):
        df[['polarity', 'subjectivity']] = df[text_column].apply(lambda x: pd.Series(self.get_sentiment_features(x)))
        return df
