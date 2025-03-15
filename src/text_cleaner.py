import re

class TextCleaner:
    def clean_text(self, text):
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.lower()

    def apply_cleaning(self, df, column):
        df[column] = df[column].apply(self.clean_text)
        return df
