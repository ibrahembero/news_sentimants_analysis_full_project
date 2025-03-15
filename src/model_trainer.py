import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        with open('results.txt', 'w') as file:
            file.write(f"Accuracy: {accuracy}\n")
            file.write(f"Classification Report:\n{report}\n")
        print(f"Results logged in results.txt")
        return accuracy, report

    def save_model(self, filename='models/random_forest_news_sentiment_model.pkl'):
        # Ensure the directory exists
        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
        joblib.dump(self.model, filename)
        print(f"Model saved as {filename}")

