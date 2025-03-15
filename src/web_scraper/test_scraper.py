import pandas as pd
from news_scraper import NewsScraper

def test_scraper():
    # Initialize the NewsScraper with the number of articles you want to scrape
    scraper = NewsScraper(num_articles=10)
    
    # Scrape news from the website
    news_df = scraper.fetch_news(url="https://news.ycombinator.com/")  # Example URL
    
    if news_df is not None:
        # Define a ranking system for sentiment
        sentiment_ranking = {'POSITIVE': 1, 'NEGATIVE': 2}
        
        # Add a column for sentiment rank
        news_df['sentiment_rank'] = news_df['sentiment_class'].map(sentiment_ranking)
        
        # Add a column for article length (you can modify this to other criteria)
        news_df['title_length'] = news_df['title'].apply(len)  # Length of the article title
        
        # Rank by sentiment first (POSITIVE -> NEGATIVE), then by title length (longest title is best)
        news_df = news_df.sort_values(by=['sentiment_rank', 'title_length'], ascending=[True, False])
        
        print("Scraped News:")
        print(news_df[['title', 'sentiment_class', 'sentiment_rank', 'title_length']])  # Display the titles, sentiment, rank, and length
    else:
        print("Error scraping news.")

# Run the test
if __name__ == "__main__":
    test_scraper()
