# first we should run the main.py python file to load the data and make preprocessing and train model then save it with results as follows:
## Data loaded successfully... , Data cleaned successfully... , Sentiment analysis applied successfully... , Model trained successfully... ,  Save the model
# second step we should run the test_scraper.py inside the web_scraber to take new news from remote url then apply TextCleaner and sentiment analyzer to add polarity and subjectivity also pos , neg , neu , compound  to memic the features that the model use and load the model then classify the news into: positive or negative then apply simple rank system that sort then
# for make a pipeline we can use task scheduler to create simple task that run for example daily at specific time to run the scraber that bring new news as i mentioned before and apply the logic and classify the extracet news then make ranking for them
