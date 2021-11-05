# Sentiment Analysis

Sentiment anaysis and prediction of ratings on Amazon customer review dataset for video games
In this project, we predicted product ratings based on the reviews provided by the users. Our work has mostly three aspects associated with it:
1. Gathering data and preprocessing: Check section 3
2. Running sentiment analysis on each reviews: In this step, we implement sentiment analysis model, that focuses on polarity (positive, negative or neutral) of reviews, thereby making it a classification problem. Pictorially, this step works as follows: See figure 1, one of the first steps in this section is feature-extraction. Generally speaking, either of bag-of-words, bag-of-ngrams or word vectors can be used in order to extract features. 
3. Generate ratings: Using predictive analysis on the output of step 2 (using Linear Regression or SVM), one can 
4. classify a given rating in either of the required classes (for example: 1-5 stars)
