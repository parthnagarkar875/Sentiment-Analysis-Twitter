# Sentiment-Analysis-Twitter

##Will only work well for tweets posted in English language.

Sentiment analysis is the process of computationally identifying and categorizing opinions 
expressed in a piece of text, especially in order to determine whether the writer's attitude towards a
particular topic, product, etc. is positive, negative, or neutral.


In this project I have used the training data which consists of 25000 IMDB movie reviews and their respective sentiments
labelled as either 0 or 1. 0 is for negative and 1 is for positive. 

The Multinomial Naive Bayes classifier has been used for training the data and predicting the output(here sentiment) of the
tweet.


The Tweets have been pulled with the help of Tweepy module. The tweepy module has its own methods for OAuthHandling, 
setting the access tokens and establishing connection by logging in to the twitter developer account. 

The user will be asked for the topic on which the tweets are to be pulled. Based on the live tweets that are pulled, the 
predictions will be made during the runtime. 

Once the tweets have been pulled, they will be written to the file 'Scores.txt' with their respective sentiments. 
0 for negative and 1 for positive.
