'''
-------------------------------------------------------------
About Dataset:
The IMDB movie reviews dataset has been used for this project.
The dataset contains 25000 reviews that are preprocessed and it also contains the sentiment score for each review out of 10.
Higher the sentiment value, the more is the positivity present in the statement.
-------------------------------------------------------------
About Training algorithm:
The naive bayes Multinomial classifier is used for training the data.
-------------------------------------------------------------
About Pulling Tweets:
The Tweepy module has been used for pulling tweets which has a built in OAuthHandler.
It also has multiple methods to establish connection to twitter from our account and to pull tweets
-------------------------------------------------------------
About Plotting pie chart:
MatplotLib has been used for plotting the pie-chart.
-------------------------------------------------------------
About storing results:
Scores.txt file has been used to store the results in a TSV(Tab Separated Value) format
-------------------------------------------------------------
'''

import matplotlib.pyplot as plt
import tweepy
from sklearn.metrics import accuracy_score
import numpy as np
from time import sleep
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from nltk.corpus import stopwords

print("Reading the DataSet...")
sleep(3)
df=pd.read_csv("labeledTrainData.tsv",sep="\t")



c=df.columns.tolist()
st=set(stopwords.words('english'))
print("Calculating the frequency of tokens in stopwords...")
sleep(4)
ve=TfidfVectorizer(use_idf=True,lowercase=True,strip_accents='ascii',stop_words=st)



print("Generating count vectors. This may take some time...")
X=ve.fit_transform(df[c[2]])
y=df[c[1]]



print("Training the model...")
sleep(5)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
classi=naive_bayes.MultinomialNB()
classi.fit(X_train,y_train)
print("Accuracy score is:",accuracy_score(y_test,classi.predict(X_test)))
sleep(4)
print("Model trained. Ready to predict!")

'''
In the variables mentioned below, you must add your respective keys and tokens that you'll get from 
your twitter developer account
'''
consumer_key=''
consumer_secret=''
access_token=''
access_token_secret=''
print("Connecting to twitter...")
sleep(3)
auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)    #Gives access to the account
print("Authenticating the account...")
sleep(3)
api=tweepy.API(auth)            #Connecting to the API to pull tweets
x=input("Enter the string on which the tweets are to be pulled: ")
tweets=api.search(x,count=100)         #Pulling tweets
print("Pulling tweets from Twitter...")
sleep(4)
senti=[]
tweet=[]
pos=0
neg=0


print("Predicting tweet sentiments...")
sleep(5)
for i in tweets:
    iarr=np.array([i.text])
    iaarve=ve.transform(iarr)
    z=classi.predict(iaarve)
    senti.append(z)
    tweet.append(i.text)
    if z[0]==0:
        neg+=1
    else:
        pos+=1
posp=(pos/len(tweets))*100
negp=(neg/len(tweets))*100



data={'Tweets':tweet,'Sentiment':senti}
ab=pd.DataFrame(data)
print("Writing results to Scores.txt ...")
sleep(7)
ab.to_csv("Scores.txt",sep="\t",mode="r+")



print("Plotting Pie Chart")
labels = 'Positive',"Negative"
sizes = [posp,negp]
m=max(posp,negp)
if m==posp:
    explode=(0.1,0)
else:
    explode = (0, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
sleep(2)
plt.show()
