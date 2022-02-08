from textblob import TextBlob
import csv
import tweepy
import unidecode
import os
from urllib.parse import unquote


from twitter_credentials import acc_secret,acc_token,con_key,con_secret

#Use tweepy.OAuthHandler to create an authentication using the given key and secret
auth = tweepy.OAuthHandler(consumer_key=con_key, consumer_secret=con_secret)
auth.set_access_token(acc_token, acc_secret)

#Connect to the Twitter API using the authentication
api = tweepy.API(auth)


def collect(search, num=10):


    tweet_list = []
    last_id = -1 # id of last tweet seen
    while len(tweet_list) < num:
        try:
            new_tweets = api.search(\
                q = search, \
                count = 10, \
                max_id = str(last_id - 1),
                tweet_mode='extended')
        except tweepy.TweepError as e:
            print("Error", e)
            break
        else:
            if not new_tweets:
                print("Could not find any more tweets!")
                break
            tweet_list.extend(new_tweets)
            last_id = new_tweets[-1].id

    return tweet_list
#**********************************************************************************************************
# descriptive statistics and histogram correlation analysis and regression to connect independent variable
#**********************************************************************************************************
# Tweeter search with keyword
target_num = 50
query = "climatechange"

csvFile = open('results_climatechange.csv','w',encoding='utf8')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(["username","author id","created", "text", "retwc", "hashtag", "followers", "friends","polarity","subjectivity"])
counter = 0

for tweet in tweepy.Cursor(api.search, q = query, lang = "en", result_type = "popular", count = target_num).items():
    created = tweet.created_at
    text = tweet.text
    text = unidecode.unidecode(text) 
    retwc = tweet.retweet_count
    try:
        hashtag = tweet.entities[u'hashtags'][0][u'text'] #hashtags used
    except:
        hashtag = "None"
    username  = tweet.author.name            #author/user name
    authorid  = tweet.author.id              #author/user ID#
    followers = tweet.author.followers_count #number of author/user followers (inlink)
    friends = tweet.author.friends_count     #number of author/user friends (outlink)

    text_blob = TextBlob(text)
    polarity = text_blob.polarity
    subjectivity = text_blob.subjectivity
    csvWriter.writerow([username, authorid, created, text, retwc, hashtag, followers, friends, polarity, subjectivity])

    counter = counter + 1
    if (counter == target_num):
        break

csvFile.close()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

change_climate = pd.read_csv('results_climatechange.csv')


plt.figure()
hist1,edges1 = np.histogram(change_climate.friends)
plt.bar(edges1[:-1],hist1,width=edges1[1:]-edges1[:-1])

plt.scatter(change_climate.followers,change_climate.retwc)


print(change_climate.corr())

# import statsmodels.api, as sm. So to do the linear modeling
plt.scatter(change_climate.polarity, change_climate.subjectivity)
x = change_climate.polarity
y = change_climate.subjectivity
x = sm.add_constant(x)

# Ordinary Least Square Function to give us details about the regression
lr_model = sm.OLS(y,x).fit()

print(lr_model.summary())

x_prime = np.linspace(x.polarity.min(),x.polarity.max(), 100)

x_prime = sm.add_constant(x_prime)
y_hat = lr_model.predict(x_prime)
plt.scatter(x.polarity,y)
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')

plt.plot(x_prime[:,1],y_hat)












