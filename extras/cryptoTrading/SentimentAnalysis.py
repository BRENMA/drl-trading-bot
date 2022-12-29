#import praw
#import pandas as pd
#from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
#import re
#
#tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
#model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
#pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding = 'max_length')
#
#count = 1000
#subReddits = ['CryptoCurrency', 'CryptoMarkets', 'CryptoCurrencies', "DeFi"]
#
#reddit = praw.Reddit(client_id=client_id, client_secret=secret, user_agent=user_agent)
#
#posts = []
#for i in subReddits:
#    hot_posts = reddit.subreddit(i).hot(limit=30)
#    innerPosts = []
#    for post in hot_posts:
#        textBody = post.title + " " + post.selftext
#        textBody = textBody.replace("***","").replace("\n","").replace("&#37","%").replace("&#x200B;","")
#        textBody = textBody.strip()
#        textBody = re.sub(r'http\S+', '', textBody).strip()
#        textBody = re.sub(r'(?<!\.)\n', '', textBody)
#        print(textBody)
#        print("\n\n\n\n")
#        innerPosts.append(textBody)
#
#    for post in innerPosts[2:]:
#        posts.append(post)
#        
#    print('\nNEW SUBREDDIT\n')
#
#preds = pipe(posts)
#print(preds)
#
#df = pd.DataFrame(preds)
#sentiment_counts = df.groupby(['label']).size()
#print(sentiment_counts)
#