import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn import svm, datasets

from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize
from collections import Counter

import sys

def create_wordcloud(dataframe, stop_words):

    data = {k: v['Title'] + ' ' + v['Content'] for k, v in df.groupby('Category')}

    for category, newframe  in data.items():
        count_vect = CountVectorizer(stop_words=stop_words,analyzer="word", token_pattern=r"(?u)\b\w\w*\'*\w+\b")
        X = count_vect.fit_transform(newframe)

        vocab = list(count_vect.get_feature_names())    

        counts = X.sum(axis=0).A1
        freq_distribution = Counter(dict(zip(vocab, counts)))

        top = freq_distribution.most_common(100)
        #print (top)

        wordcloud = WordCloud(width = 600, height = 400, background_color = 'white').generate_from_frequencies(dict(top))
        image = wordcloud.to_image()
        image.save(category + ".png")

df = pd.read_csv(sys.argv[1], sep="\t")

additional_stop_words = ['said', 'new', 'film', 'year', 'years', 'like', 'player', 'players', 'team', 'teams', 'game', 'say', 'time', 'times', 'says', 'club', 'movie', 'people']
stop_words = ENGLISH_STOP_WORDS.union(additional_stop_words).union(set(STOPWORDS))

print ("="*60)
print ("*******Creating Wordclouds*******")
create_wordcloud(dataframe=df, stop_words=stop_words)
