import pandas as pd
import numpy as np 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize
from collections import Counter

import sys

def create_wordcloud(dataframe, stop_words):

	wordclouds = dict()
	for index, row in dataframe.iterrows():
		if (row.Category in wordclouds):
			wordclouds[row.Category] += (row.Content.lower() + ' ')
		else:
			wordclouds[row.Category] = (row.Content.lower() + ' ')

	for key, value in wordclouds.items():
		wordcloud = WordCloud(width = 600, height = 400, stopwords = stop_words, background_color = 'white').generate(value)
		image = wordcloud.to_image()
		image.save(key + ".png")


dataset = pd.read_csv(sys.argv[1], sep="\t")

newframe = dataset[["Title", "Content"]]
f = lambda x: x['Title'] + ' ' + x['Content']
newframe = newframe.apply(f, 1)

additional_stop_words = ['said', 'new', 'film', 'year', 'years', 'like', 'player', 'players', 'team', 'teams', 'game', 'say', 'time', 'times', 'says', 'club', 'movie', 'people']
stop_words = ENGLISH_STOP_WORDS.union(additional_stop_words).union(set(STOPWORDS))

count_vect = CountVectorizer(stop_words=stop_words)
X = count_vect.fit_transform(newframe)

vocab = list(count_vect.get_feature_names())

counts = X.sum(axis=0).A1
freq_distribution = Counter(dict(zip(vocab, counts)))

print (freq_distribution.most_common(10))