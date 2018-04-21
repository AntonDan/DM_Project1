import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
import argparse

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle
from sklearn import svm, datasets, preprocessing 

from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize
from collections import Counter

import sys

# Public vars
test_size = 0.25
svm_C = 1.0
naive_bayes_a = 0.05
random_forests_estimators = 100
k_fold = 10

def create_wordcloud(dataframe, stop_words):

	data = {k: v['Title'] + ' ' + v['Content'] for k, v in df.groupby('Category')}

	for category, newframe  in data.items():
		count_vect = CountVectorizer(stop_words=stop_words,analyzer="word", token_pattern=r"(?u)\b[a-zA-Z]+\'*[a-zA-Z]+\b")
		X = count_vect.fit_transform(newframe)

		vocab = list(count_vect.get_feature_names())    

		counts = X.sum(axis=0).A1
		freq_distribution = Counter(dict(zip(vocab, counts)))

		top = freq_distribution.most_common(100)
		#print (top)

		wordcloud = WordCloud(width = 600, height = 400, background_color = 'white').generate_from_frequencies(dict(top))
		image = wordcloud.to_image()
		image.save(category + ".png")

def classify(classifier, name, load_grids, load_labels, load_proba):
	print("< Beginning " + name + " classification >")

	count_vectorizer = CountVectorizer()
	transformer = TfidfTransformer()
	svd = TruncatedSVD(n_components=100, random_state=42)

	if name == "Multinomial Naive Bayes": 
		pipeline = Pipeline([
			('vect', count_vectorizer),
			('tfidf', transformer),
			('clf', classifier)
		])
	else:
		pipeline = Pipeline([
			('vect', count_vectorizer),
			('tfidf', transformer),
			('svd',svd),
			('clf', classifier)
		])

	grid_search = None
	if (load_grids and os.path.exists(name + ".pic")):
		print ("Loading grid data from file")
		grid_search = pickle.load(open(name + ".pic", "rb")) 
	else:
		if (load_grids):
			print ("Error: no " + name + " file")
		print ("Creating new grid for " + name)
		grid_search = GridSearchCV(pipeline, {}, cv=k_fold, n_jobs=8, verbose=1)
		grid_search.fit(train_data, train_labels)
		pickle.dump(grid_search, open(name + ".pic", "wb"))


	predicted_labels = None
	if (load_labels and os.path.exists(name + "_labels.pic")):
		print ("Loading prediction data from file")
		predicted_labels = pickle.load(open(name + "_labels.pic", "rb"))
	else:
		if (load_labels):
			print ("Error: no " + name + "_labels file")
		print ("Creating new label list for " + name + " labels")
		predicted_labels = grid_search.predict(test_data)
		pickle.dump(predicted_labels, open(name + "_labels.pic", "wb"))


	print ("Calculating label probabilities")
	label_proba = None
	if (load_proba and os.path.exists(name + "_proba.pic")):
		print ("Loading prediction data from file")
		label_proba = pickle.load(open(name + "_proba.pic", "rb"))
	else:
		if (load_proba):
			print ("Error: no " + name + "_proba file")
		print ("Creating new label prediction list for " + name + " labels")
		label_proba = grid_search.best_estimator_.predict_proba(test_data)
		pickle.dump(label_proba, open(name + "_proba.pic", "wb"))

	# Getting label names
	predicted_categories = le.inverse_transform(predicted_labels)

	print predicted_labels
	print predicted_categories
	print label_proba
	print ("\n")
	return predicted_categories

def print_step_info(step_name, info=""):
	print ("="*60)
	temp = 60 - len(step_name)
	print ("*"*(int(temp/2)) + step_name + "*"*(int(temp/2) + temp%2))
	print (info)





# MAIN
parser = argparse.ArgumentParser()
parser.add_argument("--train-file", help="The path to the training data file",  action="store", required=True)
parser.add_argument("--test-file", help="The path to the test data file", action="store", default=None)
parser.add_argument("--wordcloud", help="Load label probability data previously saved in .pic files", action="store_true")
parser.add_argument("--load-grids", help="Load grid data previously saved in .pic files", action="store_true")
parser.add_argument("--load-labels", help="Load label data previously saved in .pic files", action="store_true")
parser.add_argument("--load-probs", help="Load label probability data previously saved in .pic files", action="store_true")

args = parser.parse_args()

print ("Reading training set " + args.train_file)
df = pd.read_csv(args.train_file, sep="\t")
print ("Reading test set " + args.train_file)
test_df = None
if (args.test_file != None):
	test_df = pd.read_csv(args.test_file, sep="\t")

# Creating label list
le = preprocessing.LabelEncoder()
le.fit(df['Category'])
labels = le.transform(df['Category'])

# Merging content with title
f = lambda x: x['Title'] + ' ' + x['Content']

train_df = None
train_data   = None
test_data    = None
train_labels = None
test_labels  = None

if (test_df is not None):
	train_df = df
	train_labels = labels
	if ('Category' in test_df.columns):
		le.fit(test_df['Category'])
		test_labels = le.transform(test_df['Category'])
else:
	train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=test_size, random_state=0)

train_data = train_df[['Title','Content']]
test_data = test_df[['Title','Content']]
train_data = train_data.apply(f, 1)
test_data = test_data.apply(f, 1)


# Wordcloud creation
if (args.wordcloud):
	print_step_info(step_name="Creating Wordclouds")
	additional_stop_words = ['said', 'new', 'film', 'year', 'years', 'like', 'player', 'players', 'team', 'teams', 'game', 'say', 'time', 'times', 'says', 'club', 'movie', 'people']
	stop_words = ENGLISH_STOP_WORDS.union(additional_stop_words).union(set(STOPWORDS))
	create_wordcloud(dataframe=df, stop_words=stop_words)

print_step_info(step_name="Testing")

classifier_list = [
		(SVC(probability=True), "SVM","c"),
		(RandomForestClassifier(n_estimators=random_forests_estimators,n_jobs=-1), "Random forest","m"),
		(MultinomialNB(alpha=naive_bayes_a),"Multinomial Naive Bayes","y"),
		#(KNeighborsClassifier(n_neighbors=k_neighbors_num,n_jobs=-1), "k-Nearest Neighbor","g"),
	]

for clf, name, color in classifier_list:
	predicted_labels = classify(clf, name, args.load_grids, args.load_labels, args.load_probs)
	category_df = pd.DataFrame({"Predicted_Category" : predicted_labels})
	out_df = pd.DataFrame({"ID" : test_df['Id']})
	out_df = out_df.join(category_df)
	out_df.to_csv(name + "_output.csv", sep='\t')

#print feature_matrix.
#print classification_report(test_labels, predicted_labels, target_names=list(le.classes_))
#print accuracy_score(test_labels, predicted_labels)