import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
import argparse

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from scipy import stats
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

# Public parameteres
test_size = 0.3

svm_C = 1.0
naive_bayes_a = 0.05
random_forests_estimators = 10
svd_n_components = 100
k_fold = 10

tuned_parameters = {
		"SVM" :[ 
			#{'clf__kernel': ['rbf'], 'clf__gamma': stats.expon(scale=.1), 'clf__C': stats.expon(scale=10)}
			 {'clf__gamma': [0.6909752275782135], 'clf__C': [2.627930391414668], 'clf__kernel': ['rbf']}
		],

		"Multinomial Naive Bayes" : [
			{}
		],

		"Random forest" : [
			{}
		]
	}

# Create Wordcloud
# Initially we create a dictionary with an entry of titles and contents for earch category
# For every category a countvectorizer is created and the wordcloud is created by the 100 
# most common tokens in the frequency distribution   
def create_wordcloud(dataframe, stop_words):
	data = {k: v['Title'] + ' ' + v['Content'] for k, v in df.groupby('Category')}

	for category, newframe  in data.items():
		count_vect = CountVectorizer(stop_words=stop_words,analyzer="word", token_pattern=r"(?u)\b[a-zA-Z]+\'*[a-zA-Z]+\b")
		X = count_vect.fit_transform(newframe)

		vocab = list(count_vect.get_feature_names())    

		counts = X.sum(axis=0).A1
		freq_distribution = Counter(dict(zip(vocab, counts)))

		top = freq_distribution.most_common(100)

		wordcloud = WordCloud(width = 600, height = 400, background_color = 'white').generate_from_frequencies(dict(top))
		image = wordcloud.to_image()
		image.save(category + ".png")


# The classification function uses the pipeline in order to ease the procedure
# 10-fold cross validation is achieved through the GridSearchCV parameteres  
def classify(classifier, name, grid_params, load_grids, load_labels, load_proba):
	print("< Beginning " + name + " classification >")

	count_vectorizer = CountVectorizer()
	transformer = TfidfTransformer()
	svd = TruncatedSVD(n_components=svd_n_components, random_state=42)

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

	print_seperator()
	grid_search = None
	if (load_grids and os.path.exists(name + ".pic")):
		print ("Loading grid data from file")
		grid_search = pickle.load(open(name + ".pic", "rb")) 
	else:
		if (load_grids):
			print ("Error: no " + name + " file")
		print ("Creating new grid for " + name)
		print ("Running grid search with the following parameters: ")
		print (grid_params)
		#grid_search = RandomizedSearchCV(pipeline, param_distributions=grid_params, cv=k_fold, n_jobs=8, verbose=1)
		grid_search = GridSearchCV(pipeline, grid_params, cv=k_fold, n_jobs=8, verbose=1)
		grid_search.fit(train_data, train_labels)
		pickle.dump(grid_search, open(name + ".pic", "wb"))


	print_seperator()
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


	print_seperator()
	label_proba = None
	if (load_proba and os.path.exists(name + "_proba.pic")):
		print ("Loading prediction data from file")
		label_proba = pickle.load(open(name + "_proba.pic", "rb"))
	else:
		if (load_proba):
			print ("Error: no " + name + "_proba file")
		print ("Creating new label probability list for " + name + " labels")
		label_proba = grid_search.best_estimator_.predict_proba(test_data)
		pickle.dump(label_proba, open(name + "_proba.pic", "wb"))

	print ("Done!")
	print ("Found best result with params:")
	print grid_search.best_params_
	print_seperator()
	print predicted_labels
	print label_proba
	print ("\n")
	return predicted_labels


# Simple print function that displays a seperator and the step name 
def print_step_info(step_name, info=None):
	print ("="*60)
	temp = 60 - len(step_name)
	print ("*"*(int(temp/2)) + step_name + "*"*(int(temp/2) + temp%2))
	if (info is not None):
		print (info)
	print ('')

def print_seperator():
	print('-'*60)


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
		(MultinomialNB(alpha=naive_bayes_a),"Multinomial Naive Bayes","y"),
		(RandomForestClassifier(n_estimators=random_forests_estimators,n_jobs=-1), "Random forest","m"),
		#(KNeighborsClassifier(n_neighbors=k_neighbors_num,n_jobs=-1), "k-Nearest Neighbor","g"),
	]

for clf, name, color in classifier_list:
	for grid_params in tuned_parameters[name]:
		predicted_labels = classify(clf, name, grid_params, args.load_grids, args.load_labels, args.load_probs)
		predicted_categories = le.inverse_transform(predicted_labels)
		if (test_labels is not None):
			print (classification_report(predicted_labels, test_labels))		
		dic = {
			"Id" : test_df['Id'],
			"Category" : predicted_categories
		}
		out_df = pd.DataFrame(dic, columns=['Id', 'Category'])
		out_df.to_csv(name + "_output.csv", sep=',', index=False)

#print feature_matrix.
#print classification_report(test_labels, predicted_labels, target_names=list(le.classes_))
#print accuracy_score(test_labels, predicted_labels)