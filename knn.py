import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import itertools
import scipy
import nltk
import operator

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from sklearn import svm, datasets, preprocessing, neighbors
from sklearn.utils.estimator_checks import check_estimator

from gensim import corpora, models, similarities
from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize
from collections import Counter
from math import sqrt

class KNeighborsClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, n_neighbors=5, n_jobs=1):
		self.n_neighbors = n_neighbors
		self.n_jobs = n_jobs

	def fit(self, X, y):
		# Check that X and y have correct shape
		X = check_array(X, accept_sparse='csr')
		X, y = check_X_y(X, y, accept_sparse=True)

		# Store the classes seen during fit
		self.classes_ = unique_labels(y)

		self.X_ = X
		self.y_ = y

		return self

	def predict(self, X):
		# Check if fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X, accept_sparse='csr')

		predicted_labels = list()

		similarity_matrix = cosine_similarity(self.X_, X)

		for idx, document in enumerate(X):
			neighbors = self.get_neighbors(similarity_matrix, idx)
			predicted_labels.append(self.majority_voting(neighbors))

		return np.array(predicted_labels)

	def get_neighbors(self, similarity_matrix, document_index):
		potential_neighbors = list()
		for idx, document in enumerate(self.X_):
			potential_neighbors.append(similarity_matrix[idx, document_index])

		enumlist = enumerate(potential_neighbors)
		pot_nlist = sorted(enumlist, key=lambda enumlist: enumlist[1], reverse = True)
		return list(itertools.islice(pot_nlist, self.n_neighbors))


	def majority_voting(self, neighbors):
		majority_dict = dict()

		for neighbor_index, similarity in neighbors:
			if self.y_[neighbor_index] in majority_dict:
				majority_dict[self.y_[neighbor_index]] += similarity
			else:
				majority_dict[self.y_[neighbor_index]] = similarity

		return max(majority_dict.items(), key=operator.itemgetter(1))[0]


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train-file", help="The path to the training data file",  action="store", required=True)
	parser.add_argument("--test-file", help="The path to the test data file", action="store", default=None)

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

	# truncate test data
	# train_data = train_data[:train_data_trunk]
	# test_data = test_data[:test_data_trunk]
	# train_labels = train_labels[:train_data_trunk]
	# test_labels = test_labels[:test_data_trunk]

	print("="*60)

	# Creating feature matix
	count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
	feature_count = count_vectorizer.fit_transform(train_data)
	test_feature_count = count_vectorizer.transform(test_data)

	tfidftrans = TfidfTransformer(use_idf=True)
	feature_matrix = tfidftrans.fit_transform(feature_count)
	test_feature_matrix = tfidftrans.transform(test_feature_count)

	# feature_tfidf = tfidftrans.fit_transform(feature_count)
	# test_feature_tfidf = tfidftrans.fit_transform(test_feature_count)
	# svd = TruncatedSVD(random_state=42)
	# feature_matrix = svd.fit_transform(feature_tfidf)
	# test_feature_matrix = svd.fit_transform(test_feature_tfidf)

	# Creating model
	# model = neighbors.KNeighborsClassifier(n_neighbors = k_neighbors_num)
	model = KNeighborsClassifier(n_neighbors = k_neighbors_num)

	# Fitting train set
	model.fit(feature_matrix, train_labels)
	# Predicting test set
	predicted_labels = model.predict(test_feature_matrix)
	# Getting label names
	# predicted_categories = le.inverse_transform(predicted_labels)

	# check_estimator(KNeighborsClassifier)

	cnt = Counter()
	for label in labels:
		cnt[label] += 1
	print(cnt)
	print(list(test_labels))
	print(predicted_labels)
	print(predicted_labels.shape)

	# predicted_categories = le.inverse_transform(predicted_labels)
	# print(le.inverse_transform(test_labels))
	# print(predicted_categories)

	print(classification_report(test_labels, predicted_labels, target_names=list(le.classes_)))

	# print(feature_matrix.shape)
	# print(predicted_categories)

	print("="*60)