import numpy as np
import itertools
import operator

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from sklearn.utils.estimator_checks import check_estimator

# Our custom K-Nearest Neighbor implementation according to the scikit-learn standard
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

		# Store arguments, since we have a lazy student type classifier
		self.X_ = X
		self.y_ = y

		return self

	def predict(self, X):
		# Check if fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X, accept_sparse='csr')

		predicted_labels = list()

		# We create a similarity matrix between our train set and our test set
		# This pairwise function operates on the train and the test set, to create a 2D matrix of cosine similarity between them
		similarity_matrix = cosine_similarity(self.X_, X)

		for idx, document in enumerate(X):
			neighbors = self.get_neighbors(similarity_matrix, idx)
			predicted_labels.append(self.majority_voting(neighbors))

		return np.array(predicted_labels)

	# Getting neighbors by finding the most similar documents through the similarity matrix
	def get_neighbors(self, similarity_matrix, document_index):
		potential_neighbors = list()
		for idx, document in enumerate(self.X_):
			potential_neighbors.append(similarity_matrix[idx, document_index])

		enumlist = enumerate(potential_neighbors)
		pot_nlist = sorted(enumlist, key=lambda enumlist: enumlist[1], reverse = True)
		return list(itertools.islice(pot_nlist, self.n_neighbors))

	# Weighted majority voting. Votes are cast by using the similarity to the test document, so that predictions are more accurate
	def majority_voting(self, neighbors):
		majority_dict = dict()

		for neighbor_index, similarity in neighbors:
			if self.y_[neighbor_index] in majority_dict:
				majority_dict[self.y_[neighbor_index]] += similarity
			else:
				majority_dict[self.y_[neighbor_index]] = similarity

		return max(majority_dict.items(), key=operator.itemgetter(1))[0]