import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import sys
import os
import errno
import re
import argparse
import nltk

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer 

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.utils import shuffle
from sklearn import svm, datasets, preprocessing 
from scipy import stats
from sklearn import neighbors

from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize
from collections import Counter

from knn import KNeighborsClassifier

nltk.download("stopwords") 
nltk.download("wordnet")
#stemmer = SnowballStemmer("english", ignore_stopwords=True)
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

class LemmatizedStemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(LemmatizedStemmedCountVectorizer, self).build_analyzer()
		return lambda doc: ([stemmer.stem(lemmatizer.lemmatize(w)) for w in analyzer(doc)])

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end

# Public parameteres
output_directories = {
	'main' : '',
	'wourcloud' : 'wordclouds/',
	'data' : 'data/',
	'csv' : 'csv/',
	'plot' : 'plots/'
}  

random_forests_estimators = 10
k_fold = 10

tuned_parameters = {
		"SVC" : 
		[	#{'tfidf__use_idf': [True], 'svd__n_components' : [100], 'clf__kernel': ['rbf'], 'clf__gamma': stats.expon(scale=.1), 'clf__C': stats.expon(scale=10), 'clf__class_weight': ['balanced']}
			 {'tfidf__use_idf': [True], 'svd__n_components' : [100], 'clf__gamma': [0.6909752275782135], 'clf__C': [2.627930391414668], 'clf__class_weight': ['balanced'], 'clf__kernel': ['rbf'], 'clf__probability': [True]}
		],

		"GPC" : 
		[
			{
				'svd__n_components' : [100],
				"clf__optimizer": ["fmin_l_bfgs_b"],
				"clf__n_restarts_optimizer": [1],
			#	"clf__normalize_y": [False],
				"clf__copy_X_train": [True], 
				"clf__random_state": [0]
			}
		],
		"MNB" : 
		[
			{ "clf__alpha" : [0.025] }
		],
		"GNB" : 
		[
			{ }
		],

		"LR" : 
		[
			{ 'svd__n_components' : [100], "clf__tol" : [1e-5] }
		],

		"RF" : 
		[	
			{'svd__n_components' : [100], 'clf__n_estimators' : [200], 'clf__class_weight': ['balanced']}
		],

		"KNN" : 
		[	
			{'svd__n_components' : [100]}
		],

		"KNNC" : 
		[	
			{'svd__n_components' : [100]}
		],

		"VE" :
		[
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

		wordcloud = WordCloud(width = 600, height = 400, background_color = "white").generate_from_frequencies(dict(top))
		image = wordcloud.to_image()
		image.save(output_directories['wordcloud'] + category + ".png")


# The classification function uses the pipeline in order to ease the procedure
# Mutlinomial Naive Bayes does not support SVD preprocessing due to its nature 
# because of that we remove it from the pipeline when using MNB classification
# 10-fold cross validation is achieved through the GridSearchCV parameteres
# Random search available for searching parameters using scipy methods
# Search is run in pararel to reduce execution time
# Probability data are calculated in case they are needed for graph creation
# Data can be loaded from files when rerunning the program, this helps save time
# when making changes in later stages, since data doesn't have to be recalculated
def classify(classifier, name, grid_params, load_grids, load_labels, load_proba, random_search):
	print("< Beginning " + name + " classification >")

	#vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
	#vectorizer = StemmedCountVectorizer(stop_words=ENGLISH_STOP_WORDS)
	#vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=ENGLISH_STOP_WORDS)
	vectorizer = LemmatizedStemmedCountVectorizer(stop_words=ENGLISH_STOP_WORDS)
	#vectorizer = HashingVectorizer(stop_words=ENGLISH_STOP_WORDS)
	transformer = TfidfTransformer()

	# Initialization
	if name == "Multinomial Naive Bayes" or name == "Voting Estimator": 
		pipeline = Pipeline([
			('vect', vectorizer),
			('tfidf', transformer),
			('clf', classifier)
		])
	else:
		svd = TruncatedSVD(random_state=42)
		pipeline = Pipeline([
			('vect', vectorizer),
			('tfidf', transformer),
			('svd',svd),
			('clf', classifier)
		])

	# Search creation and fitting
	print_seperator()
	grid_search = None
	if (load_grids and os.path.exists(output_directories['data'] + name + ".pic")):
		print ("Loading grid data from file " + output_directories['data'] + name + ".pic") 
		grid_search = pickle.load(open(output_directories['data'] + name + ".pic", "rb")) 
	else:
		if (load_grids):
			print ("Error: no " + name + " file <" + output_directories['data'] + name + ".pic>")
		print ("Creating new grid for " + name)
		print ("Running grid search with the following parameters: ")
		print (grid_params)
		if (random_search):
			grid_search = RandomizedSearchCV(pipeline, param_distributions=grid_params, cv=k_fold, n_jobs=8, verbose=1)
		else:
			grid_search = GridSearchCV(pipeline, grid_params, cv=k_fold, n_jobs=-1, verbose=1)
		grid_search.fit(train_data, train_labels)
		pickle.dump(grid_search, open(output_directories['data'] + name + ".pic", "wb"))


	# Label prediction 
	print_seperator()
	predicted_labels = None
	if (load_labels and os.path.exists(output_directories['data'] + name + "_labels.pic")):
		print ("Loading prediction data from file " + output_directories['data'] + name + "_labels.pic")
		predicted_labels = pickle.load(open(output_directories['data'] + name + "_labels.pic", "rb"))
	else:
		if (load_labels):
			print ("Error: no " + name + "_labels file <" + output_directories['data'] + name + "_labels.pic>")
		print ("Creating new label list for " + name + " labels")
		predicted_labels = grid_search.predict(test_data)
		pickle.dump(predicted_labels, open(output_directories['data'] + name + "_labels.pic", "wb"))


	# Proba calculation
	print_seperator()
	label_proba = None
	if (name != "k-Nearest Neighbor Custom" and name != "Voting Estimator"):
		if (load_proba and os.path.exists(output_directories['data'] + name + "_proba.pic")):
			print ("Loading prediction data from file " + output_directories['data'] + name + "_proba.pic")
			label_proba = pickle.load(open(output_directories['data'] + name + "_proba.pic", "rb"))
		else:
			if (load_proba):
				print ("Error: no " + name + "_proba file <" + output_directories['data'] + name + "_proba.pic>")
			print ("Creating new label probability list for " + name + " labels")
			label_proba = grid_search.best_estimator_.predict_proba(test_data)
			pickle.dump(label_proba, open(output_directories['data'] + name + "_proba.pic", "wb"))

	print ("Done!")
	print ("Found best result with params:")
	print (grid_search.best_params_)
	print_seperator()
	#print predicted_labels
	#print label_proba
	print ("\n")
	return predicted_labels, label_proba

# roc_curve_estimator, converts the set to binary an then estimates the auc
# For the ROC AUC plot we make use of the following examles from sklearn
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#example-model-selection-plot-roc-py
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
def roc_curve_estimator(test_labels, label_proba, clfname, color):
	y_binary = preprocessing.label_binarize(test_labels, le.transform(le.classes_))
	fpr, tpr, thresholds = roc_curve(y_binary[:,1],label_proba[:,1])
	roc_auc = auc(fpr, tpr)
	print ("Area under the ROC curve: %f" % roc_auc)
	#print ("Thresholds: %f", thresholds)
	plt.plot(fpr, tpr, 'k', label="%s , (area = %0.3f)" % (clfname,roc_auc), lw=2, c="%s" % color)

	return roc_auc

# Simple print function that displays a seperator and the step name 
def print_step_info(step_name, info=None):
	print ("="*60)
	temp = 60 - len(step_name)
	print ("*"*(int(temp/2)) + step_name + "*"*(int(temp/2) + temp%2))
	if (info is not None):
		print (info)
	print ('')

# Simple seperator
def print_seperator():
	print('-'*60)


# MAIN
# Argument definition
parser = argparse.ArgumentParser()
parser.add_argument("--train-file", help="The path to the training data file",  action="store", required=True)
parser.add_argument("--test-file", help="The path to the test data file", action="store", default=None)
parser.add_argument("--classifier", help="The classifier that will be used", action="store", default=None, choices=["SVC", "GPC", "MNB", "RF", "KNN", "GNB", "KNNC", "LR"])
parser.add_argument("--output-dir", help="Output directory. ", default="./output/")
parser.add_argument("--wordcloud", help="Load label probability data previously saved in .pic files", action="store_true")
parser.add_argument("--load-grids", help="Load grid data previously saved in .pic files", action="store_true")
parser.add_argument("--load-labels", help="Load label data previously saved in .pic files", action="store_true")
parser.add_argument("--load-probs", help="Load label probability data previously saved in .pic files", action="store_true")
parser.add_argument("--random-search", help="Load label probability data previously saved in .pic files", action="store_true")
parser.add_argument("--voting", help="Run a voting estimator with all available classifiers", action="store_true")
parser.add_argument("--subsample-train", help="Create a balanced subsample of the train data", action="store", type=float, default=None, choices=[Range(0.01, 1.0)])

# Argument parsing and validation
args = parser.parse_args()

print ("Reading training set " + args.train_file)
df = pd.read_csv(args.train_file, sep="\t")
print ("Reading test set " + args.train_file)
test_df = None
if (args.test_file != None):
	test_df = pd.read_csv(args.test_file, sep="\t")
if (args.output_dir[-1] != '/'):
	args.output_dir += '/'
for key, value in output_directories.items():
	dirpath = args.output_dir  + value 
	if not os.path.exists(os.path.dirname(dirpath)):
		try:
			print ("Directory " + dirpath + " doesn't exist. Creating new one")
			os.makedirs(os.path.dirname(dirpath))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	output_directories[key] = args.output_dir + value # update output directories

# Creating label list
print ("Creating label list")
le = preprocessing.LabelEncoder()
le.fit(df['Category'])
labels = le.transform(df['Category'])

# Merging content with title
f = lambda x: x['Title'] + ' ' + x['Content']

train_df = None
train_data = None
test_data  = None
train_labels = None
test_labels  = None

if (test_df is not None):
	if (args.subsample_train is not None):
		print ("Subsampling %f%% of the train data",  args.subsample_train * 100)
		train_df, _, train_labels, _ = train_test_split(df, labels, test_size=args.subsample_train, random_state=0)
	else:
		train_df = df
		train_labels = labels
	if ('Category' in test_df.columns):
		le.fit(test_df['Category'])
		test_labels = le.transform(test_df['Category'])
else:
	if (args.subsample_train is not None):
		train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=1 - args.subsample_train, random_state=0)
	else:
		train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=0.3, random_state=0)


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

# Classification
print_step_info(step_name="Classification")

# Initialization
classifier_list = {
	#	"GPC" : (GaussianProcessClassifier(), "Gaussian Process Classifier","b"), # Don't run this unless you have a LOT of ram available
    #	"GNB" : (GaussianNB(), "Gaussian Naive Bayes","r"),
		"KNNC" : (KNeighborsClassifier(5, 0), "k-Nearest Neighbor Custom","g", 1.5),
		"KNN" : (neighbors.KNeighborsClassifier(n_neighbors=5), "k-Nearest Neighbor","g", 1),
		"MNB" : (MultinomialNB(),"Multinomial Naive Bayes","y", 1),
		"LR"  : (LogisticRegression(random_state=42), "Logistic Regression","k", 1),
		"RF"  : (RandomForestClassifier(n_estimators=100, class_weight='balanced'), "Random forest","m", 1),
		"SVC" : (SVC(gamma=0.7, C=2.6, kernel='rbf', probability=True, class_weight='balanced'), "Support Vector Classifier","c", 1.5)
}

validation_results = {"Accuracy": {}, "ROC": {}, "CompGraph": {}, "Predictions": {}}

estimators = []
weights = []

# Begining classifiaction
for clf_id, clf_info in classifier_list.items():
	if (args.classifier is not None and clf_id != args.classifier):
		continue
	clf, name, color, weight = clf_info
	estimators += [(name, clf)]
	weights += [weight]
	if (args.voting):
		continue
	predicted_labels, label_proba = classify(clf, name, tuned_parameters[clf_id], args.load_grids, args.load_labels, args.load_probs, args.random_search)
	predicted_categories = le.inverse_transform(predicted_labels)

	# Outputting results
	if (test_labels is not None):
		print_step_info(step_name="Classification Report")
		print (classification_report(predicted_labels, test_labels))
		accuracy = accuracy_score(test_labels, predicted_labels)
		print ("Accuracy: " + str(accuracy))
		validation_results["Accuracy"][name] = accuracy 
		if (label_proba is not None):
			roc_auc = roc_curve_estimator(test_labels, label_proba, name, color)
			validation_results["ROC"][name] = roc_auc
		
	# Outputting resulting dataframe to csv
	print ("Outputting resulting dataframe in " + output_directories['csv'] + name + "_output.csv")	
	validation_results["Predictions"][name] = predicted_labels
	dic = {
		"Id" : test_df['Id'],
		"Category" : predicted_categories
	}
	out_df = pd.DataFrame(dic, columns=['Id', 'Category'])
	out_df.to_csv(output_directories['csv'] + name + "_output.csv", sep=',', index=False)

# Run the voting classifier if asked
if (args.voting):
	print ("Voting Estimators:")
	for name, clf in estimators:
		print (name)
	print ("Weights:")
	print (weights)
	clf = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
	predicted_labels, _ = classify(clf, "Voting Estimator", tuned_parameters["VE"], args.load_grids, args.load_labels, args.load_probs, args.random_search)
	predicted_categories = le.inverse_transform(predicted_labels)
	if (test_labels is not None):
		print_step_info(step_name="Classification Report")
		print (classification_report(predicted_labels, test_labels))
		accuracy = accuracy_score(test_labels, predicted_labels)
		print ("Accuracy: " + str(accuracy))
		validation_results["Accuracy"][name] = accuracy 
	
	# Outputting resulting dataframe to csv
	print ("Outputting resulting dataframe in " + output_directories['csv'] + name + "_output.csv")	
	validation_results["Predictions"][name] = predicted_labels
	dic = {
		"Id" : test_df['Id'],
		"Category" : predicted_categories
	}
	out_df = pd.DataFrame(dic, columns=['Id', 'Category'])
	out_df.to_csv(output_directories['csv'] + name + "_output.csv", sep=',', index=False)

if ([test_labels is not None]):
	#create the ROC plot with the data generate from above
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right")
	plt.savefig(output_directories['plot']  + "roc_10fold.png")
