# extract features from list of text instances based on configuration set of features

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
import numpy as np

def function_words(texts):
	bow = []
	header = stopwords.words('english')
	for text in texts:	#get stopwords counts for each text
		counts = []
		tokens = nltk.word_tokenize(text)
		for sw in stopwords.words('english'):
			sw_count = tokens.count(sw)
			normed = sw_count/float(len(tokens))
			counts.append(normed)
		bow.append(counts)
	bow_np = np.array(bow).astype(float)
	return bow_np, header	

def punctuation(texts):
	bow = []
	puncs = ['.', ',', ':', ';', '-', '\'', '\"', '(', '!', '?']
	for text in texts:
		counts = []
		tokens = nltk.word_tokenize(text)
		
			


'''
TODO: write the following functions
1. def syntax(texts):
	common_pos_tags = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS','CC','PRP','VB','VBG']

    returns:  
    a. np matrix of dimension num_essays x 10, where each feature represents a count of common_pos_tags[i]
    b. corresponding header with names of features

2. def lexical(texts):
    returns:  
    a. np matrix of dimension num_essays x 30, where the 30 features are counts for the top 30 unigrams across all essays
    b. corresponding header with names of features 
    NOTE: remove stop words before computing the top 30 unigrams

3. def punctuation(texts):
	punct = '.,:;-\'\"(!?'

    returns: 
    a. np matrix of dimension num_essays x 10, where the 10 features are counts for the 10 punctuation marks provided
    b. corresponding header with names of features

4. def complexity(texts):
    compute the following features:
    - average number of characters per word
    - #unique words / #total words
    - average number of words per sentence
    - count of "long" words - words with >= 6 letters

    returns:
    a. np matrix of dimension num_essays x 4, where the 4 features are the complexity feature listed here
    b. corresponding header with names of features
'''

def extract_features(texts, conf):
	features = []
	headers = []

	if 'function_words' in conf:
		f,h = function_words(texts)
		features.append(f)
		headers.extend(h)
	'''
	if 'syntax' in conf:
		f,h = syntax(texts)
		features.append(f)
		headers.extend(h)

	if 'lexical' in conf:
		f,h = lexical(texts)
		features.append(f)
		headers.extend(h)

	if 'punctuation' in conf:
		f,h = punctuation(texts)
		features.append(f)
		headers.extend(h)

	if 'complexity' in conf:
		f,h = complexity(texts)
		features.append(f)
		headers.extend(h)
	'''

	all_features = np.concatenate(features,axis=1)
	return all_features, headers