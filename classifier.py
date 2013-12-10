from collections import Counter, defaultdict
from nltk.corpus import wordnet
from stemming.porter2 import stem
import nltk
import operator

class Classifier(object):
    def __init__(self, labels):
	"""
	@param (string, string): Pair of positive, negative labels
	@return string y: either the positive or negative label
	"""
	self.labels = labels

    def classify(self, text):
	"""
	@param string text: e.g. email
	@return double y: classification score; >= 0 if positive label
	"""
	raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, text):
	"""
	@param string text: the text message
	@return string y: either 'ham' or 'spam'
	"""
	if self.classify(text) >= 0.:
	    return self.labels[0]
	else:
	    return self.labels[1]

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as Counters,
    return their dot product.
    You might find it useful to use sum() and a list comprehension.
    """
    return sum([v1[k] * v2[k]
		for k in (v1.keys() if len(v1) < len(v2) else v2.keys())])

class WeightedClassifier(Classifier):
    def __init__(self, labels, featureFunction, params):
	"""
	@param (string, string): Pair of positive, negative labels
	@param func featureFunction: function to featurize text,
	e.g. extractUnigramFeatures
	@param dict params: the parameter weights used to predict
	"""
	super(WeightedClassifier, self).__init__(labels)
	self.featureFunction = featureFunction
	self.params = params

    def classify(self, x):
	"""
	@param string x: the text message
	@return double y: classification score; >= 0 if positive label
	"""
	phi = self.featureFunction(x)
	return self.classifyFeatureVector(phi)

    def classifyFeatureVector(self, phi):
	return sparseVectorDotProduct(self.params, phi)

def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels,
			       iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features
    @params labels: tuple of labels ('positive', 'negative'),
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature
    (string) to value.
    """
    weights = Counter()
    classifier = WeightedClassifier(labels, featureExtractor, weights)
    for i in range(iters):
	for x, y in trainExamples:
	    phi = featureExtractor(x)
	    y_prime = classifier.classifyFeatureVector(phi)
	    if ((y != labels[0] and y_prime >= 0) or
		(y == labels[0] and y_prime < 0)):
		if y == labels[0]:
		    weights.update(phi)
		else:
		    weights.subtract(phi)
    return weights

class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
	"""
	@param list string: List of labels
	@param list (string, Classifier): tuple of (label, classifier);
	the classifier is the one-vs-all classifier
	"""
	self.classifiers = classifiers

    def classify(self, x):
	"""
	@param string x: the text message
	@return list (string, double): list of labels with scores
	"""
	return [(label, classifier.classify(x))
		for label, classifier in self.classifiers]

    def classifyWithLabel(self, x):
	"""
	@param string x: the text message
	@return string y: the top two scoring output labels
	"""
	labels_and_scores = self.classify(x)
	maxPair = pairmax(labels_and_scores)
	labels_and_scores = [p for p in labels_and_scores if p != maxPair]
	maxPair2 = pairmax(labels_and_scores)
	return maxPair[0], maxPair2[0]

def pairmax(pairs):
    return None if len(pairs) == 0 else max(pairs, key=lambda p:p[1])

class OneVsAllClassifier(MultiClassClassifier):
    def __init__(self, labels, classifiers):
	"""
	@param list string: List of labels
	@param list (string, Classifier): tuple of (label, classifier);
	the classifier is the one-vs-all classifier
	"""
	super(OneVsAllClassifier, self).__init__(labels, classifiers)

    def classify(self, x):
	"""
	@param string x: the text message
	@return list (string, double): list of labels with scores
	"""
	return MultiClassClassifier.classify(self, x)

def learnOneVsAllClassifiers(trainExamples, featureFunction, labels,
			     perClassifierIters = 10):
    """
    Split the set of examples into one label vs all and train classifiers
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @param func featureFunction: function to featurize text,
    e.g. extractUnigramFeatures
    @param list string labels: List of labels
    @param int perClassifierIters: number of iterations to train each
    classifier
    @return list (label, Classifier)
    """
    retval = []
    for label in labels:
		per_classifer_labels = [label, "not " + label]
		weights = learnWeightsFromPerceptron(
		    trainExamples, featureFunction, per_classifer_labels,
		    perClassifierIters)
		classifier = WeightedClassifier(
		    per_classifer_labels, featureFunction, weights)
		retval.append((label, classifier))
    return retval

class PoetryClassifier(OneVsAllClassifier):
    def __init__(self,trainExamples, featureExtractor, labels, iters):
	classifiers = learnOneVsAllClassifiers(
	    trainExamples, featureExtractor, labels, iters)
	super(PoetryClassifier, self).__init__(labels, classifiers)

    def getParams(self, index):
        return self.classifiers[index][1].params

cached_tags = {}
cached_stems = {}
cached_rhyme_scores = {}

def rhymes(w1, w2):
    l = min(len(w1), len(w2))
    w1 = w1[len(w1)-l:]
    w2 = w2[len(w2)-l:]
    hammingDist = sum(ch1 != ch2 for ch1, ch2 in zip(w1, w2))
    return (float(hammingDist)/l < .25)

def getRhymeScore(x):
    x.replace(',', '').replace(':', '').replace(';', '')
    lines = x.split('\n')
    lastWords = []
    for line in lines:
	words = [word for word in line.split() if len(word) >= 3]
        if len(words) > 0:
            lastWords.append(words[len(words)-1])
    count = 0
    for i in xrange(len(lastWords)):
        for j in xrange(i+1, len(lastWords)):
            if rhymes(lastWords[i], lastWords[j]):
                count += 1
    return float(count)/len(lastWords)

def extractBigramFeatures(x):
    """
    Extract unigram + bigram features for a text document $x$.

    @param string x: represents the contents of an email message.
    @return dict: feature vector representation of x.
    """
    f = open("stopwords.txt", "r")
    stopwords = set(f.read().split())
    f.close()
    x = x.lower()
    x = x.replace('.', ' . ').replace('?', ' ? ').replace('!', ' ! ')
    retval = defaultdict(int)
    if x not in cached_rhyme_scores:
        cached_rhyme_scores[x] = getRhymeScore(x)
    retval['--RHYME--'] = cached_rhyme_scores[x]
    words = [word for word in x.split() if word not in stopwords]
    if x in cached_tags:
	   tagged = cached_tags[x]
    else:
	tagged = nltk.pos_tag(words)
	cached_tags[x] = tagged
    tagcount = defaultdict(int)
    for word, tag in tagged:
	retval[tag] += 1
	if word in cached_stems:
	    if cached_stems[word]:
		retval[cached_stems[word]] += 1
	else:
	    wordstem = stem(word)
	    if wordnet.synsets(wordstem):
		retval[wordstem] += 1
		cached_stems[word] = wordstem
	    else:
		cached_stems[word] = None
    for tag in tagcount:
        retval[tag] = float(tagcount[tag])/len(tagged)
    for i in range(0, len(words)):
	if (i == 0 or
	    words[i-1].endswith('.') or
	    words[i-1].endswith('?') or
	    words[i-1].endswith('!')):
	    bigram = '-BEGIN- ' + words[i]
	else:
	    bigram = words[i-1] + ' ' + words[i]
	retval[bigram] += 1
    return retval
