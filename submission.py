"""
CS221 2013
AssignmentID: spam
"""

import util
import operator
from collections import Counter
from collections import defaultdict

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

class RuleBasedClassifier(Classifier):
    def __init__(self, labels, blacklist, n=1, k=-1):
        """
        @param (string, string): Pair of positive, negative labels
        @param list string: Blacklisted words
        @param int n: threshold of blacklisted words before email marked spam
        @param int k: number of words in the blacklist to consider
        """
        super(RuleBasedClassifier, self).__init__(labels)
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        super(RuleBasedClassifier, self).__init__(labels)
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        if k <= 0:
                self.blacklist_set = set(blacklist)
        else:
                self.blacklist_set = set()
                for i in xrange(min(k, len(blacklist))):
                    self.blacklist_set.add(blacklist[i])
        self.n = n
        # END_YOUR_CODE

    def classify(self, text):
        """
        @param string text: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        matches = 0
        for word in text.split():
            if word in self.blacklist_set:
                matches += 1
        return (1 if matches >= self.n else -1)
        # END_YOUR_CODE

def extractUnigramFeatures(x):
    """
    Extract unigram features for a text document $x$. 
    @param string x: represents the contents of an text message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    c = Counter()
    for word in x.split():
        c[word] += 1
    return c
    # END_YOUR_CODE

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as Counters,
    return their dot product.
    You might find it useful to use sum() and a list comprehension.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    retval = 0
    v = v1 if len(v1) < len(v2) else v2
    for k in v.keys():
        retval += v1[k] * v2[k]
    return retval
    # END_YOUR_CODE


class WeightedClassifier(Classifier):
    def __init__(self, labels, featureFunction, params):
        """
        @param (string, string): Pair of positive, negative labels
        @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
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
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        phi = self.featureFunction(x)
        return self.classifyFeatureVector(phi)
        # END_YOUR_CODE

    def classifyFeatureVector(self, phi):
        return sparseVectorDotProduct(self.params, phi)

        
def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels, iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features, e.g. extractUnigramFeatures
    @params labels: tuple of labels ('positive', 'negative'), e.g. ('spam', 'ham').
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature (string) to value.
    """
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
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
    # END_YOUR_CODE

def sanitize(x):
    # TODO: think of other sanitization steps?
    return x.lower()

def extractBigramFeatures(x):
    """
    Extract unigram + bigram features for a text document $x$. 

    @param string x: represents the contents of an email message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    f = open("stopwords.txt", "r")
    stopwords = f.read()
    f.close()
    stopset = set(stopwords.split())
    x = x.replace('.', ' . ').replace('?', ' ? ').replace('!', ' ! ')
    words = x.split()
    retval = defaultdict(int)
    words = [sanitize(word) for word in words if word not in stopset]
    for word in words:
        retval[word] += 1
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
    # END_YOUR_CODE

class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self.classifiers = classifiers
        # END_YOUR_CODE

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        retval = []
        for pair in self.classifiers:
            label = pair[0]
            score = pair[1].classify(x)
            retval.append((label, score))
        return retval

    def classifyWithLabel(self, x):
        """
        @param string x: the text message
        @return string y: one of the output labels
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        labels_and_scores = self.classify(x)
        max_label = labels_and_scores[0][0]
        max_score = labels_and_scores[0][1]
        for i in range(1, len(labels_and_scores)):
            label = labels_and_scores[i][0]
            score = labels_and_scores[i][1]
            if score > max_score:
                max_score = score
                max_label = label
        return max_label
        # END_YOUR_CODE

class OneVsAllClassifier(MultiClassClassifier):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        super(OneVsAllClassifier, self).__init__(labels, classifiers)

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        return MultiClassClassifier.classify(self, x)
        # END_YOUR_CODE

def learnOneVsAllClassifiers( trainExamples, featureFunction, labels, perClassifierIters = 10 ):
    """
    Split the set of examples into one label vs all and train classifiers
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
    @param list string labels: List of labels
    @param int perClassifierIters: number of iterations to train each classifier
    @return list (label, Classifier)
    """
    # BEGIN_YOUR_CODE (around 10 lines of code expected)
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
    # END_YOUR_CODE

