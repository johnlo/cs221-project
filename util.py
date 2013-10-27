import operator
import os
import random
from collections import Counter, defaultdict

def loadTrainAndDevExamples(path, labels, numExamples):
    def loadExamples(path, labels):
        """
        Reads examples from disk.
        @param string path: a directory containing subdirectories, one for each
        label.  Each file in each of those directories is an example:
        <path>/<label>/<example1>
        @return list of examples: (x, y) pairs, where x is an email (string)
        and y is a label (string).
        """
        examples = []
        for label in labels:
            i = 0
            for emailFile in os.listdir(os.path.join(path, label)):
                x = open(os.path.join(path, label, emailFile)).read()
                y = label
                examples.append((x, y))
                i += 1
        # Randomly shuffle the examples
        random.seed(41)
        random.shuffle(examples)
        return examples
    def holdoutExamples(examples, frac=0.2):
        """
        @param list examples
        @param float frac: fraction of examples to holdout.
        @return (examples1, examples2): two lists of examples.
        """
        examples1 = []
        examples2 = []
        random.seed(42)
        for ex in examples:
            if random.random() < frac:
                examples2.append(ex)
            else:
                examples1.append(ex)
        return (examples1, examples2)
    examples = loadExamples(path, labels)[:numExamples]
    return holdoutExamples(examples)

def computeConfusionMatrix(examples, classifier): 
    """
    @param list examples
    @param Classifier classifier: 
    @return float[][]: confusion matrix; rows are true labels, columns are
    predicted
    """
    # First extract all keys
    keys = set([])
    for _, y in examples:
        keys.add(y)

    confusion = {}
    for y in keys:
        confusion[y] = dict( ( (y_, 0) for y_ in keys ) )
    for x, y in examples:
        y1, y2 = classifier.classifyWithLabel(x)
        confusion[y][y1] = confusion[y].get(y1, 0) + 1
        confusion[y][y2] = confusion[y].get(y2, 0) + 1
    return confusion

def printConfusionMatrix(confusion): 
    """
    @param list examples
    @param Classifier classifier: 
    @return float[][]: confusion matrix; rows are true labels, columns are
    predicted
    """
    print "\t" + "\t".join(confusion.keys())
    for key in confusion.keys():
            print key + "\t" + "\t".join(map(str, confusion[key].values()))

def computeErrorRate(examples, classifier): 
    """
    @param list examples
    @param dict params: parameters
    @param function predict: (params, x) => y
    @return float errorRate: fraction of examples we make a mistake on.
    """
    numErrors = 0
    for x, y in examples:
        y1, y2 = classifier.classifyWithLabel(x)
        if not (y1 == y or y2 == y):
            numErrors += 1
    return 1.0 * numErrors / len(examples)

def evaluateClassifier(trainExamples, devExamples, classifier):
    printConfusionMatrix(computeConfusionMatrix(trainExamples, classifier))
    trainErrorRate = computeErrorRate(trainExamples, classifier)
    print 'trainErrorRate: %f' % trainErrorRate
    printConfusionMatrix(computeConfusionMatrix(devExamples, classifier))
    devErrorRate = computeErrorRate(devExamples, classifier)
    print 'devErrorRate: %f' % devErrorRate
