import operator

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

