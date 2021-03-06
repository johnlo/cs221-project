import classifier
import datetime
import operator
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
import poem

MAIN_MOODS = ('Excited', 'Tender', 'Scared', 'Angry', 'Sad', 'Happy')
EXCITED_MOODS = ('Ecstatic', 'Energetic', 'Aroused', 'Bouncy', 'Nervous',
		 'Perky', 'Antsy')
TENDER_MOODS = ('Intimate', 'Loving', 'Warm-hearted', 'Sympathetic', 'Touched',
		'Kind', 'Soft')
SCARED_MOODS = ('Tense', 'Nervous', 'Anxious', 'Jittery', 'Frightened',
		'Panic-stricken', 'Terrified')
ANGRY_MOODS = ('Irritated', 'Resentful', 'Miffed', 'Upset', 'Mad', 'Furious',
	       'Raging')
SAD_MOODS = ('Down', 'Blue', 'Mopey', 'Grieved', 'Dejected', 'Depressed',
	     'Heartbroken')
HAPPY_MOODS = ('Fulfilled', 'Contented', 'Glad', 'Complete', 'Satisfied',
	       'Optimistic', 'Pleased')
ALL_MOOD_CLASSES = (MAIN_MOODS, EXCITED_MOODS, TENDER_MOODS, SCARED_MOODS,
		    ANGRY_MOODS, SAD_MOODS, HAPPY_MOODS)

def getClassifierForMainMood(classifiers, main_mood):
    for i in xrange(len(MAIN_MOODS)):
	if main_mood == MAIN_MOODS[i]:
	    return classifiers[i+1]
    return None

def loadPoemsFromDisk(path, numExamples):
    poems = {}
    for poemdir in os.listdir(path)[:numExamples]:
	text = open(os.path.join(path, poemdir, 'text')).read()
	moods = tuple(open(os.path.join(path, poemdir, 'mood')).read().split())
	poems[text] = (poemdir, text, moods)
    return poems

def shuffleIntoTrainAndDevExamples(rawExamples, labels):
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
    formatted = []
    for example in rawExamples.values():
	_, x, (y1, yy1, y2, yy2) = example
	if y1 in labels:
	    formatted.append((x, y1))
	if yy1 in labels:
	    formatted.append((x, yy1))
	if y2 in labels:
	    formatted.append((x, y2))
	if yy2 in labels:
	    formatted.append((x, yy2))
    return holdoutExamples(formatted)

def evaluate(poems, classifiers, examples, debugscore):
    total_score = 0
    total_possible = 0
    total_errors = 0
    total_examples = len(examples)
    for text, _ in examples:
	gpoemdir, _, (gmain1, gsub1, gmain2, gsub2) = poems[text]
	main1, main2 = classifiers[0].classifyWithLabel(text)
	sub1, _ = getClassifierForMainMood(
	    classifiers, main1).classifyWithLabel(text)
	sub2, _ = getClassifierForMainMood(
	    classifiers, main2).classifyWithLabel(text)
	score = 0
	if gmain1 == main1 or gmain1 == main2:
	    score += 2
	if gmain2 == main1 or gmain2 == main2:
	    score += 2
	if gsub1 == sub1 or gsub1 == sub2:
	    score += 1
	if gsub2 == sub1 or gsub2 == sub2:
	    score += 1
	if score == 0:
	    total_errors += 1
	if score <= debugscore:
	    print ""
	    print gpoemdir
	    print "expected:", gmain1, gsub1, gmain2, gsub2
	    print "actual:", main1, sub1, main2, sub2
	total_score += score
	total_possible += 6
    return total_score, total_possible, total_errors, total_examples

def printPercentage(a, b):
    return "%d of %d (%f)" % (a, b, float(a)/b)

def classify(args):
    poems = loadPoemsFromDisk(args.path, args.examples)
    classifiers = []
    trainDevSets = {}
    train_score = 0
    train_possible = 0
    train_errors = 0
    train_examples = 0
    dev_score = 0
    dev_possible = 0
    dev_errors = 0
    dev_examples = 0
    for labels in ALL_MOOD_CLASSES:
	train, dev = shuffleIntoTrainAndDevExamples(poems, labels)
	trainDevSets[labels] = (train, dev)
	classifiers.append(classifier.PoetryClassifier(
	    train, classifier.extractBigramFeatures, labels, args.iters))
    for labels in ALL_MOOD_CLASSES:
	train, dev = trainDevSets[labels]
	s, p, e, n = evaluate(poems, classifiers, train, args.debugscore)
	train_score += s
	train_possible += p
	train_errors += e
	train_examples += n
	s, p, e, n = evaluate(poems, classifiers, dev, args.debugscore)
	dev_score += s
	dev_possible += p
	dev_errors += e
	dev_examples += n
    print ""
    print "Score"
    print "train:", printPercentage(train_score, train_possible)
    print "dev:", printPercentage(dev_score, dev_possible)
    print ""
    print "Error Rate"
    print "train:", printPercentage(train_errors, train_examples)
    print "dev:", printPercentage(dev_errors, dev_examples)

def moodIndex(mood):
    return list(MAIN_MOODS).index(mood)

def submoodIndex(mood, submood):
    return list(ALL_MOOD_CLASSES[moodIndex(mood)+1]).index(submood)

def incrementSparseVector(v1, scale, v2):
    for key in v1.keys():
        v1[key] += v2[key] * scale

def getClassifierParams(classifiers, mood, submood):
    if not submood:
	return classifiers[0].getParams(moodIndex(mood))
    retval = classifiers[0].getParams(moodIndex(mood))
    incrementSparseVector(
        retval, 1,
        getClassifierForMainMood(classifiers, mood).getParams(
            submoodIndex(mood, submood)))
    return retval

def loadClassifiers(path, examples, iters):
    pkl_filename = 'classifiers.pkl'
    if not os.path.isfile(pkl_filename):
	poems = loadPoemsFromDisk(path, examples)
	classifiers = []
	trainDevSets = {}
	for labels in ALL_MOOD_CLASSES:
	    train, dev = shuffleIntoTrainAndDevExamples(poems, labels)
	    trainDevSets[labels] = (train, dev)
	    classifiers.append(classifier.PoetryClassifier(
		    train, classifier.extractBigramFeatures, labels, args.iters))
	pkl_file = open(pkl_filename, 'wb')
	pickle.dump(classifiers, pkl_file)
    pkl_file = open(pkl_filename, 'rb')
    return pickle.load(pkl_file)

def generate(args):
    classifiers = loadClassifiers(args.path, args.examples, args.iters)
    params = getClassifierParams(classifiers, args.mood, args.submood)
    mp = poem.MarkovPoem(args.path, args.ngramSize, args.mood, args.submood, params)
    print mp.generate(args.rhyme, args.meter)

def score(args):
    classifiers = loadClassifiers(args.path, args.examples, args.iters)
    i = random.choice(range(len(MAIN_MOODS)))
    j = random.choice(range(len(ALL_MOOD_CLASSES[i+1])))
    while True:
	id = int(time.time())
	mood = MAIN_MOODS[i] if args.mood == None else args.mood
	submood = ALL_MOOD_CLASSES[i+1][j] if args.submood == None else args.submood
	params = getClassifierParams(classifiers, mood, submood)
	mp = poem.MarkovPoem(args.path, args.ngramSize, mood, submood, params)
	generated_poem = mp.generate()
	print '\n' + generated_poem
	print 'mood=' + mood + ', submood=' + submood
	print 'enter score (out of 3 points for mood, submood, human-ness): '
	score = int(sys.stdin.readline())
	if score > 3:
	    continue
	f = open('scores.csv', 'a')
	f.write(mood + ',' + submood + ',' + str(score) + ',' + str(id) + '\n')
	f.close()
	f = open('generated_poems/' + str(id), 'w')
	f.write(generated_poem)
	f.close()
	j += 1
	if j == len(ALL_MOOD_CLASSES[i+1]):
	    j = 0
	    i += 1
	    if i == len(MAIN_MOODS):
		i = 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Poetry classifier")
    parser.add_argument("--examples", type=int, default=10000,
			help="Maximum number of examples to use" )
    parser.add_argument("--path", type=str, default="./tmp",
			help="Path to data")
    parser.add_argument("--iters", type=int, default="20",
			help="Number of iterations to run perceptron")
    subparsers = parser.add_subparsers()

    cparser = subparsers.add_parser("classify", help = "Run classifier")
    cparser.add_argument(
	"--debugscore", type=int, default=-1,
	help="Print debug info if poem score is <= debugscore")
    cparser.set_defaults(func=classify)

    gparser = subparsers.add_parser("generate", help = "Run generator")
    gparser.add_argument("--ngramSize", type=int, default=3,
			 help="Value of N to use in N-gram fluency constraints.")
    gparser.add_argument("--mood", type=str, default='Sad',
			 help="Mood of the poem to be generated.")
    gparser.add_argument("--submood", type=str, default='Heartbroken',
			 help="Mood of the poem to be generated.")
    gparser.add_argument("--rhyme", type=bool, default=False,
                         help="Whether the poem should rhyme.")
    gparser.add_argument("--meter", type=int, default=0,
                         help="Meter as defined by number of syllables per " +
                         "line, where 0 indicates no meter constraint.")
    gparser.set_defaults(func=generate)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
