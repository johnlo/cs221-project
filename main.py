import datetime
import sys, operator
import util, submission

# Main entry point to test your spam classifier.

TRAIN_PATH_TOPICS = 'tmp2'

def evaluateClassifier(trainExamples, devExamples, classifier):
    util.printConfusionMatrix(
        util.computeConfusionMatrix(trainExamples, classifier))
    trainErrorRate = util.computeErrorRate(trainExamples, classifier) 
    print 'trainErrorRate: %f' % trainErrorRate
    util.printConfusionMatrix(util.computeConfusionMatrix(
        devExamples, classifier))
    devErrorRate = util.computeErrorRate(devExamples, classifier) 
    print 'devErrorRate: %f' % devErrorRate

def part3(args):
    print "Part 3 Topic Classification"
    examples = util.loadExamples(TRAIN_PATH_TOPICS)[:args.examples]
    labels = util.LABELS_TOPICS
    trainExamples, devExamples = util.holdoutExamples(examples)

    start = datetime.datetime.now() # Get your starting time
    classifiers = submission.learnOneVsAllClassifiers(
        trainExamples, submission.extractBigramFeatures, labels, args.iters)
    classifier = submission.OneVsAllClassifier(labels, classifiers)
    print "Time taken: ", (datetime.datetime.now() - start).seconds
    
    evaluateClassifier(trainExamples, devExamples, classifier)

def main():
    import argparse
    parser = argparse.ArgumentParser( description='Spam classifier' )
    parser.add_argument('--examples', type=int, default=10000,
                        help="Maximum number of examples to use" )
    subparsers = parser.add_subparsers()

    # Part 3
    parser3 = subparsers.add_parser('part3a', help = "Part 3")
    parser3.add_argument('--iters', type=int, default="20",
                         help="Number of iterations to run perceptron") 
    parser3.set_defaults(func=part3)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
