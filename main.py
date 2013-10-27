import classifier
import datetime
import operator
import util

MAIN_LABELS = ('Excited', 'Tender', 'Scared', 'Angry', 'Sad', 'Happy')

def run(args):
    l = MAIN_LABELS
    t, d = util.loadTrainAndDevExamples(args.path, l, args.examples) 
    c = classifier.PoetryClassifier(
        t, classifier.extractBigramFeatures, l, args.iters)
    util.evaluateClassifier(t, d, c)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Poetry classifier")
    parser.add_argument("--examples", type=int, default=10000,
                        help="Maximum number of examples to use" )
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser("run", help = "Run classifier")
    parser_run.add_argument("--iters", type=int, default="20",
                         help="Number of iterations to run perceptron") 
    parser_run.add_argument("--path", type=str, default="./tmp/mainonly",
                        help="Path to data")
    parser_run.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
