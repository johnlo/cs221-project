import math
import os
import solver
from collections import Counter

MAX_POEMS = 1

class PoemCSP(solver.CSP):
    def __init__(self, path, ngramSize, mood):
        solver.CSP.__init__(self)

        self.ngramSize = ngramSize

        self.poems = {}
        num_lines = Counter()
        line_lengths = Counter()
        num_poems = 0
        for poemdir in os.listdir(path):
            moods = open(os.path.join(path, poemdir, 'mood')).read().split()
            if mood in moods:
                num_poems += 1
                if num_poems > MAX_POEMS: break
                self.poems[poemdir] = \
                    open(os.path.join(path, poemdir, 'text')).read().replace(
                    '?', ' ?').replace('!', ' !').replace('.', ' .').replace(',', ' ,')
                print self.poems[poemdir]
                lines = self.poems[poemdir].split('\n')
                num_lines[len(lines)] += 1
                for line in lines:
                    length = len(line.split())
                    if length > 0:
                        line_lengths[length] += 1
        self.lineLength = line_lengths.most_common(1)[0][0]
        self.numLines = num_lines.most_common(1)[0][0]
        self.numNgrams = self.lineLength * self.numLines

        self.ngramCounter = Counter(self.getNGrams(self.poems.values(), ngramSize))
        self.ngrams = list(self.ngramCounter.keys())
        print self.ngrams

        print len(self.ngrams), self.numNgrams

    def getNGrams(self, poems, n):
        if n == 0:
            return [()]
        retval = []
        for poem in poems:
            words = poem.split()
            lastNWords = []
            for word in words:
                lastNWords.append(word)
                if len(lastNWords) == n:
                    retval.append(tuple(lastNWords))
                    lastNWords = lastNWords[1:]
        return retval

    def addVariables(self):
        for i in xrange(self.numNgrams):
            self.add_variable('v' + str(i), self.ngrams)

    def addLineLengthConstraints(self):
        for i in xrange(0, self.poemLength, self.lineLength):
            if i > 0:
                self.add_unary_potential('w' + str(i), lambda x: x == '\n')

    def addNGramFluencyConstraints(self):
        for i in xrange(1, self.numNgrams):
            prevVar = 'v' + str(i-1)
            nextVar = 'v' + str(i)
            def consistency(v1, v2):
                for i in xrange(1, len(v1)):
                    if v1[i] != v2[i-1]:
                        return False
                return True
            self.add_binary_potential(prevVar, nextVar, consistency)
        for i in xrange(self.numNgrams):
            self.add_unary_potential('v' + str(i), lambda x: self.ngramCounter[x])
            self.add_unary_potential('v' + str(i), lambda x: 'by' not in x)
        self.add_unary_potential(
            'v0', lambda x: x[0] not in [',', '.', '?', '!'] and x[0][0].isupper())

    # john
    def addMoodFluencyConstraints(self):
        pass

    def addRhymeConstraints(self):
        pass

    def addMeterConstraints(self):
        pass


def main():
    csp = PoemCSP('./tmp', 3, 'Happy')  # add args
    csp.addVariables()  # words as a set, length of poem
    #csp.addLineLengthConstraints()  # line length
    csp.addNGramFluencyConstraints()  # counter from N-gram seen to count
    #csp.addMoodFluencyConstraints()  # mapping from mood, word to weight vector score
    #print csp.unaryPotentials
    #print csp.binaryPotentials
    alg = solver.BacktrackingSearch()
    alg.solve(csp, True, True, True)
    print alg.optimalAssignment
    poem = ""
    for i in xrange(csp.numNgrams):
        if i == 0:
            for w in alg.optimalAssignment['v' + str(i)]:
                poem += w + ' '
        else:
            ngram = alg.optimalAssignment['v' + str(i)]
            poem += ngram[len(ngram)-1] + ' '
    print poem

if __name__ == "__main__":
    main()
