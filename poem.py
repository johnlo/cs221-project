import copy
import math
import os
import random
import solver
from collections import Counter

class PoemCSP(solver.CSP):
    def __init__(self, path, ngramSize, mood):
        solver.CSP.__init__(self)

        self.ngramSize = ngramSize

        self.poems = {}
        num_lines = Counter()
        line_lengths = Counter()
        for poemdir in os.listdir(path):
            moods = open(os.path.join(path, poemdir, 'mood')).read().split()
            if mood in moods:
                self.poems[poemdir] = \
                    open(os.path.join(path, poemdir, 'text')).read().replace(
                    '?', ' ?').replace('!', ' !').replace('.', ' .').replace(',', ' ,')
                lines = self.poems[poemdir].split('\n')
                num_lines[len(lines)] += 1
                for line in lines:
                    length = len(line.split())
                    if length > 0:
                        line_lengths[length] += 1
        self.lineLength = line_lengths.most_common(5)[random.randint(0, 4)][0]
        self.numLines = num_lines.most_common(5)[random.randint(0, 4)][0] / 2
        self.numNgrams = self.lineLength * self.numLines

        self.ngrams = self.getNGrams(self.poems.values(), ngramSize)
        self.ngramCounter = Counter(self.ngrams)
        self.domains = []
        for _ in xrange(self.numLines):
            sample_index = int((len(self.ngrams)-self.lineLength*2) * random.random())
            self.domains.append(self.ngrams[sample_index:sample_index+self.lineLength*2])
        #print self.domains

        print self.lineLength, self.numLines, self.numNgrams

    def getNGrams(self, poems, n):
        if n == 0:
            return [()]
        retval = []
        for poem in poems:
            words = poem.split() + ['\n']
            lastNWords = []
            for word in words:
                lastNWords.append(word)
                if len(lastNWords) == n:
                    retval.append(tuple(lastNWords))
                    lastNWords = lastNWords[1:]
        return retval

    def addVariables(self):
        for i in xrange(self.numNgrams):
            self.add_variable('v' + str(i), self.domains[i/self.lineLength])

    def addNGramFluencyConstraints(self):
        for i in xrange(1, self.numNgrams):
            if (i-1)/self.lineLength == i/self.lineLength:
                def line_consistency(v1, v2):
                    for i in xrange(1, len(v1)):
                        if v1[i] != v2[i-1]:
                            return False
                        return True
                self.add_binary_potential('v' + str(i-1), 'v' + str(i), line_consistency)
        for i in xrange(self.numNgrams):
            self.add_unary_potential('v' + str(i), lambda x: self.ngramCounter[x])
            self.add_unary_potential('v' + str(i), lambda x: 'by' not in x)
            for j in xrange(i+1,self.numNgrams):
                self.add_binary_potential('v' + str(i), 'v' + str(j), lambda x, y: x != y)
        self.add_unary_potential('v0', lambda x: x[0] not in [',', '.', '?', '!'])
        def capitalized(x):
            if not x[0][0].isupper():
                return False
            for i in xrange(1, len(x)):
                if x[i][0].isupper():
                    return False
            for i in xrange(1, len(x[0])):
                if x[0][i].isupper():
                    return False
            return True
        self.add_unary_potential('v0', capitalized)
        self.add_unary_potential('v' + str(self.numNgrams-1),
                                 lambda x: x[self.ngramSize-1] in ['.', '?', '!'])

    def addRhymeConstraints(self):
        pass

    def addMeterConstraints(self):
        pass


def main():
    alg = solver.BacktrackingSearch()
    while True:
        csp = PoemCSP('./tmp', 3, 'Happy')  # add args
        csp.addVariables()  # words as a set, length of poem
        #csp.addLineLengthConstraints()  # line length
        csp.addNGramFluencyConstraints()  # counter from N-gram seen to count
        #csp.addMoodFluencyConstraints()  # mapping from mood, word to weight vector score
        #print csp.unaryPotentials
        #print csp.binaryPotentials
        alg.solve(csp, True, True, True)
        #print alg.optimalAssignment
        if len(alg.optimalAssignment) > 0:
            break
    poem = ""
    for i in xrange(csp.numNgrams):
        if i == 0:
            for w in alg.optimalAssignment['v' + str(i)]:
                poem += w + ' '
        else:
            ngram = alg.optimalAssignment['v' + str(i)]
            poem += ngram[len(ngram)-1] + ' '
    words = poem.split()
    length = 0
    i = 0
    while i < len(words):
        if length == csp.lineLength:
            words.insert(i, '\n')
            length = 0
        length += 1
        i += 1
    poem = ' '.join(words)
    print ''
    print (poem.replace(
            ' ?', '?').replace(' !', '!').replace(' .', '.').replace(' ,', ',').replace(
            ' \n?', '?\n').replace('\n!', '!\n').replace('\n.', '.\n').replace(
            ' \n,', ',\n').replace('\n ', '\n'))

if __name__ == "__main__":
    main()
