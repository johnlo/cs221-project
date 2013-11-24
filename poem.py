import os
import solver
from collections import Counter

class PoemCSP(solver.CSP):
    def __init__(self, path, ngramSize, mood):
        solver.CSP.__init__(self)
        self.poems = {}
        poem_lengths = Counter()
        line_lengths = Counter()
        for poemdir in os.listdir(path):
            self.poems[poemdir] = \
                open(os.path.join(path, poemdir, 'text')).read().replace(
                '?', ' ?').replace('!', ' !').replace('.', ' .')
            moods = open(os.path.join(path, poemdir, 'mood')).read().split()
            if mood in moods:
                poem_lengths[len(self.poems[poemdir].split())] += 1
                for line in self.poems[poemdir].split('\n'):
                    length = len(line.split())
                    if length > 0:
                        line_lengths[length] += 1
        self.poemLength = poem_lengths.most_common(1)[0][0]
        self.lineLength = line_lengths.most_common(1)[0][0]

        self.words = list()
        self.words.append('\n')  # newline should be in the domain
        nGramCounter = Counter()
        for poem in self.poems.values():
            split_poem = poem.split()
            self.words += split_poem
            lastNWords = []
            for word in split_poem:
                lastNWords.append(word)
                if len(lastNWords) == ngramSize:
                    nGramCounter[tuple(lastNWords)] += 1
                    lastNWords = lastNWords[1:]
        self.words = list(set(self.words))

        self.subsets = []
        for i in xrange(ngramSize + 1):
            self.subsets.append(self.getNGrams(self.poems.values(), i))

    def getNGrams(self, poems, n):
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
        for i in xrange(self.poemLength):
            self.add_variable('w' + str(i), self.words)

    def addLineLengthConstraints(self):
        for i in xrange(0, self.poemLength, self.lineLength):
            if i > 0:
                self.add_unary_potential('w' + str(i), lambda x: x == '\n')

    # amber
    def addNGramFluencyConstraints(self):
        pass

    def addNGramFactorFromNTuple(self, vars): # vars is tuple of n var names
        for var in vars:
            self.addVariable(''.join(vars) + var, )

    # john
    def addMoodFluencyConstraints(self):
        pass

    def addRhymeConstraints(self):
        pass

    def addMeterConstraints(self):
        pass


def main():
    poemCSP = PoemCSP('./tmp', 5, 'Happy')
    poemCSP.addVariables()
    poemCSP.addLineLengthConstraints()

if __name__ == "__main__":
  main()
