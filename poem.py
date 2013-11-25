import os
import solver
from collections import Counter

MAX_POEMS = 1

class PoemCSP(solver.CSP):
    def __init__(self, path, ngramSize, mood):
        solver.CSP.__init__(self)

        self.ngramSize = ngramSize

        self.poems = {}
        poem_lengths = Counter()
        line_lengths = Counter()
        num_poems = 0
        for poemdir in os.listdir(path):
            moods = open(os.path.join(path, poemdir, 'mood')).read().split()
            if mood in moods:
                num_poems += 1
                if num_poems > MAX_POEMS: break
                self.poems[poemdir] = \
                    open(os.path.join(path, poemdir, 'text')).read().replace(
                    '?', ' ?').replace('!', ' !').replace('.', ' .')
                poem_lengths[len(self.poems[poemdir].split())] += 1
                for line in self.poems[poemdir].split('\n'):
                    length = len(line.split())
                    if length > 0:
                        line_lengths[length] += 1
        self.lineLength = line_lengths.most_common(1)[0][0]
        self.poemLength = 3 * self.lineLength #poem_lengths.most_common(1)[0][0]

        self.words = list()
        self.words.append('\n')  # newline should be in the domain

        for poem in self.poems.values():
            self.words += poem.split()
        self.words = list(set(self.words))

        self.ngramCounter = Counter(self.getNGrams(self.poems.values(), ngramSize))

        self.subsets = []
        for i in xrange(ngramSize + 1):
            self.subsets.append(self.getNGrams(self.poems.values(), i))

        self.cached_ngram_domains = {}
        print len(self.words), self.poemLength, self.lineLength


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
        for i in xrange(self.poemLength):
            self.add_variable('w' + str(i), self.words)

    def addLineLengthConstraints(self):
        for i in xrange(0, self.poemLength, self.lineLength):
            if i > 0:
                self.add_unary_potential('w' + str(i), lambda x: x == '\n')

    def addNGramFluencyConstraints(self):
        lastNVars = []
        for i in xrange(self.poemLength):
            var = 'w' + str(i)
            lastNVars.append(var)
            if len(lastNVars) == self.ngramSize:
                self.addNGramFactorFromNTuple(tuple(lastNVars))
                lastNVars = lastNVars[1:]

    def addNGramFactorFromNTuple(self, varNames):
        print varNames
        for i in xrange(len(varNames)):
            newVarName = ('ngram', varNames, i)
            self.add_variable(newVarName, self.getNGramDomain(i))
            def observation(v, a):
                before, after = a
                retval = len(after) > 0 and v == after[len(after)-1]
                #if retval: print "observation: ", v, a
                return retval
            self.add_binary_potential(varNames[i], newVarName, observation)
            if i > 0:
                def transfer(a1, a2):
                    return a1[1] == a2[0]
                prevVarName = ('ngram', varNames, i-1)
                self.add_binary_potential(prevVarName, newVarName, transfer)
        def score(a):
            before, after = a
            retval = self.ngramCounter[after]
            #if retval: print "score: ", a, retval
            return retval
        self.add_unary_potential(('ngram', varNames, len(varNames)-1), score)

    def getNGramDomain(self, i):
        if i not in self.cached_ngram_domains:
            self.cached_ngram_domains[i] = []
            for l1 in (self.subsets[i] if i > 0 else [()]):
                for l2 in self.subsets[i+1]:
                    def consistent(a1, a2):
                        for j in xrange(len(a1)):
                            if a1[j] != a2[j]: return False
                        return True
                    if consistent(l1, l2):
                        self.cached_ngram_domains[i].append((l1, l2))
        return self.cached_ngram_domains[i]

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
    for i in xrange(csp.poemLength):
        poem += alg.optimalAssignment['w' + str(i)] + ' '
    print poem

if __name__ == "__main__":
    main()
