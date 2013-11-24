import os
from solver import *
from collections import Counter

class PoemCSP():
	def __init__(self, path, n, mood):
		self.poems = {}

		for poemdir in os.listdir(path):
			self.poems[poemdir] = open(os.path.join(path, poemdir, 'text')).read()

		self.words = list()
		nGramCounter = Counter()
		for poem in self.poems.values():
			split_poem = poem.split()
			self.words += split_poem
			lastNWords = []
			for word in split_poem:
				lastNWords.append(word)
				if len(lastNWords) == n:
					nGramCounter[tuple(lastNWords)] += 1
					lastNWords = lastNWords[1:]

		self.words = list(set(self.words))

		self.poemLength = 100 # todo

		self.subsets = []
		for i in xrange(n+1):
			self.subsets.append(self.getNGrams(self.poems.values(), i))
		print self.subsets

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
			self.addVariable('w' + i, self.words)

	def addLineLengthConstraints(self):
		pass

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

if __name__ == "__main__":
    main()
