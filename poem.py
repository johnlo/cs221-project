import os
from solver import *
from collections import Counter

class PoemCSP():
	def __init__(self, path, n, mood):
		self.poems = {}

		for poemdir in os.listdir(path):
			self.poems[poemdir] = open(os.path.join(path, poemdir, 'text')).read()

		self.words = set()
		nGramCounter = Counter()
		for poem in self.poems.values():
			lastNWords = []
			for word in poem.split():
				lastNWords.append(word)
				if len(lastNWords) == n:
					nGramCounter[tuple(lastNWords)] += 1
					lastNWords = lastNWords[1:]

		self.words = list(self.words)

		self.poemLength = 100 # todo 

		self.subsets = []
		for i in xrange(n):
			subset = []

		

			subsets.append(subset)



	def addVariables(self):
		for i in xrange(self.poemLength):
			self.addVariable('w' + i, self.words)

	def addLineLengthConstraints(self):
		pass

	# amber
	def addNGramFluencyConstraints(self):
		

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
	poemCSP = PoemCSP('./tmp', 2)

if __name__ == "__main__":
    main()

