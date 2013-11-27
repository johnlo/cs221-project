import copy
import math
import nltk
import os
import random
import solver
from collections import Counter
from collections import defaultdict
import string, 	re

def rhymes(w1, w2):
    l = min(len(w1), len(w2))
    w1 = w1[len(w1)-l:]
    w2 = w2[len(w2)-l:]
    hammingDist = sum(ch1 != ch2 for ch1, ch2 in zip(w1, w2))
    return len(w1) > 3 and len(w2) > 3 and w1 != w2 and (float(hammingDist)/l < .25)

syllable_count = Counter()

subsyl = ["cial", "tia", "cius", "cious", "gui", "ion", "iou",
                   "sia$", ".ely$"]

addsyl = ["ia", "riet", "dien", "iu", "io", "ii",
                   "[aeiouy]bl$", "mbl$",
                   "[aeiou]{3}",
                   "^mc", "ism$",
                   "(.)(?!\\1)([aeiouy])\\2l$",
                   "[^l]llien",
                   "^coad.", "^coag.", "^coal.", "^coax.",
                   "(.)(?!\\1)[gq]ua(.)(?!\\2)[aeiou]",
                   "dnt$"]

def syllables(wordList):
	syllables = 0
	for word in wordList:
		word = word.strip().lower()
		if syllable_count[word] > 0:
			syllables += syllable_count[word]
		else:
			if word[-1] == 'e': # silent e!
				word = word[:-1]

			count = 0
			prev_was_vowel = 0
			for c in word:
				is_vowel = c in ['a', 'e', 'i', 'o', 'u', 'y']
				if is_vowel and not prev_was_vowel:
					count += 1
				prev_was_vowel = is_vowel

			# Add & subtract syllables
			for r in addsyl:
				r = re.compile(r)
				if r.search(word):
					count += 1
			for r in subsyl:
				r = re.compile(r)
				if r.search(word):
					count -= 1

			syllable_count[word] = count
			syllables += count
	return syllables


class PoemCSP(solver.CSP):
	def __init__(self, path, ngramSize, mood):
		solver.CSP.__init__(self)

		self.ngramSize = ngramSize

		self.poems = {}
		num_lines = Counter()
		line_lengths = Counter()
	        self.pos = defaultdict(set)
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

def main():
    alg = solver.BacktrackingSearch()
    while True:
	csp = PoemCSP('./tmp', 3, 'Happy')
	csp.addVariables()
	csp.addNGramFluencyConstraints()
	alg.solve(csp, True, True, True)
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
    poem = poem.replace(
        ' ?', '?').replace(' !', '!').replace(' .', '.').replace(' ,', ',').replace(
        ' \n?', '?\n').replace('\n!', '!\n').replace('\n.', '.\n').replace(
        ' \n,', ',\n').replace('\n ', '\n').replace(
        ' ?', '?').replace(' !', '!').replace(' .', '.').replace(' ,', ',').replace(
        ' \n?', '?\n').replace('\n!', '!\n').replace('\n.', '.\n').replace(
        ' \n,', ',\n').replace('\n ', '\n')

    # rhyming code start
    pos = defaultdict(set)
    for p in csp.poems.values():
        for w, t in nltk.pos_tag(p.split()):
            pos[t].add(w)
    lines = poem.split('\n')
    lines = [line.split() for line in lines]
    for i in xrange(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        line1 = lines[i]
        line2 = lines[i+1]
        if len(line1) == 0 or len(line2) == 0:
            break
        last1 = line1[len(line1)-1]
        last2 = line2[len(line2)-1]
        pos1 = nltk.pos_tag([last1])[0][1]
        pos2 = nltk.pos_tag([last2])[0][1]
        done = False
        for otherword in pos[pos1]:
            if rhymes(last1, otherword):
                line2[len(line2)-1] = otherword
                done = True
                break
        if not done:
            for otherword in pos[pos2]:
                if rhymes(last2, otherword):
                    line1[len(line1)-1] = otherword
                    break
    lines = [' '.join(line) for line in lines]
    poem = '\n'.join(lines)
    print poem

if __name__ == "__main__":
    main()
