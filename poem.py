import copy
import math
import nltk
import os
import random
from collections import Counter
from collections import defaultdict
import string, re
from nltk.corpus import cmudict
from collections import Sequence

def normalize(d):
    factor = float(sum(d.values()))
    for k in d:
	d[k] /= factor
    return d

def multinomial( pdf ):
    """
    Draw from a multinomial distribution
    @param pdf list double - probability of choosing value i
    OR
    @param pdf Counter - probability of choosing value i
    @return int - a sample from a multinomial distribution with above pdf

    Example:
      multinomial([0.4, 0.3, 0.2, 0.1]) will return 0 with 40%
      probability and 3 with 10% probability.
      multinomial({'a':0.4, 'b':0.3, 'c':0.2, 'd':0.1}) will return 'a' with 40%

    """
    if isinstance(pdf, Sequence):
	assert( abs( sum(pdf) - 1. ) < 1e-4 )

	cdf = [0.] * len(pdf)
	for i in xrange(len(pdf)):
	    cdf[i] = cdf[i-1] + pdf[i] # Being clever in using cdf[-1] = 0.
	rnd = random.random()
	for i in xrange(len(cdf)):
	    if rnd < cdf[i]:
		return i
	else:
	    return len(cdf) - 1
    elif isinstance(pdf, dict):
	names, pdf = zip(*pdf.iteritems())
	return names[ multinomial( pdf ) ]
    else:
	raise TypeError

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

def stressedSyllables(word):
	pronunciations = cmu.dict()[word]
	stresses = []
	for pronunciation in pronunciations: # there are often multiple possible pronunciations
		stresses.append([i[-1] for i in pronunciation if i[-1].isdigit()])
	return stresses

class MarkovPoem():
    def __init__(self, path, ngram_size, mood, submood, params):
	self.ngram_dict = defaultdict(list)
	self.poems = dict()
	self.line_lengths = Counter()
	self.num_lines = Counter()
	self.ngram_size = ngram_size
	self.params = params
        self.mood = mood
        self.submood = submood
        self.mood_seeds = defaultdict(set)
	for poemdir in os.listdir(path):
            moods = open(os.path.join(path, poemdir, 'mood')).read().split()
            self.poems[poemdir] = \
                open(os.path.join(path, poemdir, 'text')).read().replace('?', ' ?').replace('!', ' !').replace('.', ' .').replace(',', ' ,')
            lines = self.poems[poemdir].split('\n')
            lines = lines[1:]
            self.num_lines[len(lines)] += 1
            for line in lines:
                words = line.split()
                if len(words):
                    self.line_lengths[len(words)] += 1
            self.poems[poemdir] = '\n'.join(lines)
            words = self.poems[poemdir].split()
            lastN = []
            for i in xrange(len(words)+self.ngram_size):
                lastN.append(words[i % len(words)])
                if len(lastN) == self.ngram_size:
                    self.ngram_dict[tuple(lastN)].append(words[(i+1) % len(words)])
                    for mood in moods:
                        self.mood_seeds[mood].add(tuple(lastN))
                    lastN = lastN[1:]

    def chooseNext(self, poem, curr_line_number, curr_line):
	if self.poem:
	    return
	if curr_line_number == self.num_lines:
	    self.poem = poem
	    return
	if self.meter:
	    if syllables(curr_line) > self.meter:
		return
	    if syllables(curr_line) == self.meter:
		if self.rhyme and not rhymes(poem[len(poem)-1], poem[len(poem)-len(curr_line)-1]):
		    return
		self.chooseNext(poem, curr_line_number+1, [])
	else:
	    if len(curr_line) > self.line_length:
		return
	    if len(curr_line) == self.line_length:
		if self.rhyme and curr_line_number % 2 == 1 and not rhymes(poem[len(poem)-1], poem[len(poem)-len(curr_line)-1]):
		    return
		self.chooseNext(poem, curr_line_number+1, [])
	cur_ngram = tuple(poem[len(poem)-self.ngram_size:])
	tried = []
	next_word_choices = {w: self.params[w] for w in set(self.ngram_dict[cur_ngram]) if w not in tried}
	while not self.poem and len(next_word_choices) > 0:
	    scaling_factor = max(next_word_choices.values()) - min(next_word_choices.values())
	    for k in next_word_choices:
		next_word_choices[k] += scaling_factor
	    if sum(next_word_choices.values()) == 0:
		next_word = random.choice(next_word_choices.keys())
	    else:
                max_value = max(next_word_choices.values())
                for k in next_word_choices:
                    next_word_choices[k] += max_value
		normalize(next_word_choices)
		next_word = multinomial(Counter(next_word_choices))
	    tried.append(next_word)
	    self.chooseNext(poem + [next_word], curr_line_number, curr_line + [next_word])
	    next_word_choices = {w: self.params[w] for w in set(self.ngram_dict[cur_ngram]) if w not in tried}

    def generate(self, rhyme=False, meter=5):
	self.rhyme = rhyme
	self.meter = meter
	self.line_length = random.choice(self.line_lengths.most_common(5))[0]
	self.num_lines = random.choice(self.num_lines.most_common(5))[0]

	self.poem = None
	while self.poem is None:
	    seed = [x for x in random.choice([y for y in self.mood_seeds[self.mood] if y[0][0].isupper()])]
	    self.chooseNext(seed, 0, seed)

	curr_line = []
	retval = ''
	for i in xrange(len(self.poem)):
	    retval += (self.poem[i] + ' ')
	    curr_line.append(self.poem[i])
	    if not self.meter:
		if i > 0 and i % self.line_length == 0:
		    retval += '\n'
	    else:
		if syllables(curr_line) == self.meter:
		    retval += '\n'
		    curr_line = []
	return retval.replace('\n?', '?').replace('\n!', '!').replace('\n.', '.\n').replace('\n,', ',\n').replace('\n  ', '\n') \
		.replace(' ?', '?').replace(' !', '!').replace(' .', '.').replace(' ,', ',').replace('\n ', ' \n')

if __name__ == "__main__":
    pass
