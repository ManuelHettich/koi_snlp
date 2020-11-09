import string
word_frequency = dict()

# Exercise 1a)


def process_line(line):
    # Remove whitespace at the end and beginning of the sentence
    line = line.strip()
    # Remove punctuation (Source: https://stackoverflow.com/a/60725620)
    line = line.translate(str.maketrans('', '', string.punctuation))
    # Tokenize sentence into a list and add keywords for the beginning and the end of the sentence
    line_list = ["BOS"] + line.split() + ["EOS"]


def __main__():
    processed_corpus = []
    with open("corpus.txt", "rt") as infile:
        for line in infile:
            # Add processed sentence to the list of all sentences in memory
            processed_corpus += process_line(line)

    # Calculate frequency of each word and sequences of up to three words in the corpus
    size_of_processed_corpus = len(processed_corpus)
    for idx, word in enumerate(processed_corpus):
        # Unigram frequency
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1
        # Bigram frequency
        if idx <= size_of_processed_corpus - 2:
            if (word, processed_corpus[idx + 1]) in word_frequency:
                word_frequency[(word, processed_corpus[idx + 1])] += 1
            else:
                word_frequency[(word, processed_corpus[idx + 1])] = 1
        # Trigram frequency
        if idx <= size_of_processed_corpus - 3:
            if (word, processed_corpus[idx + 1], processed_corpus[idx + 2]) in word_frequency:
                word_frequency[(word, processed_corpus[idx + 1], processed_corpus[idx + 2])] += 1
            else:
                word_frequency[(word, processed_corpus[idx + 1], processed_corpus[idx + 2])] = 1


# Exercise 1b)
# Calculate unigram probabilities
def word_prob_unigram(word_i):
    # Calculate sum of all unigram word frequencies
    sum_of_frequencies = sum(word_frequency.values())
    return word_frequency[word_i] / sum_of_frequencies


# Calculate bigram probabilities
def word_prob_bigram(word_i, word_i1):
    if (word_i1, word_i) in word_frequency:
        return word_frequency[(word_i1, word_i)] / word_frequency[word_i1]
    else:
        return 0


# Calculate trigram probabilities
def word_prob_trigram(word_i, word_i1, word_i2):
    if (word_i2, word_i1, word_i) in word_frequency:
        return word_frequency[(word_i2, word_i1, word_i)] / word_frequency[(word_i2, word_i1)]
    else:
        return 0




# Example
# from collections import defaultdict
# import numpy as np
#
# dice3 = defaultdict(int)
#
# for observation in [0, 1, 2, 2, 1, 0, 2, 2, 1, 1, 0]:
#     dice3[observation] += 1
#
# vocabulary = list(dice3.keys())
# values = np.array(list(dice3.values()))
#
# print("vocabulary", vocabulary)
# print("observations", values)
#
# distribution = values / values.sum()
# print("values", distribution, distribution.sum())
#
# print("distribution-cumulative", distribution.cumsum())
#
# N = 10
# throws = []
#
# def sample(vocabulary, distribution):
#     dist_sum = distribution.cumsum()
#     randval = np.random.random()
#
#     for index, value in enumerate(dist_sum):
#         if value < randval:
#             continue
#         return index
#
#     return len(dist_sum) - 1
#
#
# throws = [sample(vocabulary, distribution) for _ in range(N)]
# print(throws)
#
# throws = [np.random.choice(vocabulary, p=distribution) for _ in range(N)]
# print(throws)
