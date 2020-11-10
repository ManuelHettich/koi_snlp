import string
import numpy as np

word_frequency_unigram = dict()
word_frequency_bigram = dict()
word_frequency_trigram = dict()

# Exercise 1a)
def main():
    def process_line(line):
        # Remove whitespace at the end and beginning of the sentence
        line = line.strip()
        # Make sentence lowercase
        line = line.lower()
        # Remove punctuation (Source: https://stackoverflow.com/a/60725620)
        line = line.translate(str.maketrans('', '', string.punctuation))
        # Tokenize sentence into a list and add keywords for the beginning and the end of the sentence
        return ["[SOS]"] + line.split() + ["[EOS]"]

    processed_corpus = []
    with open("corpus.txt", "rt") as infile:
        for line in infile:
            # Add processed sentence to the list of all sentences in memory
            processed_corpus += process_line(line)

    # Calculate frequency of each word and sequences of up to three words in the corpus
    size_of_processed_corpus = len(processed_corpus)
    for idx, word in enumerate(processed_corpus):
        # Unigram frequency
        if word in word_frequency_unigram:
            word_frequency_unigram[word] += 1
        else:
            word_frequency_unigram[word] = 1
        # Bigram frequency
        if idx <= size_of_processed_corpus - 2:
            if (word, processed_corpus[idx + 1]) in word_frequency_bigram:
                word_frequency_bigram[(word, processed_corpus[idx + 1])] += 1
            else:
                word_frequency_bigram[(word, processed_corpus[idx + 1])] = 1
        # Trigram frequency
        if idx <= size_of_processed_corpus - 3:
            if (word, processed_corpus[idx + 1], processed_corpus[idx + 2]) in word_frequency_trigram:
                word_frequency_trigram[(word, processed_corpus[idx + 1], processed_corpus[idx + 2])] += 1
            else:
                word_frequency_trigram[(word, processed_corpus[idx + 1], processed_corpus[idx + 2])] = 1

if __name__ == "__main__":
    main()


# Exercise 1b)
# Calculate unigram probabilities
def word_prob_unigram(word_i):
    # Calculate sum of all unigram word frequencies
    sum_of_frequencies = sum(word_frequency_unigram.values())
    return word_frequency_unigram[word_i] / sum_of_frequencies


# Calculate bigram probabilities
def word_prob_bigram(word_i, word_i1):
    if (word_i1, word_i) in word_frequency_bigram:
        return word_frequency_bigram[(word_i1, word_i)] / word_frequency_unigram[word_i1]
    else:
        return 0


# Calculate trigram probabilities
def word_prob_trigram(word_i, word_i1, word_i2):
    if (word_i2, word_i1, word_i) in word_frequency_trigram:
        return word_frequency_trigram[(word_i2, word_i1, word_i)] / word_frequency_bigram[(word_i2, word_i1)]
    else:
        return 0


# Exercise 2a)
def sample_unigram(all_words_prob_unigram):
    return np.random.choice([*word_frequency_unigram],
                            p=all_words_prob_unigram)


def sample_bigram(word_i1):
    matching_bigrams = [(w_i1, w_i)
                        for (w_i1, w_i)
                        in word_frequency_bigram
                        if w_i1 == word_i1]
    return np.random.choice([w_i for (_, w_i) in matching_bigrams],
                            p=[word_prob_bigram(w_i, word_i1)
                               for (_, w_i)
                               in matching_bigrams])


def sample_trigram(word_i2, word_i1):
    matching_trigrams = [(w_i2, w_i1, w_i)
                         for (w_i2, w_i1, w_i)
                         in word_frequency_trigram
                         if w_i2 == word_i2 and w_i1 == word_i1]
    return np.random.choice([w_i for (_, _, w_i) in matching_trigrams],
                            p=[word_prob_trigram(w_i, word_i1, word_i2)
                               for (_, _, w_i)
                               in matching_trigrams])


# Exercise 2b)
def gen_sentence_unigram():
    all_words_prob_unigram = [word_prob_unigram(word) for word in word_frequency_unigram.keys()]

    sentence = list()
    while True:
        word = sample_unigram(all_words_prob_unigram)
        if word == '[EOS]':
            del sentence[0]
            return ' '.join(sentence) + '.'
        else:
            sentence.append(word)


def gen_sentence_bigram():
    i = 1
    sentence = ['[SOS]']
    while True:
        word = sample_bigram(sentence[i - 1])
        if word == '[EOS]':
            del sentence[0]
            return ' '.join(sentence) + '.'
        else:
            sentence.append(word)
            i += 1


def gen_sentence_trigram():
    i = 1
    sentence = ['[SOS]']
    while True:
        if i >= 2:
            word = sample_trigram(sentence[i - 2], sentence[i - 1])
        if i == 1:
            word = sample_bigram(sentence[i - 1])
        if word == '[EOS]':
            del sentence[0]
            return ' '.join(sentence) + '.'
        else:
            sentence.append(word)
            i += 1
