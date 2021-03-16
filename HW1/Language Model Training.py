import json
import re
import numpy as np
from math import log10
import matplotlib.pyplot as plt

spec_chars = {'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
              'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ه', 'ی', '0', '1', '2', '3', '4', '5', '6', '7', '8',
              '9', '.', '؟'}


class LanguageModel:
    n = 0
    smoothing = bool()
    training_data = []
    validation_data = []
    unigram = dict()
    bigram = dict()
    total_token_count = 0
    total_frequent_token_count = 0
    max_sentence_length = 0
    avg_sentence_length = 0

    def __init__(self, corpus_dir, n, smoothing=False):
        self.n = n
        self.smoothing = smoothing

        # Reading from file (training data)
        with open(corpus_dir + '\\train.json', encoding='utf-8') as file:
            self.training_data = json.load(file)
        # Reading from file (validation data)
        with open(corpus_dir + '\\valid.json', encoding='utf-8') as file:
            self.validation_data = json.load(file)

        # Removing all characters except Persian letters, numbers, {dot} and {؟}
        self.training_data = rm_spec_ch(self.training_data)
        self.validation_data = rm_spec_ch(self.validation_data)

        # Decomposing into tokens (training data)
        for i in range(len(self.training_data)):
            self.training_data[i] = str(self.training_data[i]).replace('.', ' . ')
            self.training_data[i] = str(self.training_data[i]).replace('؟', ' ؟ ')
            self.training_data[i] = re.split(' ', self.training_data[i])
            while '' in self.training_data[i]:
                self.training_data[i].remove('')
            for j in range(len(self.training_data[i])):
                if str.isnumeric(self.training_data[i][j]):
                    self.training_data[i][j] = 'N'
            self.total_token_count += len(self.training_data[i])

        # Decomposing into tokens (validation data)
        for i in range(len(self.validation_data)):
            self.validation_data[i] = str(self.validation_data[i]).replace('.', ' . ')
            self.validation_data[i] = str(self.validation_data[i]).replace('؟', ' ؟ ')
            self.validation_data[i] = re.split(' ', self.validation_data[i])
            while '' in self.validation_data[i]:
                self.validation_data[i].remove('')
            for j in range(len(self.validation_data[i])):
                if str.isnumeric(self.validation_data[i][j]):
                    self.validation_data[i][j] = 'N'

        # Unigram is a prerequisite of any N-gram
        self.train_unigram()

        if self.n == 2:
            self.train_bigram()

    def train_unigram(self):
        # Populate 1-gram dictionary
        for i in range(len(self.training_data)):
            for j in range(len(self.training_data[i])):
                if self.training_data[i][j] in self.unigram:
                    self.unigram[self.training_data[i][j]] += 1
                else:
                    # Start a new entry with 1 count since saw it for the first time.
                    self.unigram[self.training_data[i][j]] = 1

        # Turn into a list of (word, count) sorted by count from most to least.
        self.unigram = sorted(self.unigram.items(), key=lambda kv: kv[1], reverse=True)

        # Frequent words are discovered only through the training data
        top_thousands = []
        freq_word_dict = dict()
        file = open('most_frequent.txt', 'w', encoding='utf-8')
        for i in range(10000):
            file.write(self.unigram[i][0] + '\n')
            top_thousands.append(self.unigram[i][0])
            self.total_frequent_token_count += self.unigram[i][1]
            freq_word_dict[self.unigram[i][0]] = self.unigram[i][1]
        file.close()

        print('Total Number of Tokens:\t\t\t', str(self.total_token_count))
        print('Number of Unique Tokens:\t\t' + str(self.unigram.__len__()))
        print('Proportion of Frequent Words:\t% ', str((self.total_frequent_token_count / self.total_token_count) * 100))

        # Replacing least frequent words with "UNK"
        # Using a dictionary to access elements in O(1).
        for i in range(len(self.training_data)):
            for j in range(len(self.training_data[i])):
                if self.training_data[i][j] not in freq_word_dict:
                    self.training_data[i][j] = 'UNK'

        for i in range(len(self.validation_data)):
            for j in range(len(self.validation_data[i])):
                if self.validation_data[i][j] not in freq_word_dict:
                    self.validation_data[i][j] = 'UNK'

        # Decomposition & storage of sentences
        file = open('sentences.txt', 'w', encoding='utf-8')
        sentences = []
        sentence_count = 0
        for i in range(len(self.training_data)):
            text = ''
            for j in range(len(self.training_data[i])):
                if self.training_data[i][j] == '.':
                    text += ' . '
                    sentences.append(text)
                    sentence_count += 1
                    text = ''
                elif self.training_data[i][j] == '؟':
                    text += ' ؟ '
                    sentences.append(text)
                    sentence_count += 1
                    text = ''
                else:
                    text += self.training_data[i][j] + ' '
        for sentence in sentences:
            file.write(sentence + '\n\n')
            self.avg_sentence_length += len(sentence.split(' '))
        self.avg_sentence_length //= sentence_count
        file.close()

        print('Average Sentence Length:\t\t' + str(self.avg_sentence_length))

        # Plotting the Power Law Distribution
        x = []
        y = []
        for i in range(5000):
            x.append(log10(self.unigram[i][1]))
            y.append(log10(i + 1))
        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(x, y, '-b', label='Scatter Line')
        plt.plot(x, poly1d_fn(x), '--r', label='Regression Line')
        plt.xlabel('Log of Frequency')
        plt.ylabel('Log of Rank')
        plt.legend()
        plt.show()

    def train_bigram(self):
        # Populate 2-gram dictionary
        for i in range(len(self.training_data)):
            for j in range(len(self.training_data[i]) - 1):
                key = (self.training_data[i][j], self.training_data[i][j + 1])
                if key in self.bigram:
                    self.bigram[key] += 1
                else:
                    self.bigram[key] = 1

        # Turn into a list of (word, count) sorted by count from most to least.
        self.bigram = sorted(self.bigram.items(), key=lambda kv: kv[1], reverse=True)

    def prob(self, sentence):
        if self.n == 1:
            res = sentence.split(' ')
            word_dict = dict(self.unigram)
            pr = 1
            for word in res:
                pr *= word_dict[word] / self.total_token_count
            return pr
        elif self.n == 2:
            res = [word for word in zip(sentence.split(" ")[:-1], sentence.split(" ")[1:])]
            prob_dict = dict(self.bigram)
            word_dict = dict(self.unigram)
            probs = []
            if self.smoothing:
                for bg in res:
                    if bg in prob_dict:
                        probs.append((1 + prob_dict[bg]) / word_dict[bg[0]] + len(self.bigram))
                    else:
                        probs.append(1 / (word_dict[bg[0]] + len(self.bigram)))
                        prob_dict[bg] = 1
            else:
                for bg in res:
                    if bg in prob_dict.items():
                        probs.append(prob_dict[bg] / len(self.bigram))
                    else:
                        probs.append(0)
            if probs:
                per = 1
                for pr in probs:
                    per *= pr
                return per
            else:
                return 0

    def generate(self, starting_word, limit=avg_sentence_length):
        sentence = ''
        # Using Unigram
        if self.n == 1:
            for i in range(limit):
                sentence += self.unigram[i][0] + ' '

        # Using Bigram
        elif self.n == 2:
            sentence = starting_word + ' '
            cur_word = starting_word
            for i in range(1, limit):
                pot_follow = ''
                max_hit = 0
                for j in range(len(self.bigram)):
                    if self.bigram[j][0][0] == cur_word and self.bigram[j][1] > max_hit:
                        max_hit = self.bigram[j][1]
                        pot_follow = self.bigram[j][0][1]
                sentence += pot_follow + ' '
                cur_word = pot_follow

        return sentence

    def evaluate(self, limit):
        total_sentence_count = 0
        total_wer_sum = 0
        file = open('valid_sentences.txt', 'w', encoding='utf-8')
        for i in range(len(self.validation_data)):
            text = ''
            for j in range(len(self.validation_data[i])):
                text += self.validation_data[i][j] + ' '
            file.write(text + '.\n\n')
            total_sentence_count += 1
            valid_sentence = text.split(' ')
            generated_sentence = self.generate(valid_sentence[0], limit)
            total_wer_sum += wer(valid_sentence, generated_sentence)

        avg_wer = total_wer_sum / total_sentence_count
        return avg_wer


# Removing special characters
def rm_spec_ch(data):
    for i in range(len(data)):
        text = ''
        for j in range(len(data[i])):
            if data[i][j] not in spec_chars:
                text += ' '
            else:
                text += data[i][j]
        data[i] = text
    return data


def wer(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """

    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # Computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)] / len(r) * 100


lm = LanguageModel('E:\\NLP\\HW01', 2, True)

# Sentence generation
print('Generated Sentence:\n' + lm.generate('ترامپ', 10))

# Sentence Probability
print('Sentence Occurrence Probability:\n' + str(lm.prob('سلام خبر مدارس')))

# Average word-error rate
print('Average WER:\n%' + str(lm.evaluate(10)))
