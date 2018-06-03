
# coding: utf-8

#  # Twitter Sentiment Analysis dataset processor

# In[1]:


from matplotlib import pyplot as plt

import numpy as np

import math
import csv
from nltk import sent_tokenize, word_tokenize, re
import itertools
import string

# ## Utilities

# In[2]:


def get_sentences_per_label(dataset_filename):
    positive_statements = []
    negative_statements = []

    with open(dataset_filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)

        for sentence_counter, sentence in enumerate(csv_reader, 1):
            sentiment = sentence[1]
            if sentiment == '0':
                negative_statements.append(sentence_counter)
            elif sentiment == '1':
                positive_statements.append(sentence_counter)
            else:
                print('Unexpected sentiment value {}'.format(sentiment))
    
    return positive_statements, negative_statements


def filter_words_with_ampersand_at_and_hashtag(text):
    return re.sub(r"\w*[@&#]\w*", " ", text)


def filter_links(text):
    return re.sub(r"https?[^\s]*", " ", text)


def filter_rubbish(text):

    # add standard punctuation to translate table
    translate_table = dict((ord(char), None) for char in string.punctuation)

    # add unicode punctuation and strange characters
    for i in range(8208, 8286):
        translate_table[i] = None

    # add digits
    for i in range(0, 9):
        translate_table[ord(str(i))] = None

    translate_table[8211] = " "   # long dash
    translate_table[ord("-")] = " "
    return text.translate(translate_table)


def flatten(list_of_lists):
    return list(itertools.chain(*list_of_lists))


# ## Dataset splitting

# In[3]:


# CONSTANTS
TRAIN_DATASET_FACTOR = 0.5          # TODO
TEST_DATASET_FACTOR = 0.4           # TODO

MOBILE_DATASETS_NUMBER = 3          # TODO
MOBILE_DATASET_FACTOR = 0.3         # TODO


# UTILITIES
def clean_statement(statement):
    sentences = sent_tokenize(statement.lower().strip())
    words = [clean_sentence(s) for s in sentences]
    words = flatten(words)
    # words = [filter_links(w) for w in words]
    return " ".join(words)


def clean_sentence(sentence):
    sentence = filter_links(sentence)
    sentence = filter_words_with_ampersand_at_and_hashtag(sentence)
    return word_tokenize(filter_rubbish(sentence))

# In[4]:


def split_dataset():
    def create_dataset(available_statements, factor, csv_reader, output_filename, should_clean_statements, remove_chosen_statements): 
        chosen_positive = np.random.choice(available_statements[0], math.ceil(factor * len(available_statements[0])), replace=False)
        chosen_negative = np.random.choice(available_statements[1], math.ceil(factor * len(available_statements[1])), replace=False)
        chosen_shuffled = np.concatenate((chosen_positive, chosen_negative))
        np.random.shuffle(chosen_shuffled)
        
        with open(output_filename, 'w', encoding='utf-8', newline='') as output_file:
            csv_writer = csv.writer(output_file, delimiter=',')
            for statement_index in chosen_shuffled:
                statement = csv_reader[statement_index]
                csv_writer.writerow([statement[1], clean_statement(statement[3]) if should_clean_statements else statement[3]])
        print('File {} saved.'.format(output_filename))
        
        if remove_chosen_statements:
            list_difference = lambda first, second: list(set(first) - set(second))
            return list_difference(available_statements[0], chosen_positive.tolist()), list_difference(available_statements[1], chosen_negative.tolist())
        return available_statements[0], available_statements[1]
    
    
    positive_statements, negative_statements = get_sentences_per_label('dataset.csv')    
    with open('dataset.csv', 'r', encoding='utf-8') as dataset_file:
        csv_reader = list(csv.reader(dataset_file, delimiter=','))
        positive_statements, negative_statements = create_dataset((positive_statements, negative_statements), TRAIN_DATASET_FACTOR, csv_reader, 'train.csv', True, True)
        positive_statements, negative_statements = create_dataset((positive_statements, negative_statements), TEST_DATASET_FACTOR, csv_reader, 'test.csv', True, True)
        for mobile_dataset_index in range(1, MOBILE_DATASETS_NUMBER + 1):
            positive_statements, negative_statements = create_dataset((positive_statements, negative_statements), MOBILE_DATASET_FACTOR, csv_reader, 'mobile_{}.csv'.format(mobile_dataset_index), True, False)


split_dataset()


# ## Dataset analysis

# In[5]:


def analyse_dataset(dataset_filename, has_header=True):
    positive_statements, negative_statements = get_sentences_per_label(dataset_filename)
    total_statements = len(positive_statements) + len(negative_statements)
    
    # CLASSES DISTRIBUTION
    barchart = plt.bar([0, 1], [len(positive_statements) / total_statements, len(negative_statements) / total_statements], tick_label=['positive', 'negative'])
    
    for bar in barchart:
        height = bar.get_height()
        plt.text(bar.get_x() + 0.3, height + 0.005, "{:.5f}".format(height))
    plt.title('Classes distribution')
    plt.xlabel('Class')
    plt.ylabel('Participation in the dataset')
    plt.show()
    
    print('Positive statements: {} ({:.5f}%).'.format(len(positive_statements), len(positive_statements) / total_statements))
    print('Negative statements: {} ({:.5f}%).'.format(len(negative_statements), len(negative_statements) / total_statements))
    print('Total statements: {}.'.format(total_statements))
    
    # WORDS STATISTICS
    with open(dataset_filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        if has_header:
            next(csv_reader)
        
        words_count = []
        for sentence in csv_reader:
            words_count.append(len(sentence[3].split()))
    
    plt.hist(words_count, max(words_count))
    
    plt.title('Words count histogram')
    plt.xlabel('Words count')
    plt.ylabel('Frequency')
    plt.show()
    
    print('Maximal number of words: {}.'.format(max(words_count)))
    print('Average number of words: {:.2f}.'.format(sum(words_count) / len(words_count)))
    
    sorted_words_count = sorted(words_count)
    quartile_index = len(words_count) // 4
    print('First quartile: {}, second quartile (median): {}, third quartile: {}.'.format(sorted_words_count[quartile_index], sorted_words_count[2 * quartile_index], sorted_words_count[3 * quartile_index]))


# In[6]:


analyse_dataset('dataset.csv')

