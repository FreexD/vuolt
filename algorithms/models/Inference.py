# coding: utf-8

# # Inference on the LSTM network

# In[ ]:


import numpy as np
import tensorflow as tf
import os
import csv
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

# In[2]:


# CONSTANTS
TESTING_SAMPLES = 19734
MAX_WORDS_PER_SENTENCE = 64
BATCH_SIZE = 32
LSTM_UNITS = 128


# ## Utilities

# In[3]:


def save_numpy_array(filename, array):
    np.save(filename, array)

def load_numpy_array(filename):
    return np.load(filename)


# ## Word embeddigns - GloVe's `Word2Vec`

# In[4]:


words_list = [word.decode('UTF-8') for word in load_numpy_array('wordsList.npy')]
word_vectors = load_numpy_array('wordVectors.npy')


# ## LSTM model

# In[5]:


tf.reset_default_graph()

# Placeholders for training input data.
batch_placeholder = tf.placeholder(tf.int32, (BATCH_SIZE, MAX_WORDS_PER_SENTENCE))
labels_placeholder = tf.placeholder(tf.float32, (BATCH_SIZE, 2))

# Converting the batch input to a 3D tensor of shape (BATCH_SIZE, MAX_WORDS_PER_SENTENCE, Word2Vec vector length).
batch_input_tensor = tf.Variable(tf.zeros((BATCH_SIZE, MAX_WORDS_PER_SENTENCE, 50), dtype=tf.float32))
batch_input_tensor = tf.nn.embedding_lookup(word_vectors, batch_placeholder)

# Creating LSTM cells inside a Dropout layer.
lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_UNITS)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)

# Combining the input with the LSTM layer.
output_tensor, _ = tf.nn.dynamic_rnn(lstm_cell, batch_input_tensor, dtype=tf.float32)

# Creating the output layer.
weight_matrix = tf.Variable(tf.truncated_normal((LSTM_UNITS, 2)))
bias_neurons = tf.Variable(tf.constant(0.1, shape=(2,)))
# Transposing output_tensor to shape (MAX_WORDS_PER_SENTENCE, BATCH_SIZE, LSTM_UNITS)
output_tensor = tf.transpose(output_tensor, [1, 0, 2])
# Creating the activation tensor of shape (BATCH_SIZE, LSTM_UNITS), to be multiplied by the weight_matrix.
output_activation_tensor = tf.gather(output_tensor, int(output_tensor.get_shape()[0]) - 1)
prediction = (tf.matmul(output_activation_tensor, weight_matrix) + bias_neurons)

# Training evaluation metrics.
is_prediction_correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_placeholder, 1))
prediction_accuracy = tf.reduce_mean(tf.cast(is_prediction_correct, tf.float32))


# In[6]:


session = tf.InteractiveSession()
checkpoint_saver = tf.train.Saver()
checkpoint_saver.restore(session, tf.train.latest_checkpoint('models'))


# ## Inference

# In[7]:


# UTILITIES
def build_identities_matrix(dataset_filename, identities_matrix_filename, labels_matrix_filename):
    identities = np.zeros((TESTING_SAMPLES, MAX_WORDS_PER_SENTENCE), dtype='int32')
    labels = np.zeros((TESTING_SAMPLES, 2), dtype='float32')

    with open(dataset_filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for sentence_index, sentence in enumerate(csv_reader, 0):
            splitted_sentence = sentence[1].split()
            for word_index, word in enumerate(splitted_sentence, 0):
                try:
                    identities[sentence_index, word_index] = words_list.index(word)
                except ValueError:
                    identities[sentence_index, word_index] = len(words_list) - 1
            if sentence[0] == '0':
                labels[sentence_index, :] = [0, 1]
            else:
                labels[sentence_index, :] = [1, 0]
            
            if (sentence_index + 1) % (TESTING_SAMPLES // 10) == 0:
                print('#')
            elif (sentence_index + 1) % (TESTING_SAMPLES // 500) == 0:
                print('#', end='')
    
    save_numpy_array(identities_matrix_filename, identities)
    save_numpy_array(labels_matrix_filename, labels)


def get_batches(identities, labels):
    identities_batches = [identities[batch_start_index: batch_start_index + BATCH_SIZE, :] for batch_start_index in
                          range(0, TESTING_SAMPLES, BATCH_SIZE)]
    labels_batches = [labels[batch_start_index: batch_start_index + BATCH_SIZE, :] for batch_start_index in
                      range(0, TESTING_SAMPLES, BATCH_SIZE)]
    if TESTING_SAMPLES % BATCH_SIZE != 0:
        identities_batches[-1] = np.pad(identities_batches[-1],
                                        [(0, BATCH_SIZE - identities_batches[-1].shape[0]), (0, 0)], mode='constant',
                                        constant_values=0)
        labels_batches[-1] = np.pad(labels_batches[-1], [(0, BATCH_SIZE - labels_batches[-1].shape[0]), (0, 0)],
                                    mode='constant', constant_values=0)
    return identities_batches, labels_batches


# In[9]:


def transform(result):
    if np.array_equal(result, [0, 1]):
        return 0
    else:
        return 1


def batch_inference():
    dataset = "mobile_1.csv"
    dataset_name, ext = os.path.splitext(dataset)
    # build_identities_matrix(dataset, 'idsMatrix_%s' % dataset_name, 'labelsMatrix_%s' % dataset_name)
    identities_matrix = load_numpy_array('idsMatrix_%s.npy' % dataset_name)
    labels_matrix = load_numpy_array('labelsMatrix_%s.npy' % dataset_name)

    incorrect_predictions = []
    identities_batches, labels_batches = get_batches(identities_matrix, labels_matrix)
    labels = []
    for batch in labels_batches:
        labels.extend([transform(el) for el in batch])
    results = []
    for batch_number, (identities_batch, labels_batch) in enumerate(zip(identities_batches, labels_batches), 1):
        prediction_results = session.run(is_prediction_correct,
                                         {batch_placeholder: identities_batch, labels_placeholder: labels_batch})
        for result_index, result in enumerate(prediction_results, 0):
            results.append(transform(labels_batch[result_index]) if result else not transform(labels_batch[result_index]))
            if result == False:
                incorrect_predictions.append((batch_number - 1) * BATCH_SIZE + result_index)
        if batch_number % (len(identities_batches) // 50) == 0:
            print('#', end='')

    precision, recall, tresholds = precision_recall_curve(labels, results)
    average_precision = average_precision_score(labels, results)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.savefig("pr_curve_lstm%s.png" % dataset_name)
    print(dataset_name)
    save_incorrect_results(dataset, "errors_lstm_%s.csv" % dataset_name, incorrect_predictions)
    print('\nIncorrect predictions: {} ({:.2f}%)'.format(len(incorrect_predictions),
                                                         (len(incorrect_predictions) / TESTING_SAMPLES) * 100))


def save_incorrect_results(dataset_file, output_file, incorret_predictions):
    with open(dataset_file, 'r') as dataset_csv_file:
        with open(output_file, 'w') as error_file:
            csv_reader = csv.reader(dataset_csv_file, delimiter=',')
            csv_writer = csv.writer(error_file)
            for i, row in enumerate(csv_reader):
                if i in incorret_predictions:
                    csv_writer.writerow(row)


def simple_inference(sentence):
    network_input = np.zeros((BATCH_SIZE, MAX_WORDS_PER_SENTENCE))
    splitted_sentence = sentence.split()    
    for word_index, word in enumerate(splitted_sentence, 0):
        try:
            network_input[0, word_index] = words_list.index(word)
        except ValueError:
            network_input[0, word_index] = len(words_list) - 1

    prediction_result = session.run(tf.nn.softmax(prediction), {batch_placeholder: network_input})[0]
    print('Predicted sentiment: {:.2f}% positive, {:.2f}% negative'.format(prediction_result[0] * 100, prediction_result[1] * 100))

batch_inference()
simple_inference('I am rather happy than sad.')
simple_inference('I am rather sad than happy.')
