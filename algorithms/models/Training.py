
# coding: utf-8

# # Sentiment analysis using LSTM networks and TensorFlow
# ### Based on the following [O'Reilly Tutorial](https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow).

# In[1]:


import numpy as np
import tensorflow as tf

import csv
import datetime


# In[2]:


# CONSTANTS
TRAINING_SAMPLES = 789314
MAX_WORDS_PER_SENTENCE = 64


# ## Utilities

# In[3]:


def save_numpy_array(filename, array):
    np.save(filename, array)

def load_numpy_array(filename):
    return np.load(filename)


# ## Word embeddigns - GloVe's `Word2Vec`

# In[4]:


# The list of 400'000 words defined within this Word2Vec model
words_list = [word.decode('UTF-8') for word in load_numpy_array('wordsList.npy')]
# The i-th vector represents the i-th word from the words_list, each vector is composed of 50 dimensions
word_vectors = load_numpy_array('wordVectors.npy')


# ## Identities matrix

# In[5]:


# Each row of the indetities matrix represents single sentence, while columns contain IDs of its consecutive words obtained
# from the words_list.
def build_identities_matrix(training_dataset_filename, identities_matrix_filename, labels_matrix_filename):
    identities = np.zeros((TRAINING_SAMPLES, MAX_WORDS_PER_SENTENCE), dtype='int32')
    training_labels = np.zeros((TRAINING_SAMPLES, 2), dtype='float32')

    with open(training_dataset_filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for sentence_index, sentence in enumerate(csv_reader, 0):
            splitted_sentence = sentence[1].split()
            for word_index, word in enumerate(splitted_sentence, 0):
                try:
                    identities[sentence_index, word_index] = words_list.index(word)
                except ValueError:
                    # If this word is not defined within the Word2Vec model, we put the last vector representing unknown words.
                    identities[sentence_index, word_index] = len(words_list) - 1
            if sentence[0] == '0':
                training_labels[sentence_index, :] = [0, 1]
            else:
                training_labels[sentence_index, :] = [1, 0]
            
            if (sentence_index + 1) % (TRAINING_SAMPLES // 10) == 0:
                print('#')
            elif (sentence_index + 1) % (TRAINING_SAMPLES // 500) == 0:
                print('#', end='')
    
    save_numpy_array(identities_matrix_filename, identities)
    save_numpy_array(labels_matrix_filename, training_labels)


# build_identities_matrix('train.csv', 'idsMatrix_train', 'labelsMatrix_train')
identities_matrix = load_numpy_array('idsMatrix_train.npy')
training_labels_matrix = load_numpy_array('labelsMatrix_train.npy')


# ## LSTM model

# In[6]:


# CONSTANTS
LSTM_UNITS = 64
BATCH_SIZE = 32


# In[7]:


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


# In[24]:


# Training evaluation metrics.
is_prediction_correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_placeholder, 1))
prediction_accuracy = tf.reduce_mean(tf.cast(is_prediction_correct, tf.float32))

# Putting a binary softmax classifier on the final prediction value, with standard cross-entropy cost.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_placeholder))
# Choosing the Adam gradient optimizer.
gradient_optimizer = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(loss)


# ## Training

# In[ ]:


# CONSTANTS
TRAINING_ITERATIONS = 100000

# UTILITIES
session = tf.InteractiveSession()

def get_batch():
    batch = np.zeros((BATCH_SIZE, MAX_WORDS_PER_SENTENCE), dtype='int32')
    labels = np.zeros((BATCH_SIZE, 2), dtype='float32')
    
    # Note: Randomization with equal probabilities does not preserve the proportions of labels in the training dataset.
    batch_members = np.random.choice(TRAINING_SAMPLES, BATCH_SIZE, replace=False)
    for batch_index, sentence_index in enumerate(batch_members, 0):
        batch[batch_index, :] = identities_matrix[sentence_index]
        labels[batch_index, :] = training_labels_matrix[sentence_index]    
    return batch, labels


def load_pretrained_network(checkpoint_saver, models_directory):
    checkpoint_saver.restore(session, tf.train.latest_checkpoint(models_directory))


def initialize_network():
    session.run(tf.global_variables_initializer())


# TENSORBOARD CONFIGURATION
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', prediction_accuracy)
merged_summary = tf.summary.merge_all()

logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, session.graph)


# In[ ]:


checkpoint_saver = tf.train.Saver()

# load_pretrained_network(checkpoint_saver, 'models')
initialize_network()

for iteration in range(1, TRAINING_ITERATIONS + 1):
    batch, labels = get_batch();
    session.run(gradient_optimizer, {batch_placeholder: batch, labels_placeholder: labels})
    
    if iteration % 50 == 0:
        summary = session.run(merged_summary, {batch_placeholder: batch, labels_placeholder: labels})
        writer.add_summary(summary, iteration)
    
    if iteration % 10000 == 0:
        save_path = checkpoint_saver.save(session, "models/pretrained_lstm.ckpt", global_step=iteration)
        print("Checkpoint saved to: %s." % save_path)

writer.close()

