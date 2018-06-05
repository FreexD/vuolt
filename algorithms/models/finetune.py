import csv
import datetime
import sys

import numpy as np
import tensorflow as tf


def main(arguments):
    # CONSTANTS
    TRAINING_SAMPLES = int(arguments[1])
    MAX_WORDS_PER_SENTENCE = 64


    # UTILITIES
    def save_numpy_array(filename, array):
        np.save(filename, array)

    def load_numpy_array(filename):
        return np.load(filename)


    # WORD2VEC
    words_list = [word.decode('UTF-8') for word in load_numpy_array('wordsList.npy')]
    word_vectors = load_numpy_array('wordVectors.npy')


    # IDENTITIES MATRIX
    DATASET_FILENAME = arguments[0]

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

    ids_matrix_filename = 'idsMatrix_finetuning_{}.npy'.format(DATASET_FILENAME)
    labels_matrix_filename = 'labelsMatrix_train.npy'.format(DATASET_FILENAME)

    build_identities_matrix(DATASET_FILENAME, ids_matrix_filename, labels_matrix_filename)
    identities_matrix = load_numpy_array(ids_matrix_filename)
    training_labels_matrix = load_numpy_array(labels_matrix_filename)


    # LSTM MODEL
    LSTM_UNITS = 128
    BATCH_SIZE = 32

    tf.reset_default_graph()

    batch_placeholder = tf.placeholder(tf.int32, (BATCH_SIZE, MAX_WORDS_PER_SENTENCE))
    labels_placeholder = tf.placeholder(tf.float32, (BATCH_SIZE, 2))

    batch_input_tensor = tf.Variable(tf.zeros((BATCH_SIZE, MAX_WORDS_PER_SENTENCE, 50), dtype=tf.float32))
    batch_input_tensor = tf.nn.embedding_lookup(word_vectors, batch_placeholder)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_UNITS)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)

    output_tensor, _ = tf.nn.dynamic_rnn(lstm_cell, batch_input_tensor, dtype=tf.float32)

    weight_matrix = tf.Variable(tf.truncated_normal((LSTM_UNITS, 2)))
    bias_neurons = tf.Variable(tf.constant(0.1, shape=(2,)))
    output_tensor = tf.transpose(output_tensor, [1, 0, 2])
    output_activation_tensor = tf.gather(output_tensor, int(output_tensor.get_shape()[0]) - 1)
    prediction = (tf.matmul(output_activation_tensor, weight_matrix) + bias_neurons)

    is_prediction_correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_placeholder, 1))
    prediction_accuracy = tf.reduce_mean(tf.cast(is_prediction_correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_placeholder))
    gradient_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


    # FINETUNING
    TRAINING_ITERATIONS = int(arguments[2])
    session = tf.InteractiveSession()

    def get_batch():
        batch = np.zeros((BATCH_SIZE, MAX_WORDS_PER_SENTENCE), dtype='int32')
        labels = np.zeros((BATCH_SIZE, 2), dtype='float32')

        batch_members = np.random.choice(TRAINING_SAMPLES, BATCH_SIZE, replace=False)
        for batch_index, sentence_index in enumerate(batch_members, 0):
            batch[batch_index, :] = identities_matrix[sentence_index]
            labels[batch_index, :] = training_labels_matrix[sentence_index]
        return batch, labels

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', prediction_accuracy)
    merged_summary = tf.summary.merge_all()

    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, session.graph)

    checkpoint_saver = tf.train.Saver()
    checkpoint_saver.restore(session, tf.train.latest_checkpoint(arguments[3]))

    for iteration in range(1, TRAINING_ITERATIONS + 1):
        batch, labels = get_batch()
        session.run(gradient_optimizer, {batch_placeholder: batch, labels_placeholder: labels})

        if iteration % 50 == 0:
            summary = session.run(merged_summary, {batch_placeholder: batch, labels_placeholder: labels})
            writer.add_summary(summary, iteration)

        if iteration == TRAINING_ITERATIONS:
            save_path = checkpoint_saver.save(session, "models/finetuned_lstm.ckpt", global_step=iteration)
            print("Checkpoint saved to: %s." % save_path)

    writer.close()


# Running: python finetune.py [DATASET FILENAME] [NUMBER OF TRAINING SAMPLES] [NUMBER OF ITERATIONS] [MODELS DIRECTORY]
if __name__ == "__main__":
    main(sys.argv[1:])
