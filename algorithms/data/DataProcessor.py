{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " # Twitter Sentiment Analysis dataset processor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from matplotlib import pyplot as plt\n",
    "\n",
    "import numpy as np\n",
    "\n",
    "import math\n",
    "import csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Utilities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_sentences_per_label(dataset_filename):\n",
    "    positive_statements = []\n",
    "    negative_statements = []\n",
    "\n",
    "    with open(dataset_filename, 'r', encoding='utf-8') as csv_file:\n",
    "        csv_reader = csv.reader(csv_file, delimiter=',')\n",
    "        next(csv_reader)\n",
    "\n",
    "        for sentence_counter, sentence in enumerate(csv_reader, 1):\n",
    "            sentiment = sentence[1]\n",
    "            if sentiment == '0':\n",
    "                negative_statements.append(sentence_counter)\n",
    "            elif sentiment == '1':\n",
    "                positive_statements.append(sentence_counter)\n",
    "            else:\n",
    "                print('Unexpected sentiment value {}'.format(sentiment))\n",
    "    \n",
    "    return positive_statements, negative_statements"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dataset splitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# CONSTANTS\n",
    "TRAIN_DATASET_FACTOR = 0.5          # TODO\n",
    "TEST_DATASET_FACTOR = 0.4           # TODO\n",
    "\n",
    "MOBILE_DATASETS_NUMBER = 3          # TODO\n",
    "MOBILE_DATASET_FACTOR = 0.3         # TODO\n",
    "\n",
    "# UTILITIES\n",
    "def clean_statement(statement):\n",
    "    print(\"ST: \", statement)\n",
    "    # TODO:\n",
    "    #   - convert to lowercase\n",
    "    #   - strip whitespaces,\n",
    "    #   - handle hashtags (#),\n",
    "    #   - handle emoticons (consider e.g. \"< 33\", \":)\" and \":O\"),\n",
    "    #   - remove mentions (@),\n",
    "    #   - remove insignificant punctuation (watch out for things like \"I'm\" and \"How's\", however \"'s\" in \"Tom's house\" should be removed),\n",
    "    #   - remove special characters (e.g. &amp;, < b >).\n",
    "    # In general, try to search for sth like regexp for english words.\n",
    "    return statement"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "a must be non-empty",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-11-f1e49e09387f>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     28\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     29\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 30\u001b[0;31m \u001b[0msplit_dataset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m<ipython-input-11-f1e49e09387f>\u001b[0m in \u001b[0;36msplit_dataset\u001b[0;34m()\u001b[0m\n\u001b[1;32m     22\u001b[0m     \u001b[0;32mwith\u001b[0m \u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'dataset.csv'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'r'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mencoding\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'utf-8'\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mdataset_file\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     23\u001b[0m         \u001b[0mcsv_reader\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcsv\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreader\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdataset_file\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdelimiter\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m','\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 24\u001b[0;31m         \u001b[0mpositive_statements\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnegative_statements\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcreate_dataset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpositive_statements\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnegative_statements\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mTRAIN_DATASET_FACTOR\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcsv_reader\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'train.csv'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     25\u001b[0m         \u001b[0mpositive_statements\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnegative_statements\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcreate_dataset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpositive_statements\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnegative_statements\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mTEST_DATASET_FACTOR\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcsv_reader\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'test.csv'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     26\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mmobile_dataset_index\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mMOBILE_DATASETS_NUMBER\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m<ipython-input-11-f1e49e09387f>\u001b[0m in \u001b[0;36mcreate_dataset\u001b[0;34m(available_statements, factor, csv_reader, output_filename, should_clean_statements, remove_chosen_statements)\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0msplit_dataset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mcreate_dataset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mavailable_statements\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfactor\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcsv_reader\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0moutput_filename\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mshould_clean_statements\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mremove_chosen_statements\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m         \u001b[0mchosen_positive\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrandom\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mchoice\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mavailable_statements\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mceil\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfactor\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mavailable_statements\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreplace\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m         \u001b[0mchosen_negative\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrandom\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mchoice\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mavailable_statements\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mceil\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfactor\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mavailable_statements\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreplace\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m         \u001b[0mchosen_shuffled\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconcatenate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mchosen_positive\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mchosen_negative\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32mmtrand.pyx\u001b[0m in \u001b[0;36mmtrand.RandomState.choice\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: a must be non-empty"
     ],
     "output_type": "error"
    }
   ],
   "source": [
    "def split_dataset():\n",
    "    def create_dataset(available_statements, factor, csv_reader, output_filename, should_clean_statements, remove_chosen_statements): \n",
    "        chosen_positive = np.random.choice(available_statements[0], math.ceil(factor * len(available_statements[0])), replace=False)\n",
    "        chosen_negative = np.random.choice(available_statements[1], math.ceil(factor * len(available_statements[1])), replace=False)\n",
    "        chosen_shuffled = np.concatenate((chosen_positive, chosen_negative))\n",
    "        np.random.shuffle(chosen_shuffled)\n",
    "        \n",
    "        with open(output_filename, 'w', encoding='utf-8', newline='') as output_file:\n",
    "            csv_writer = csv.writer(output_file, delimiter=',')\n",
    "            for statement_index in chosen_shuffled:\n",
    "                statement = csv_reader[statement_index]\n",
    "                csv_writer.writerow([statement[1], clean_statement(statement[3]) if should_clean_statements else statement[3]])\n",
    "        print('File {} saved.'.format(output_filename))\n",
    "        \n",
    "        if remove_chosen_statements:\n",
    "            list_difference = lambda first, second: list(set(first) - set(second))\n",
    "            return list_difference(available_statements[0], chosen_positive.tolist()), list_difference(available_statements[1], chosen_negative.tolist())\n",
    "        return available_statements[0], available_statements[1]\n",
    "    \n",
    "    \n",
    "    positive_statements, negative_statements = get_sentences_per_label('dataset.csv')    \n",
    "    with open('dataset.csv', 'r', encoding='utf-8') as dataset_file:\n",
    "        csv_reader = list(csv.reader(dataset_file, delimiter=','))\n",
    "        positive_statements, negative_statements = create_dataset((positive_statements, negative_statements), TRAIN_DATASET_FACTOR, csv_reader, 'train.csv', True, True)\n",
    "        positive_statements, negative_statements = create_dataset((positive_statements, negative_statements), TEST_DATASET_FACTOR, csv_reader, 'test.csv', True, True)\n",
    "        for mobile_dataset_index in range(1, MOBILE_DATASETS_NUMBER + 1):\n",
    "            positive_statements, negative_statements = create_dataset((positive_statements, negative_statements), MOBILE_DATASET_FACTOR, csv_reader, 'mobile_{}.csv'.format(mobile_dataset_index), True, False)\n",
    "\n",
    "\n",
    "split_dataset()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dataset analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def analyse_dataset(dataset_filename, has_header=True):\n",
    "    positive_statements, negative_statements = get_sentences_per_label(dataset_filename)\n",
    "    total_statements = len(positive_statements) + len(negative_statements)\n",
    "    \n",
    "    # CLASSES DISTRIBUTION\n",
    "    barchart = plt.bar([0, 1], [len(positive_statements) / total_statements, len(negative_statements) / total_statements], tick_label=['positive', 'negative'])\n",
    "    \n",
    "    for bar in barchart:\n",
    "        height = bar.get_height()\n",
    "        plt.text(bar.get_x() + 0.3, height + 0.005, \"{:.5f}\".format(height))\n",
    "    plt.title('Classes distribution')\n",
    "    plt.xlabel('Class')\n",
    "    plt.ylabel('Participation in the dataset')\n",
    "    plt.show()\n",
    "    \n",
    "    print('Positive statements: {} ({:.5f}%).'.format(len(positive_statements), len(positive_statements) / total_statements))\n",
    "    print('Negative statements: {} ({:.5f}%).'.format(len(negative_statements), len(negative_statements) / total_statements))\n",
    "    print('Total statements: {}.'.format(total_statements))\n",
    "    \n",
    "    # WORDS STATISTICS\n",
    "    with open(dataset_filename, 'r', encoding='utf-8') as csv_file:\n",
    "        csv_reader = csv.reader(csv_file, delimiter=',')\n",
    "        if has_header:\n",
    "            next(csv_reader)\n",
    "        \n",
    "        words_count = []\n",
    "        for sentence in csv_reader:\n",
    "            words_count.append(len(sentence[3].split()))\n",
    "    \n",
    "    plt.hist(words_count, max(words_count))\n",
    "    \n",
    "    plt.title('Words count histogram')\n",
    "    plt.xlabel('Words count')\n",
    "    plt.ylabel('Frequency')\n",
    "    plt.show()\n",
    "    \n",
    "    print('Maximal number of words: {}.'.format(max(words_count)))\n",
    "    print('Average number of words: {:.2f}.'.format(sum(words_count) / len(words_count)))\n",
    "    \n",
    "    sorted_words_count = sorted(words_count)\n",
    "    quartile_index = len(words_count) // 4\n",
    "    print('First quartile: {}, second quartile (median): {}, third quartile: {}.'.format(sorted_words_count[quartile_index], sorted_words_count[2 * quartile_index], sorted_words_count[3 * quartile_index]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Positive statements: 0 (0.00000%).\nNegative statements: 2 (1.00000%).\nTotal statements: 2.\nMaximal number of words: 7.\nAverage number of words: 6.50.\nFirst quartile: 6, second quartile (median): 6, third quartile: 6.\n"
     ]
    }
   ],
   "source": [
    "analyse_dataset('dataset.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
