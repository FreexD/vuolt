
# coding: utf-8

# # Sentiment analysis using Naive Bayes classifier

# In[1]:


from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

import numpy as np

import csv


# In[2]:


# CONSTANTS
TRAINING_SAMPLES = 142077
TESTING_SAMPLES = 315726

HASHING_VECTORIZER_FEATURES = 2**18


# ## Term Frequency - Inversed Document Frequency

# In[3]:


def build_tfidf(dataset_filename):
    sentences = []
    labels = []
    
    with open(dataset_filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for sentence in csv_reader:
            sentences.append(sentence[1])
            labels.append(0 if sentence[0] == '0' else 1)
        
        hashing_vectorizer = HashingVectorizer(n_features=HASHING_VECTORIZER_FEATURES, alternate_sign=False)
        hashed_bow = hashing_vectorizer.transform(sentences)
        
        tfidf_transformer = TfidfTransformer()
        tfidf = tfidf_transformer.fit_transform(hashed_bow)
        
        return tfidf, np.array(labels)


# ##  Training

# In[4]:


def setup_classifier(input_data, labels_data):
    classifier = MultinomialNB()
    classifier.fit(input_data, labels_data)
    joblib.dump(classifier, 'bayesClassifier.pkl')
    

def setup_pretrained_classifier(input_data, labels_data):
    classifier = joblib.load('bayesClassifier.pkl')
    classifier.partial_fit(input_data, labels_data, classes=[0, 1])
    joblib.dump(classifier, 'bayesClassifier.pkl')


def load_classifier():
    return joblib.load('bayesClassifier.pkl')


def train():
    tfidf, labels = build_tfidf('mobile_1.csv')
    
    setup_classifier(tfidf, labels)
    #setup_pretrained_classifier(tfidf, labels)


# train()


# ## Inference

# In[5]:


# TODO: Implement some evaluation measures (e.g. True/False Positives/Negatives and Precision, Recall, F1-score)

def batch_inference(classifier):
    tfidf, labels = build_tfidf('test.csv')
    incorrect_predictions = []
    
    prediction_results = classifier.predict(tfidf)
    for result_index, (predicted_result, ground_truth) in enumerate(zip(prediction_results, labels)):
        if predicted_result != ground_truth:
            incorrect_predictions.append(result_index)
    
    # TODO: Handle prediction results (e.g. compute evaluation measures, save them to a *.csv file for further fine-tuning).
    print('Incorrect predictions: {} ({:.2f}%)'.format(len(incorrect_predictions), (len(incorrect_predictions) / TESTING_SAMPLES) * 100))


def simple_inference(classifier, sentence):
    hashing_vectorizer = HashingVectorizer(n_features=HASHING_VECTORIZER_FEATURES, alternate_sign=False)
    hashed_bow = hashing_vectorizer.transform([sentence])
        
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(hashed_bow)

    prediction_result = classifier.predict_proba(tfidf)[0]
    print('Predicted sentiment: {:.2f}% positive, {:.2f}% negative'.format(prediction_result[1] * 100, prediction_result[0] * 100))


classifier = load_classifier()

batch_inference(classifier)
simple_inference(classifier, 'This classifier works quite well.')
simple_inference(classifier, 'However the neural network performs better.')

