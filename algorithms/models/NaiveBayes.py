
# coding: utf-8

# # Sentiment analysis using Naive Bayes classifier

# In[1]:


from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

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


train()


# ## Inference

# In[5]:


def batch_inference(classifier):
    dataset = "mobile_1.csv"
    dataset_name = os.path.basename(dataset)
    tfidf, labels = build_tfidf(dataset)
    incorrect_predictions = []
    
    prediction_results = classifier.predict(tfidf)
    for result_index, (predicted_result, ground_truth) in enumerate(zip(prediction_results, labels)):
        if predicted_result != ground_truth:
            incorrect_predictions.append(result_index)
    print(classification_report(labels, prediction_results))

    precision, recall, tresholds = precision_recall_curve(labels, prediction_results)
    average_precision = average_precision_score(labels, prediction_results)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.savefig("pr_curve_lstm%s.png" % dataset_name)

    save_incorrect_results(dataset, "errors_nb_%s.csv" % dataset_name, incorrect_predictions)
    print('Incorrect predictions: {} ({:.2f}%)'.format(len(incorrect_predictions), (len(incorrect_predictions) / TESTING_SAMPLES) * 100))


def save_incorrect_results(dataset_file, output_file, incorret_predictions):
    with open(dataset_file, 'r') as dataset_csv_file:
        with open(output_file, 'w') as error_file:
            csv_reader = csv.reader(dataset_csv_file, delimiter=',')
            csv_writer = csv.writer(error_file)
            for i, row in enumerate(csv_reader):
                if i in incorret_predictions:
                    csv_writer.writerow(row)


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

