# VUOLT - mobile app
(a.k.a Veterans Unite One Last Time)

Mobile cloud distributed machine learning for Mobile Systems and Knowledge Engineering 
and Machine Learning last semester project.

## Project purpose

This directory contains mobile app for Vuolt.
It implements Naive Bayes and tensorflow inference of server side LSTM classifiers for 
speech sentiment recognition (https://en.wikipedia.org/wiki/Sentiment_analysis). Mobile
application is written only for devices with Android OS.

## Code description

The file `AndroidManifest.xml` declares application settings and permissions needed for
it to work. Most of the logic of the app is implemented in `MainActivity` class. It
is the activity that the application opens on launch. It triggers NaiveBayes training
with an earlier prepared subset of training data and triggers LSTM model inference.
`MainActivity` view is described in 'activity_main.xml' layout. `MainActivity` handles 
actions on each button clicked and switch value changed.

### NaiveBayes

`TraniedBayes` class is responsible for Naive Bayes classifier logic used for sentiment
analysis in our application. Since the training of Naive Bayes is done locally on each 
mobile device, the only thing you need to provide to train it is a CSV file with training
data. The file should be added to application assets and the path to it configured through
`DATASET_PATH` field in `TrainedBayes` class. Once provided, NaiveBayes will be trained
on each application launch. To prevent Bayes from being trained on each application launch
use `pretrained` method, instead of `pretrain` on `TrainedBayes` instance.

### LSTM

`SentimentClassifierQuantizedMobileNet` is responsible for downloading LSTM model tfile from
thse server and running local inference. This operation is not time consuming. Assuming tfile
is has 5 MB size, the inference will take just around 5 seconds. The class inherits from
abstract `SentimentClassifier` which implements the inference. All the fields and methods are
well described in javadocs.

## Running the app

To run the app, open it in android studio and build an .apk file from it. Later move the
file onto your android device and install the application. After running the app, NaiveBayes
will be trained and LSTM inferred. Then you would see the app GUI. It allows for input of 
a sentence and choosing (with a switch) weather it is positive or not. You can later trigger
training of both models manually or classification using both models.

WARNING! To train LSTM you would need a working server where you could post your
sentence with its classification value (1 - positive or 0 - negative). You can configure the
server url through `SERVER_URL` field in `MainActivity`. 

If the difference between lstm and
bayes classification results will differ too much (values configurable in `MainActivity`) 
training would be triggered automatically on both models.



