# VUOLT
Mobile cloud distributed machine learning for Mobile Systems and Knowledge Engineering and Machine Learning last semester project.

A.k.a the Veterans Unite One Last Time.

## Related articles

Those can be found in articles folder.


## Documentation

Project schedule and final documentation can be found in docs folder.

## Server side implementation

The folder algorithms contains python notebook implementations of both LSTM and NaiveBayes models, that were set up on Google Cloud for the sake of the project. Data was stored in AWS buckets and communication with server implemented through lambdas.

## Mobile implementation

Implementation for android mobile devices can be fould in tflite-demo folder. Inside there is a README.md file which documents the whole package. Furhtermore, classes, fields and methods that play vital role in the project are documented thoroughly with javadocs.

### Dataset

It is avaliable in three parts named training.1600000.processed.noemoticon.part{01,02,03}.rar. The whole set was divided, as described in the documentation of the project to provide sufficient training and testing data.
