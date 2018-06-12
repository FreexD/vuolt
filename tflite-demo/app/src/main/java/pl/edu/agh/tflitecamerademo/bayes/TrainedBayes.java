package pl.edu.agh.tflitecamerademo.bayes;

import android.content.Context;
import android.os.Environment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.List;

import de.daslaboratorium.machinelearning.classifier.Classification;
import de.daslaboratorium.machinelearning.classifier.bayes.BayesClassifier;

public final class TrainedBayes {

    private static final String DATASET_PATH = "bayes_train.csv";

    private BayesClassifier<String, String> classifier = new BayesClassifier<>();

    private static final File modelFile = new File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
            "bayes.model"
    );

    private void pretrain(Context context) throws IOException {
        classifier.setMemoryCapacity(10000);
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(DATASET_PATH)));

        String line;
        int c = 0;
        while ((line = reader.readLine()) != null && c < 10000) {
            String label = String.valueOf(line.charAt(0));
            learn(label, line.substring(2));
            c++;
        }
        reader.close();

        FileOutputStream fos = new FileOutputStream(modelFile);
        ObjectOutputStream os = new ObjectOutputStream(fos);
        os.writeObject(classifier);
        os.close();
        fos.close();
    }

    private static BayesClassifier<String, String> pretrained(Context context) throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(modelFile);
        ObjectInputStream is = new ObjectInputStream(fis);
        BayesClassifier<String, String> result = (BayesClassifier<String, String>) is.readObject();
        is.close();
        fis.close();
        return result;
    }

    public TrainedBayes(Context context) throws IOException, ClassNotFoundException {
        pretrain(context);
        classifier = pretrained(context);
    }

    private static List<String> getFeatures(String sentence) {
        return Arrays.asList(sentence.split("\\S"));
    }

    public void learn(String label, String sentence) {
        classifier.learn(label, getFeatures(sentence));
    }

    public Classification<String, String> classify(String sentence) {
        return classifier.classify(getFeatures(sentence));
    }

}
