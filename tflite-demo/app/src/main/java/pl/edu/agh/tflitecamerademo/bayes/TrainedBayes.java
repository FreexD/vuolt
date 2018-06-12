package pl.edu.agh.tflitecamerademo.bayes;

import android.content.Context;
import android.os.Environment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.List;

import de.daslaboratorium.machinelearning.classifier.Classification;
import de.daslaboratorium.machinelearning.classifier.bayes.BayesClassifier;

public final class TrainedBayes {

    private static final String DATASET_PATH = "bayes_train.csv";
    private final BayesClassifier<String, String> classifier = new BayesClassifier<>();

    private void pretrain(Context context) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(DATASET_PATH)));

        String line;
        while ((line = reader.readLine()) != null) {
            String label = String.valueOf(line.charAt(0));
            learn(label, line.substring(2));
        }
        reader.close();

        FileOutputStream fos = new FileOutputStream(
                new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                        "bayes.model")
        );
        ObjectOutputStream os = new ObjectOutputStream(fos);
        os.writeObject(classifier);
        os.close();
        fos.close();
    }

    public TrainedBayes(Context context) throws IOException {
        pretrain(context);
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
