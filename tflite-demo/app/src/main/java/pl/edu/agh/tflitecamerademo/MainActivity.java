package pl.edu.agh.tflitecamerademo;

import android.app.Activity;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.TextView;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

import de.daslaboratorium.machinelearning.classifier.Classification;
import pl.edu.agh.tflitecamerademo.bayes.TrainedBayes;

public class MainActivity extends Activity {

    /**
     * Url of server to post each training example.
     */
    private static final String SERVER_URL = "...";
    /**
     * Maximum difference between bayes and lstm classification result, beyond
     * which retraining will be triggered on both models.
     */
    private static final Double MAX_RESULT_DIFFERENCE = 0.1;
    /**
     * If true training will be triggered (on bayes vs lstm result difference
     * bigger than {@value MAX_RESULT_DIFFERENCE}.
     */
    private static final Boolean TRAIN_AFTER_EACH_CLASSIFICATION = false;

    Button evaluateButton;
    Button trainButton;
    TextView lstmResultTextView;
    TextView bayesResultTextView;
    Switch positiveSwitch;
    EditText sentenceEditText;

    TrainedBayes trainedBayes;
    String positiveSwitchToSentenceLabel = "0";

    Map<String, String> queryToLstmMap = new LinkedHashMap<>();

    private Activity getActivity() {
        return this;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        evaluateButton = findViewById(R.id.evaluateButton);
        trainButton = findViewById(R.id.trainButton);
        lstmResultTextView = findViewById(R.id.lstmResultTextView);
        bayesResultTextView = findViewById(R.id.bayesResultTextView);
        positiveSwitch = findViewById(R.id.positiveSwitch);
        sentenceEditText = findViewById(R.id.sentenceEditText);

        try {
            trainedBayes = new TrainedBayes(this);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        positiveSwitch.setOnCheckedChangeListener(
                new CompoundButton.OnCheckedChangeListener() {
                    @Override
                    public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                        positiveSwitchToSentenceLabel = b ? "1" : "0";
                    }
                }
        );

        trainButton.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        trainedBayes.learn(positiveSwitchToSentenceLabel, sentenceEditText.getText().toString());
                        new LstmPostTask().doInBackground(SERVER_URL, sentenceEditText.getText().toString());
                    }
                }
        );

        evaluateButton.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        Classification<String, String> classify = trainedBayes.classify(sentenceEditText.getText().toString());
                        float bayesProbability = classify.getProbability();
                        String sentiment = (classify.getCategory().equals("1")) ? "Positive:" : "Negative:";
                        bayesResultTextView.setText(
                                String.format(Locale.ENGLISH, "%s %.7f", sentiment, bayesProbability)
                        );
                        try {
                            SentimentClassifier classifier = new SentimentClassifierQuantizedMobileNet(getActivity());
                            float lstmProbability = classifier.getProbability(Integer.parseInt(classify.getCategory()));
                            lstmResultTextView.setText(
                                    String.format(Locale.ENGLISH, "%s %.7f", sentiment, lstmProbability)
                            );
                            if (TRAIN_AFTER_EACH_CLASSIFICATION && Math.abs(lstmProbability - bayesProbability) > MAX_RESULT_DIFFERENCE) {
                                trainedBayes.learn(positiveSwitchToSentenceLabel, sentenceEditText.getText().toString());
                                new LstmPostTask().doInBackground(SERVER_URL, sentenceEditText.getText().toString());
                            }
                        } catch (IOException e) {
                            Log.e("INFO", "Failed to use LSTM classifier.");
                        }
                    }
                }
        );
    }

    public class LstmPostTask extends AsyncTask<String, String, String> {

        public LstmPostTask() {
            //set context variables if required
        }

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }


        @Override
        protected String doInBackground(String... params) {

            String urlString = params[0]; // URL to call

            String data = params[1]; //data to post

            OutputStream out = null;
            try {

                URL url = new URL(urlString);

                HttpURLConnection urlConnection = (HttpURLConnection) url.openConnection();

                out = new BufferedOutputStream(urlConnection.getOutputStream());

                BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, "UTF-8"));

                writer.write(data);

                writer.flush();

                writer.close();

                out.close();

                urlConnection.connect();


            } catch (Exception e) {

                System.out.println(e.getMessage());


            }

            return urlString;
        }
    }
}
