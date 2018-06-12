package pl.edu.agh.tflitecamerademo;

import android.os.Bundle;
import android.app.Activity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.TextView;

import java.io.IOException;

import de.daslaboratorium.machinelearning.classifier.Classification;
import pl.edu.agh.tflitecamerademo.bayes.TrainedBayes;

public class MainActivity extends Activity {

    Button evaluateButton;
    Button trainButton;
    TextView lstmResultTextView;
    TextView bayesResultTextView;
    Switch positiveSwitch;
    EditText sentenceEditText;

    TrainedBayes trainedBayes;

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

        evaluateButton.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        Classification<String, String> classify = trainedBayes.classify(sentenceEditText.getText().toString());
                        bayesResultTextView.setText(
                                String.format("%s %s", classify.getCategory(), classify.getProbability())
                        );
                    }
                }
        );
    }
}
