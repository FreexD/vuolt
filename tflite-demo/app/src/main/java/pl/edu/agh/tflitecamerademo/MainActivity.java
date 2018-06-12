package pl.edu.agh.tflitecamerademo;

import android.os.Bundle;
import android.app.Activity;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.TextView;

public class MainActivity extends Activity {


    Button evaluateButton;
    Button trainButton;
    TextView lstmResultTextView;
    TextView bayesResultTextView;
    Switch positiveSwitch;
    EditText sentenceEditText;


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
    }

}
