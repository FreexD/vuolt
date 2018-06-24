package pl.edu.agh.tflitecamerademo;

import android.app.Activity;

import java.io.IOException;

/**
 * This classifier works with the quantized MobileNet model.
 */
public class SentimentClassifierQuantizedMobileNet extends SentimentClassifier {

  /**
   * URL to download LSTM tfile from.
   */
  private static final String LSTM_URL = "...";

  /**
   * URL to download LSTM labels tfile from.
   */
  private static final String LSTM_LABELS_URL = "...";

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs.
   * This isn't part of the super class, because we need a primitive array here.
   */
  private byte[][] labelProbArray = null;

  /**
   * Initializes an {@code SentimentClassifier}.
   *
   * @param activity
   */
  SentimentClassifierQuantizedMobileNet(Activity activity) throws IOException {
    super(activity);
    labelProbArray = new byte[1][getNumLabels()];
  }

  @Override
  protected String getModelPath() {
    return LSTM_URL;
  }

  @Override
  protected String getLabelPath() {
    return LSTM_LABELS_URL;
  }

  @Override
  protected int getImageSizeX() {
    return 224;
  }

  @Override
  protected int getImageSizeY() {
    return 224;
  }

  @Override
  protected int getNumBytesPerChannel() {
    // the quantized model uses a single byte only
    return 1;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    textData.put((byte) ((pixelValue >> 16) & 0xFF));
    textData.put((byte) ((pixelValue >> 8) & 0xFF));
    textData.put((byte) (pixelValue & 0xFF));
  }

  @Override
  protected float getProbability(int labelIndex) {
    return labelProbArray[0][labelIndex];
  }

  @Override
  protected void setProbability(int labelIndex, Number value) {
    labelProbArray[0][labelIndex] = value.byteValue();
  }

  @Override
  protected float getNormalizedProbability(int labelIndex) {
    return (labelProbArray[0][labelIndex] & 0xff) / 255.0f;
  }

  @Override
  protected void runInference() {
    tflite.run(textData, labelProbArray);
  }
}
