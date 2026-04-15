# Real-Time Facial Emotion Recognition (Colab Guide)

This project trains a CNN to classify facial emotions using the CK+ dataset in Google Colab.


## What this notebook does

- Installs required packages (`tensorflow`, `opencv-python-headless`, `scikit-learn`, `matplotlib`, `seaborn`)
- Lets you upload a CK+ dataset zip file directly in Colab
- Extracts and auto-detects class folders
- Loads images, converts to grayscale, and resizes to `100x100`
- Expands very small classes in memory for smoother training
- Trains and evaluates a CNN model (accuracy, macro F1, classification report, confusion matrix)
- Saves the trained model to `/content/emotion_cnn_model.keras`

## Dataset format expected

Inside your uploaded zip, include these class folders:

- `neutral`
- `anger`
- `contempt`
- `disgust`
- `fear`
- `happiness`
- `sadness`
- `surprise`

Each folder should contain images (`.png`, `.jpg`, `.jpeg`, or `.bmp`).

## Run in Google Colab (step-by-step)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `facial_emotion_recognition.ipynb` and open it.
3. (Optional but recommended) Set runtime to GPU:
   - `Runtime` -> `Change runtime type` -> `T4 GPU` (or any available GPU)
4. Run cells from top to bottom.
5. When prompted by the upload cell, upload your CK+ zip file (`CK+.zip` is ideal).
6. Wait for:
   - extraction
   - dataset detection
   - training
   - evaluation plots and metrics
7. At the end, your model is saved as:
   - `/content/emotion_cnn_model.keras`
8. Download the model from the Colab file panel if you want to keep it locally.

## Outputs

- Class-wise image count table
- Train/validation loss and accuracy curves
- Test accuracy and macro F1 score
- Classification report
- Confusion matrix
