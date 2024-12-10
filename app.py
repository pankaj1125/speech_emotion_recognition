from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)

model_path = r'C:\Users\sharm\project m\speech-emotion-recognition\my_model.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' does not exist. Ensure the correct path.")
model = load_model(model_path)

emotion_labels = ['Happy', 'Sad', 'Angry', 'Neutral']


TEMP_FOLDER = 'temp'
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

def preprocess_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        expected_time_steps = 174
        if mfcc.shape[1] < expected_time_steps:
            padding = expected_time_steps - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')
        elif mfcc.shape[1] > expected_time_steps:
            mfcc = mfcc[:, :expected_time_steps]

        mfcc = np.expand_dims(mfcc, axis=-1)
        return mfcc
    except Exception as e:
        print(f"Error in preprocessing audio: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio_file']
    file_path = os.path.join(TEMP_FOLDER, file.filename)
    try:
        file.save(file_path)

       
        input_data = preprocess_audio(file_path)
        if input_data is None:
            return jsonify({'error': 'Error processing audio file.'}), 500

       
        predictions = model.predict(np.expand_dims(input_data, axis=0))
        emotion_index = np.argmax(predictions)
        predicted_emotion = emotion_labels[emotion_index]

        return jsonify({'emotion': predicted_emotion})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Failed to process the file or predict.'}), 500
    finally:
        
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
