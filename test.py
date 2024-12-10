import numpy as np
import librosa
from tensorflow.keras.models import load_model

model = load_model('my_model.keras')


def preprocess_audio(file_path):
 
    audio, sr = librosa.load(file_path, sr=16000)  
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  
    
    
    print(f"Original MFCC shape: {mfcc.shape}")
    
    
    mfcc = np.expand_dims(mfcc, axis=-1)  
    
    
    expected_time_steps = 174  
    if mfcc.shape[1] < expected_time_steps:
        padding = expected_time_steps - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, padding), (0, 0)), mode='constant')  
    elif mfcc.shape[1] > expected_time_steps:
        mfcc = mfcc[:, :expected_time_steps, :]  

    print(f"Adjusted MFCC shape: {mfcc.shape}")
    return mfcc


file_path = r'C:\Users\sharm\project m\speech-emotion-recognition\dataset\Neutral\08a07Na.wav'  

input_data = preprocess_audio(file_path)

predictions = model.predict(np.expand_dims(input_data, axis=0)) 


emotion_index = np.argmax(predictions)


emotion_labels = ['Happy', 'Sad', 'Angry', 'Neutral']


predicted_emotion = emotion_labels[emotion_index]
print(f'The predicted emotion for the audio is: {predicted_emotion}')
