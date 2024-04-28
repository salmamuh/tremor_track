
from flask import Flask, request, jsonify,url_for
import tensorflow as tf
import base64
import cv2
import numpy as np
import joblib
import datetime
import librosa
# from moviepy.editor import VideoFileClip
from scipy.io.wavfile import write
import csv
new_model = tf.keras.models.load_model('F:/flutter appps/flak first/ResNetModel.h5')
model = joblib.load('model.pkl')


app = Flask(__name__)


response_data = []


@app.route('/api', methods=['GET', 'PUT'])
def home():
    # Default empty response data
    try:
        inputchar = request.get_data()
        imgdata = base64.b64decode(inputchar)
        nparr = np.frombuffer(imgdata, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            target_size = (224, 224)
            resized_img = cv2.resize(img, target_size)
            resized_img = np.expand_dims(resized_img, axis=0)
            predictions = new_model.predict(resized_img)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            class_labels = ['Healthy', 'Parkinson']
            result = class_labels[predicted_class_index]
        else:
            result = "Error: Failed to load or decode image"
    except Exception as e:
        result = f"Error: {str(e)}"
    
    # Get the current date and time
    now = datetime.datetime.now()
    # Update the response data with the result, image URL, and date and time
    response_data.append({
        "result": result,
        "image_url": url_for('static', filename='images/ear.png'),
        "date_time": now.strftime("%Y-%m-%d %H:%M:%S")
    })
    return jsonify(response_data)




@app.route('/process_audio', methods=['PUT', 'GET'])
def process_audio():
    
    if request.method == 'PUT' or request.method == 'GET':
        try:
            # Extract audio data from request
            data = request.get_data('audioData')
            if not data:
               return jsonify(error='No audio data found'), 400


        # Decode the base64 audio data
            audio_bytes = base64.b64decode(data)
        
        # Write the audio data to a temporary WAV file
            temp_wav_file = 'temp_audio.wav'
            write(temp_wav_file, 16000, np.frombuffer(audio_bytes, dtype=np.int16))

            
            # Load audio and extract features
            y, sr = librosa.load(temp_wav_file, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            # Calculate mean of each feature
            mfcc_mean = np.mean(mfccs, axis=1)
            chroma_mean = np.mean(chroma, axis=1)

            # Combine features into a single feature vector
           # Combine features into a single feature vector
            features = np.concatenate((mfcc_mean[:11], chroma_mean[:11]))

# Ensure that features has 22 elements
            assert len(features) == 22, "Features array should have 22 elements"

            audio_features = np.array(features).reshape(1, -1)

            # Perform prediction
            pred = model.predict(audio_features.astype('double'))
            result_msg = "Ill" if pred[0] == 1 else "Healthy"
            print(result_msg) 
            now = datetime.datetime.now()
    # Update the response data with the result, image URL, and date and time
            response_data.append({
        "result": result_msg,
        "image_url": url_for('static', filename='temp_audio.wav'),
        "date_time": now.strftime("%Y-%m-%d %H:%M:%S")
    })
            return jsonify(response_data)

        except Exception as e:
            # Log the error
            print(f"Error processing audio: {e}")
            # Return an error response
            return jsonify(error=str(e)), 500
    else:
        # Return a message indicating that the method is not allowed
        return jsonify(error="Method not allowed"), 405







if __name__ == "__main__":
    app.run( host='0.0.0.0')

#   Load the audio file
# audio_path = 'path_to_your_audio_file.wav'
# y, sr = librosa.load(audio_path, sr=None)

# # Extract features


# print(features)   
#   data = request.json
#   audio_features = data.get('audioFeatures')

    # Ensure audio_features is a 2D array as expected by scikit-learn

    
    # Create a result message based on the prediction




# def home():
#       # Default empty response data
#     try:
#         inputchar = request.get_data()
#         imgdata = base64.b64decode(inputchar)
#         nparr = np.frombuffer(imgdata, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if img is not None:
#             resized = cv2.resize(img, (128, 128))
#             grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#             normalized = grayscale / 255.0
#             reshaped = np.expand_dims(normalized, axis=-1)
#             prediction = new_model.predict(np.expand_dims(reshaped, axis=0))
#             result = "Parkinson's Disease" if prediction[0][0] > 0.5 else "Healthy"
#         else:
#             result = "Error: Failed to load or decode image"
#     except Exception as e:
#         result = f"Error: {str(e)}"
    
#     # Get the current date and time
#     now = datetime.datetime.now()
#     # Update the response data with the result, image URL, and date and time
#     response_data.append({
#         "result": result,
#         "image_url": url_for('static', filename='images/ear.png'),
#         "date_time": now.strftime("%Y-%m-%d %H:%M:%S")
#     })
#     return jsonify(response_data)


# import librosa
# import numpy as np

# Load the audio file
# audio_path = 'path_to_your_audio_file.wav'
# y, sr = librosa.load(audio_path, sr=None)

# # Extract features
# mfccs = librosa.feature.mfcc(y=y, sr=sr)
# chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# # Calculate mean of each feature
# mfcc_mean = np.mean(mfccs, axis=1)
# chroma_mean = np.mean(chroma, axis=1)

# # Combine features into a single feature vector
# features = np.concatenate((mfcc_mean, chroma_mean))

# print(features)

# def extract_features(audio):
#     mean_amplitude = np.mean(audio)
#     max_amplitude = np.max(audio)
#     min_amplitude = np.min(audio)
#     zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0, 0]
#     spectral_centroid = librosa.feature.spectral_centroid(y=audio)[0, 0]
#     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio)[0, 0]
#     spectral_rolloff = librosa.feature.spectral_rolloff(y=audio)[0, 0]
#     chroma_stft = librosa.feature.chroma_stft(y=audio)[0, 0]
#     rms = librosa.feature.rms(y=audio)[0, 0]
#     mfccs = librosa.feature.mfcc(y=audio)
#     mfcc_mean = np.mean(mfccs, axis=1)
    
#     return [mean_amplitude, max_amplitude, min_amplitude, zero_crossing_rate, spectral_centroid,
#             spectral_bandwidth, spectral_rolloff, chroma_stft, rms] + mfcc_mean.tolist()

# @app.route('/process', methods=['POST'])
# def process_audio():
#     data = request.get_data()
#     audio_path = 'temp_audio.m4a'
#     with open(audio_path, 'wb') as file:
#         file.write(data)
    
#     clip = VideoFileClip(audio_path)
#     features = extract_features(clip.audio.to_soundarray())
    
#     # Perform prediction using the extracted features
#     data_array = np.array([features])
#     result = model.predict(data_array)
#     result_msg = "Ill" if result[0] == 1 else "Healthy"
    
#     return jsonify({'result': result_msg, 'prediction': result[0].tolist()}) 


# @app.route('/process_audio', methods=['PUT', 'GET'])
# def process_audio():
#     if request.method == 'PUT' or request.method == 'GET':
#         try:
#             # Use the path to the example.wav file
#             audio_path = 'example.wav'
            
#             # Load audio and extract features
#             y, sr = librosa.load(audio_path, sr=None)
#             mfccs = librosa.feature.mfcc(y=y, sr=sr)
#             chroma = librosa.feature.chroma_stft(y=y, sr=sr)

#             # Calculate mean of each feature
#             mfcc_mean = np.mean(mfccs, axis=1)
#             chroma_mean = np.mean(chroma, axis=1)

#             # Combine features into a single feature vector
#            # Combine features into a single feature vector
#             features = np.concatenate((mfcc_mean[:11], chroma_mean[:11]))

# # Ensure that features has 22 elements
#             assert len(features) == 22, "Features array should have 22 elements"

#             audio_features = np.array(features).reshape(1, -1)

#             # Perform prediction
#             pred = model.predict(audio_features.astype('double'))
#             result_msg = "Ill" if pred[0] == 1 else "Healthy"
#             print(result_msg) 
#             return jsonify(result=result_msg, prediction=int(pred[0]))

#         except Exception as e:
#             # Log the error
#             print(f"Error processing audio: {e}")
#             # Return an error response
#             return jsonify(error=str(e)), 500
#     else:
#         # Return a message indicating that the method is not allowed
#         return jsonify(error="Method not allowed"), 405








# from flask import Flask, request,jsonify
# import tensorflow as tf
# import base64
# import cv2
# import numpy as np
# import joblib
# new_model = tf.keras.models.load_model('F:/flutter appps/flak first/parkinson_disease_detection.h5')
# app = Flask(__name__)

# @app.route('/api', methods=['PUT'])
# def home():
#     inputchar = request.get_data()
#     imgdata = base64.b64decode(inputchar)
#     filename = "something.jpg"
#     with open(filename, 'wb') as f:
#         f.write(imgdata)
    
#     img = cv2.imread('something.jpg')
#     resized = cv2.resize(img, (128, 128))
#     grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     normalized = grayscale / 255.0
#     reshaped = np.expand_dims(normalized, axis=-1)  # Assuming model input shape is (128, 128, 1)
    
#     prediction = new_model.predict(np.expand_dims(reshaped, axis=0))
#     result=""
#     if prediction[0][0] > 0.5:
#         result= "Parkinson's Disease"
#     else:
#         result="Healthy"
#     return  jsonify(result)
# model = joblib.load('model.pkl')

# @app.route('/process_audio', methods=['POST'])
# def result(model, audio_features):
#     # Ensure audio_features is a 2D array as expected by scikit-learn
#     audio_features = np.array(audio_features).reshape(1, -1)
    
#     # Perform prediction
#     pred = model.predict(audio_features)
    
#     # Create a result message based on the prediction
#     result_msg = "Ill" if pred[0] == 1 else "Healthy"
    
#     return result_msg, pred[0]

# @app.route('/process_audio', methods=['POST'])
# def process_audio():
#     data = request.json
#     audio_features = data.get('audioFeatures')
    
#     # Perform prediction using the result function
#     result_msg, pred = result(model, audio_features)
    
#     return jsonify({'result': result_msg, 'prediction': pred})



# if __name__ == "__main__":
#     app.run(host='0.0.0.0')









# def process_audio():
#     data = request.json
#     audio_features = data.get('audioFeatures')
    
#     # Perform prediction using the result function
#     result_msg, pred = result(model,  audio_features)
    
#     return jsonify({'result': result_msg, 'prediction': pred})


# from flask import Flask,request
# import tensorflow as tf
# import base64
# import cv2
# import numpy as np
# new_model=tf.keras.models.load_model('F:/flutter appps/flak first/parkinson_disease_detection.h5')
# app = Flask(__name__)
# @app.route('/api',methods=['PUT'])
# def home():
#     inputchar=request.get_data()
#     imgdata=base64.b64decode(inputchar)
#     filename="something.jpg"
#     with open(filename,'wb') as f:
#         f.write(imgdata)
#     img = cv2.imread('something.jpg')
#     resized = cv2.resize(img, (128, 128))
#     grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     normalized = grayscale / 255.0
#     reshaped = np.expand_dims(normalized, axis=0)
#     prediction = new_model.predict(reshaped.reshape(-1, 128, 128, 1))
#     if prediction[0][0] > 0.5:
#         return "Parkinson's Disease"
#     else:
#         return "Healthy"
   

# if __name__ == "__main__":
#     app.run(host='0.0.0.0')





#     y, sr = librosa.load('path_to_your_audio_file.wav', sr=None)

# # Extract MFCCs
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# # Calculate statistics of MFCCs
#     mfccs_mean = np.mean(mfccs, axis=1)
#     mfccs_std = np.std(mfccs, axis=1)
#     mfccs_min = np.min(mfccs, axis=1)
#     mfccs_max = np.max(mfccs, axis=1)

# # Combine features into a single array
#     features = np.concatenate([mfccs_mean, mfccs_std, mfccs_min, mfccs_max])

# # Format the features into the desired format
#     formatted_features = "data3 = [" + ", ".join([f"{val:.6f}" for val in features]) + "]"