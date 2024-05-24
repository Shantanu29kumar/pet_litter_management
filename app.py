
import os
import cv2
import numpy as np
import requests
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Sample data storage (replace with your actual data storage mechanism)
penalty_points = {}

# Load the pre-trained model
model_path = 'model1.h5'  # Path to your pre-trained model
model = load_model(model_path)

# Function to detect litter and update penalty points
def detect_litter_and_update_penalty(frame, pet_id, penalty):
    global penalty_points

    # Preprocess the frame for model input
    img = cv2.resize(frame, (224, 224))  # Resize frame to match model input size
    img = img.astype('float32') / 255.0  # Normalize pixel values

    # Predict the class of the frame
    prediction = model.predict(np.expand_dims(img, axis=0))[0]

    # Assuming 0 represents "litter" and 1 represents "no litter"
    if prediction[0] < 0.5:  # Litter detected
        # Update penalty points for the pet
        if pet_id in penalty_points:
            penalty_points[pet_id] += penalty
        else:
            penalty_points[pet_id] = penalty

        return True  # Litter detected
    else:
        return False  # No litter detected

# Route to handle both serving the webpage and penalty updates
@app.route('/', methods=['GET', 'POST'])
def home():
    global penalty_points

    if request.method == 'POST':
        # Handle video file upload
        if 'file' not in request.files:
            return render_template('index.html', message='No file part'), 400

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file'), 400

        if file and file.filename.endswith(('.mp4', '.avi')):
            # Get pet ID and penalty from the form
            pet_id = request.form.get('pet_id')
            penalty = request.form.get('penalty')
            penalty = int(penalty) if penalty else 0

            # Save the video file temporarily
            video_path = os.path.join('uploads', 'video.mp4')
            file.save(video_path)

            # Process the video file
            cap = cv2.VideoCapture(video_path)
            litter_detected = False
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect litter and update penalty points if detected
                if detect_litter_and_update_penalty(frame, pet_id, penalty):
                    litter_detected = True
                    break

            cap.release()
            cv2.destroyAllWindows()

            # Delete the temporary video file
            os.remove(video_path)

            if litter_detected:
                if penalty > 0:
                    message = f'Litter detected for pet ID {pet_id}. Penalty points added: {penalty}'
                else:
                    message = f'Litter detected for pet ID {pet_id}. No penalty points added'
            else:
                message = f'No litter detected for pet ID {pet_id}. Penalty points added: 0'

            return render_template('index.html', message=message, penalty_points=penalty_points)
        else:
            return render_template('index.html', message='Invalid file format. Only MP4 and AVI files are supported'), 400
    else:
        # Pass penalty points to the HTML template
        return render_template('index.html', penalty_points=penalty_points)

if __name__ == '__main__':
    app.run(debug=True)
