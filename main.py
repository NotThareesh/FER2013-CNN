from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

def processImage(faceImage):
    face_resized = cv2.resize(faceImage, (48, 48))
    face_normalized = face_resized / 255.0

    face_input = np.expand_dims(face_normalized, axis=-1)  # Add channel dimension for grayscale (48x48x1)
    face_input = np.expand_dims(face_input, axis=0) # Add batch dimension (1x48x48x1)

    return face_input

def draw_and_find_emotion(imagePath):
    model = load_model("FERClassifier.keras")

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    img = cv2.imread(imagePath)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    # If Face Detected
    if len(face) >= 1:
        x,y,w,h = face[0]
        face_roi = gray_image[y : y + h, x : x + w]
        face_input = processImage(face_roi)
    
        # Predict emotion
        emotion_pred = model.predict(face_input)
        emotion_index = np.argmax(emotion_pred)

        # Get the predicted emotion label
        emotion = emotion_labels[emotion_index]

        # Draw Rectangle                    
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # Save the Image
        output_path = f"static/saved_images/{imagePath[9:]}"
        cv2.imwrite(output_path, img)

        return emotion
    else:
        return -1

@app.route("/")
def main():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['uploaded_file']

        if uploaded_file.filename != '':
            imagePath = f"./uploads/{uploaded_file.filename}"

            uploaded_file.save(imagePath)
            emotion = draw_and_find_emotion(imagePath=imagePath)

            return render_template('upload_page.html', image=f"static/saved_images/{imagePath[10:]}", text=emotion)

        else:
            return "No File Selected!", 404

app.run(debug=True, port=8080)