import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('FERClassifier.keras')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List of emotions
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Initialize webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop face from the frame
        face_roi = gray[y:y+h, x:x+w]

        # Resize the face image to normalize it
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0

        # Reshape to match model input shape
        face_input = np.expand_dims(face_normalized, axis=-1)  # Add channel dimension for grayscale (48x48x1)
        face_input = np.expand_dims(face_input, axis=0)  # Add batch dimension (1x48x48x1)

        # Predict emotion
        emotion_pred = model.predict(face_input)
        emotion_index = np.argmax(emotion_pred)

        # Get the predicted emotion label
        emotion = emotion_labels[emotion_index]

        # Display the emotion on the frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Emotion Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()