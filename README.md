# FER2013 CNN Emotion Recognition

A Flask-based web application for real-time emotion recognition. The project uses a trained deep learning model to classify emotions from facial images obtained from the FER Dataset. Currently, it can detect emotions like Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral.
In addition to this, this project also features a real-time emotion recognition app that can be run through your webcam. Currently this CNN predicts with **66.04%** accuracy on the **test dataset**.

---
    
## Features
- **Upload Images**: Users can upload an image, and the app detects the dominant emotion in the face(s).  
- **Real-Time Emotion Detection**: Detect emotions in real time using your webcam.  
- **User-Friendly UI**: Simple, modern UI built with Bootstrap for easy interaction.  
- **Custom Models**: Leverages a custom-trained deep learning model for FER2013 dataset.  

---

## Tech Stack  
- **Backend**: Flask (Python)  
- **Frontend**: HTML, CSS, Bootstrap  
- **Machine Learning**: TensorFlow and OpenCV

---

## Setup Instructions  

### 1. Clone the Repository  
```bash
git clone https://github.com/NotThareesh/FER2013-CNN.git
cd FER2013-CNN
```
### 2. Create Virtual Environment
```bash
python -m virtualenv venv
```
##### For venv or virtualenv on Linux/Mac:
```bash
source venv/bin/activate
```
##### For venv or virtualenv on Windows (Command Prompt):
```bash
venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Download the FER Dataset from Kaggle
- Dataset URL: [FER2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- Download the dataset as a ZIP file.
- Ensure that the file structure of the project looks exactly as [Folder Structure](#folder-structure)
### 5. Create Folders for running Flask App
- Create folder named ```uploads``` in root directory.
- Create folder named ```saved_images``` in ```static``` directory.
- Ensure that the file structure of the project looks exactly as [Folder Structure](#folder-structure)

### 6. Run the Application
##### For uploading images and analyzing emotions:
```bash
python main.py
```
##### For real-time emotion detection:
```bash
python realtime_camera.py
```

---

## Folder Structure
```
FER2013-CNN/
│
├── data/                    # FER2013 training and testing data
│   ├── train/
│   └── test/
│
├── static/                  # Static files (CSS, images, etc.)
│   ├── saved_images/        # Modified user uploaded images will save here
│   ├── logo.png             # App logo
│   └── styles.css           # Custom styles
│
├── templates/               # HTML templates
│   ├── index.html           # Home page
│   └── upload_page.html     # Emotion Display page
│
├── LICENSE                  # License for the Repository
├── cnn.ipynb                # Model Training in Jupyter Notebook
├── FERClassifier.keras      # Emotion classification model 
├── main.py                  # Flask application
├── realtime_camera.py       # Real-time emotion detection script
├── requirements.txt         # Python dependencies
└── README.md                # Project description
```

---

## Model Accuracy
```python
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
print(f"Test Accuracy (Direct Evaluation): {test_accuracy * 100:.2f}%")
```
```
113/113 ━━━━━━━━━━━━━━━━━━━━ 8s 68ms/step - accuracy: 0.6602 - loss: 1.6984
Test Accuracy (Direct Evaluation): 66.04%
```

---

## Future Improvements
- Responsive and more customized UI/UX experience
- Make the ML Model learn from incorrectly classified user uploaded images

---

## Contributions
Feel free to contribute by opening issues and dropping your suggestions.

---

## License
This project is licensed under the MIT License.