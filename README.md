# Sleep_detection_MachineLearning

Overview
This project is a Sleep Detection System using machine learning and computer vision techniques to detect blink frequency and predict sleep states based on facial data. The system uses the FaceMeshModule from cvzone to detect facial landmarks and a RandomForestClassifier model to predict sleep patterns in real-time. It logs blink counts and predicts the sleep state based on the average ratio of eye landmarks.

Features
Blink Detection: Detects the blink frequency based on facial landmarks.
Sleep Prediction: Predicts sleep states (e.g., light sleep, deep sleep) using a trained machine learning model.
Real-time Analysis: Provides real-time blink detection and sleep prediction using a webcam.
Data Logging: Logs the time, blink count, and eye ratio to a CSV file for further analysis.
Prerequisites
To run the project, you need the following libraries installed:

opencv-python
cvzone
pandas
numpy
sklearn
datetime
pickle
You can install the required dependencies using the following command:


pip install opencv-python cvzone pandas numpy scikit-learn

How It Works
Face Mesh Detection: The system uses the FaceMeshModule from cvzone to detect facial landmarks around the eyes.
Eye Ratio Calculation: The vertical and horizontal distances between specific eye landmarks are used to calculate the eye aspect ratio.
Blink Count: Blinks are detected when the eye aspect ratio falls below a threshold value.
Sleep Prediction: Based on the average eye aspect ratio, the system predicts the user's sleep state (e.g., "Not Sleeping", "Light Sleep", "Deep Sleep").
Data Logging: Each blink event and the corresponding eye ratio are logged in a CSV file along with the time.
Model Training
The machine learning model used in this project is a RandomForestClassifier trained on blink count and eye aspect ratio data. The training process involves:

Reading the blink data from a CSV file.
Hyperparameter tuning using GridSearchCV for optimal model performance.
Saving the trained model as a pickle file.
You can train the model yourself by running the train_model.py file, which tunes the model's hyperparameters and saves it for use in real-time detection.

How to Train the Model:
python train_model.py
Running the Project
Clone the repository:


git clone <repository-url>
cd sleep_detection
Run the real-time blink detection:


python real_time_blink_detection.py
The system will start capturing video from your webcam and display:

Blink Count
Sleep Prediction (e.g., "Not Sleeping", "Light Sleep", "Deep Sleep")

Project Structure
sleep_detection/
├── model/                          # Directory where the trained model is saved
│   └── random_forest_model.pkl      # Trained RandomForestClassifier model
├── blink_data.csv                   # CSV file to store blink data
├── real_time_blink_detection.py     # Main script for real-time blink detection and sleep prediction
├── train_model.py                   # Script to train the RandomForestClassifier model
└── README.md                        # This README file

Future Improvements
Improved Sleep Stage Detection: Implement more advanced models to detect finer sleep stages (REM, NREM).
Longer-Term Data Analysis: Incorporate additional data points for more comprehensive sleep analysis.
Emotion Detection Integration: Expand the system to detect emotional states based on facial cues.

Contributing
Feel free to open issues or create pull requests to improve this project. Contributions are welcome!



