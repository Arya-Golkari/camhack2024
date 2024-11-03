from keras.models import load_model
import cv2
from ghelper import *
from gtts import gTTS
import numpy as np
from playsound import playsound

# Initialize video capture
address = 0
# address = r"https://10.252.84.251:8080/video"
video = cv2.VideoCapture(address)

WARNING = "ROGUE ORANGE"
LANGUAGE = "en"
WARNING_AUDIO = r"OrangeWarning.mp3"
MODEL = r"../keras_model.h5"
LABELS = r"../labels.txt"

# Save the warning audio
speech = gTTS(text=WARNING, lang=LANGUAGE, slow=True)
speech.save(WARNING_AUDIO)

# Load the model
model = load_model(MODEL, compile=False)

# Load the labels
class_names = open(LABELS, "r").readlines()

# Loop through video frames
while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Fit the model
    frame = cv2.resize(frame, (224, 224))
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)

    if index == 0:
        cv2.putText(frame, WARNING, (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
        playsound(WARNING_AUDIO)

    # Show the frame
    cv2.imshow("Orange Warning", frame)

    # Break loop on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()