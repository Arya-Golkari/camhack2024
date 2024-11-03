import cv2
from ghelper import *

GOOD_MODE = True

# Initialize video capture
video = cv2.VideoCapture(0)

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained age estimation model (download required files beforehand)
age_net = cv2.dnn.readNetFromCaffe(
    "deploy_age.prototxt", 
    "age_net.caffemodel"
)

# Define age ranges for the model's output
AGE_RANGES = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GOOD_AGE_RANGES = ["(0-2)", "(4-6)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]


# Loop through video frames
while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Preprocess the face for age estimation
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(blob)
        age_predictions = age_net.forward()
        age = AGE_RANGES[age_predictions[0].argmax()]

        # Display the age range on the frame
        if GOOD_MODE:
            if age in GOOD_AGE_RANGES:
                label = "GOOD"
                colour = gen_rgb()
            else:
                label = "NOT GOOD"
                colour = RED
        else:
            label = f"Age: {age}"
            colour = GREEN

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

    # Show the frame
    cv2.imshow("Age Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
