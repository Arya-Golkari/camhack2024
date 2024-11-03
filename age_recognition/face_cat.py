import cv2
from helper import *

class Model:
    def __init__(self, protofile, caffefile, labels, labelPrefix, labelActions) -> None:
        self.files = [protofile, caffefile]
        self.labels = labels
        self.labelCount = len(labels)
        self.labelPrefix = labelPrefix
        self.labelActions = labelActions
        assert all(act in ("showgreen", "showred", "blur", "noise") for act in labelActions)

AGE_MODEL = Model(protofile="deploy_age.prototxt", 
                  caffefile="age_net.caffemodel", 
                  labels=["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"],
                  labelPrefix="Age",
                  labelActions=["showgreen"]*8
                  )

EMOTION_MODEL = Model(protofile="deploy.prototxt", 
                      caffefile="EmotiW_VGG_S.caffemodel",
                    #   labels=['Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise'],

                      labels=['Angry' , 'Disgust' , 'DANGER' , 'SAFE'  , 'SAFE' ,  'Sad' , 'Surprise'],
                      labelPrefix="",
                      labelActions=[
                          "blur",
                          "blur",
                          "showred",
                          "showgreen",
                          "showgreen",
                          "blur",
                          "blur"
                      ])

 
SELECTED_MODEL = EMOTION_MODEL

# face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# age estimation model
# Age and Gender Classification using Convolutional Neural Networks
# Gil Levi and Tal Hassner
# IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG),
# at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015
# (gender model not used)
age_net = cv2.dnn.readNetFromCaffe(
    *(SELECTED_MODEL.files)
)

def process_frame(frame):
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

        class_index = age_predictions[0].argmax()

        prefix = SELECTED_MODEL.labelPrefix
        label = SELECTED_MODEL.labels[class_index]
        if prefix: 
            label = f"{prefix} {label}"

        action = SELECTED_MODEL.labelActions[class_index]

        if action == "showgreen": colour = GREEN
        elif action == "showred": colour = RED


        if action == "showgreen" or action == "showred":
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        elif action == "blur":
            print("blur placeholder")
        elif action == "noise":
            print("noise placeholder")
        else:
            raise Exception("what")

    return frame


MODE_LABELS = {
    "age": ("(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"),
    "shower": ("3 years", "9 hours", "")
}
GOOD_MODE = False


# # Define age ranges for the model's output
AGE_RANGES = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GOOD_AGE_RANGES = ["(0-2)", "(4-6)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]