import cv2
from salt_and_pepper import salt_and_pepper
from helper import *

class Model:
    def __init__(self, protofile, caffefile, labels, labelPrefix, labelActions) -> None:
        self.files = [protofile, caffefile]
        self.labels = labels
        self.labelCount = len(labels)
        self.labelPrefix = labelPrefix
        self.labelActions = labelActions
        assert all(act in ("showgreen", "showred", "blur", "noise", "blurnoise") for act in labelActions)

AGE_MODEL = Model(protofile="deploy_age.prototxt", 
                  caffefile="age_net.caffemodel", 
                  labels=["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"],
                  labelPrefix="Age",
                  labelActions=["showgreen"]*8
                  )

AGE_AS_EMOTION_MODEL = Model(protofile="deploy_age.prototxt", 
                  caffefile="age_net.caffemodel", 
                  labels=["DANGER", "DANGER", "DANGER", "SAFE", "SAFE", "DANGER", "DANGER"],
                  labelPrefix="",
                    labelActions=[
                        "blurnoise",
                        "blurnoise",
                        "showred",
                        "showgreen",
                        "showgreen",
                        "showred",
                        "blurnoise",
                        "blurnoise"
                    ])

# https://github.com/GilLevi/AgeGenderDeepLearning/blob/master/EmotiW_Demo.ipynb
EMOTION_CENSORSHIP_MODEL = Model(protofile="deploy.prototxt", 
                      caffefile="EmotiW_VGG_S.caffemodel",
                    #   labels=['Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise'],

                      labels=['DANGER' , 'DANGER' , 'DANGER' , 'SAFE'  , 'SAFE' ,  'DANGER' , 'SAFE'],
                      labelPrefix="",
                      labelActions=[
                            "blurnoise",
                            "showred",
                            "blurnoise",
                            "showgreen",
                            "showgreen",
                            "showred",
                            "showgreen",
                      ])


EMOTION_MODEL = Model(protofile="deploy.prototxt", 
                      caffefile="EmotiW_VGG_S.caffemodel",
                      labels=['Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise'],
                      labelPrefix="",
                      labelActions=["showgreen"]*8)

EMOTION_MODEL_2 = Model(protofile="deploy.prototxt", 
                      caffefile="EmotiW_VGG_S.caffemodel",
                      labels=['Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise'],
                      labelPrefix="",
                      labelActions=[
                            "noise",
                            "showred",
                            "noise",
                            "showgreen",
                            "showgreen",
                            "showred",
                            "showgreen",
                      ])

 
SELECTED_MODEL = AGE_AS_EMOTION_MODEL

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


# returns list of (x, y, width, height) tuples
def get_faces_rects(frame):
    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


# takes in same datatype as 'frame' but cropped to a face (e.g by get_faces_rects())
# returns the index of the class
def apply_model(face):
    # Preprocess the face for age estimation
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    age_predictions = age_net.forward()

    return age_predictions[0].argmax()


def draw_label_effects(frame, class_index, facerect):
    (x, y, w, h) = facerect
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
    if "noise" in action:
        region = frame[y:y+h, x:x+w]
        try:
            frame[y:y+h, x:x+w] = salt_and_pepper(region)
        except:
            pass
    if "blur" in action:
        frame = blurRegion(frame, facerect, blurStrength=15)

    return frame


def process_frame(frame):
    for facerect in get_faces_rects(frame):
        (x, y, w, h) = facerect
        # Extract the face region
        face = frame[y:y+h, x:x+w]


        class_index = apply_model(face)

        frame = draw_label_effects(frame, class_index, facerect)

        

    return frame
