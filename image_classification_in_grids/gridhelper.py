# DOES NOT WORK :(


from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import cv2


# Load the model
try:
    model = load_model(r"keras_model.h5", compile=False)
    class_names = open(r"labels.txt", "r").readlines()
except OSError as e:
    print(e)
    print("put keras_model.h5 and labels.txt in src/ directory")

# Load the labels

import numpy as np
np.set_printoptions(suppress=True)


TARGETSIZE = (896, 896)
GRIDDIM = (4,4)
assert TARGETSIZE[0] % GRIDDIM[0] == 0 and TARGETSIZE[1] % GRIDDIM[1] == 0

# size of each section
WIDTH = TARGETSIZE[0]//GRIDDIM[0]
HEIGHT = TARGETSIZE[1]//GRIDDIM[1]

# returns list of tuples (subframe, x, y)
def splitIntoGridSections(frame):
    return [
        (frame[gridY*HEIGHT:(gridY+1)*HEIGHT, gridX*WIDTH:(gridX+1)*WIDTH],
         gridX*WIDTH, 
         gridY*HEIGHT)
        for gridX in range(GRIDDIM[0])
        for gridY in range(GRIDDIM[1])
    ]


# returns bool which is true if (an orange|a gun) is detected
# may need to be changed to return tuple (detectedBool, rect)

def applymodel(frame):
    # i didnt refactor this part but i hope its right
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGgayR2RGB))

    # resizing the image to be at least 224x224 and then cropping from the center
    # size = (250, 250)
    # image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    # class_name = class_names[index]
    # confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)


    # effectively if confidence_score > 0.5 
    # could change it to check actual confidence if needed
    return index == 0  


# frame = entire frame, rect = (leftx, topy, width, height)
# blurStrength is actually kernel size and has to be odd
# returns new frame
def blurRegion(frame, rect, blurStrength=63):
    (leftx, topy, width, height) = rect
    rightx = leftx+width
    bottomy = topy+height

    roi = frame[topy:bottomy, leftx:rightx]

    blurred_roi = cv2.GaussianBlur(roi, (blurStrength, blurStrength), 0)

    frame[topy:bottomy, leftx:rightx] = blurred_roi
    return frame


def processFrame(frame):
    for subframe, x, y in splitIntoGridSections(frame):
        
        detected = applymodel(subframe)

        if detected:
            frame = blurRegion(frame, x, y, WIDTH, HEIGHT)

            
    
    return frame

