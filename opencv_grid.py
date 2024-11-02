from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
import cv2, time

# Initialize video capture
video = cv2.VideoCapture(0)
address = r"https://10.252.84.251:8080/video"
video.open(address)

# Load the model
model = load_model(r"keras_opencv_model.h5", compile=False)

# Load the labels
class_names = open(r"labels.txt", "r").readlines()

def extract(frame):
    subframes = []
    for i in range(4):
        for j in range(4):
            subframes.append(frame[(i*224):((i+1)*224), (j*224):((j+1)*224)])
    
    return subframes

while True:
    check, frame = video.read()
    frame = cv2.resize(frame, (896, 896))
    
    if not check:
        break

    subframes = extract(frame)

    for i in range(16):
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Resize the raw image into (2224-height, 224-width) pixels
        image = cv2.resize(image, (224, 224))

        # Make the image a numpy array and reshape it to the models input shape
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalise the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        # confidence_score = prediction[0][index]

        # Print prediction and confidence score
        # print("Class:", class_name[2:], end="")
        # print("Confidence Score:", confidence_score)
        
        if index == 0:
            roi = (i // 4) * 224, (i % 4) * 224, 224, 224
            y = (i // 4) * 224
            x = (i % 4) * 224
            w = 224
            h = 224
            roi = frame[y:y+h, x:x+w]
        
            # Apply Gaussian blur to the region
            blurred_roi = cv2.GaussianBlur(roi, (63, 63), 0)  # Adjust (15, 15) for more/less blur

            # Place the blurred region back into the frame
            frame[y:y+h, x:x+w] = blurred_roi
    
    # Show the frame
    cv2.imshow("Video with Blurred Region", frame)
    
    # Break loop on 'q' key press
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # time.sleep(0.2)

# Release resources
video.release()
cv2.destroyAllWindows()