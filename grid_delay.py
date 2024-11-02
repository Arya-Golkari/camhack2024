from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2, time

# Initialize video capture
video = cv2.VideoCapture(0)
address = r"https://10.252.84.251:8080/video"
video.open(address)

# Replace these with your rectangle's top-left (x, y) and width (w) and height (h)
# x, y, w, h = 100, 100, 200, 200

# Load the model
model = load_model(r"keras_model.h5", compile=False)

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

    # Extract the region to be blurred
    # roi = frame[y:y+h, x:x+w]

    subframes = extract(frame)

    # for i in range(16):
    #     cv2.imwrite(rf"images/{i}.jpg", subframes[i])

    for i in range(16):
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        # image = Image.open(rf"images/{i}.jpg").convert("RGB")
        image = Image.fromarray(cv2.cvtColor(subframes[i], cv2.COLOR_BGR2RGB))

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
        class_name = class_names[index]
        confidence_score = prediction[0][index]

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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.2)

# Release resources
video.release()
cv2.destroyAllWindows()