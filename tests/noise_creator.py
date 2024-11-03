import cv2
import numpy as np
import random

# Initialize video capture
address = 0
# address = r"https://10.252.84.251:8080/video"
video = cv2.VideoCapture(address)

def gauss_noise(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + image * gauss
    return noisy_image

def salt_and_pepper(image):
    output = np.zeros(image.shape, np.uint8)
    block = 5
    
    for i in range(image.shape[0] // block):
        for j in range(image.shape[1] // block):
            rdn = random.random()
            
            if rdn < 0.2:
                out = 0
            elif rdn > 0.8:
                out = 255
            else:
                out = image[i*block:(i+1)*block, j*block:(j+1)*block]
            
            output[i*block:(i+1)*block, j*block:(j+1)*block] = out
    
    return output

# Loop through video frames
while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Create noise
    frame = salt_and_pepper(frame)

    # Show the frame
    cv2.imshow("Orange Warning", frame)

    # Break loop on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()