import cv2
import time

# Initialize video capture
video = cv2.VideoCapture(0)
address = "https://10.252.84.251:8080/video"
video.open(address)

# Define the rectangle coordinates
x, y, w, h = 100, 100, 200, 200  # Replace these with your rectangle's top-left (x, y) and width (w) and height (h)

while True:
    check, frame = video.read()
    
    if not check:
        break

    # Extract the region to be blurred
    roi = frame[y:y+h, x:x+w]
    
    # Apply Gaussian blur to the region
    blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)  # Adjust (15, 15) for more/less blur

    # Place the blurred region back into the frame
    frame[y:y+h, x:x+w] = blurred_roi
    
    # Show the frame
    cv2.imshow("Video with Blurred Region", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
