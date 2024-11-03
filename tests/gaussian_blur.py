import cv2
import time

# Initialize video capture
video = cv2.VideoCapture(0)
address = "https://10.252.84.251:8080/video"
video.open(address)

# Define the rectangle coordinates
x, y, w, h = 100, 100, 200, 200  # Replace these with your rectangle's top-left (x, y) and width (w) and height (h)

# for now before fetching coords
blur_coords = [[500, 500, 200, 200], [50, 50, 100, 200], [150, 50, 50, 100]]

while True:

    # fetch coords func returns list coords to blur_coords

    check, frame = video.read()

    
    if not check:
        break
    

    # Apply Gaussian blur to the regions

    for i in range (len(blur_coords)):
        coord = blur_coords[i]
        x, y, w, h = coord[0], coord[1], coord[2], coord[3]
        roi = frame[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (69, 69), 0)
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
