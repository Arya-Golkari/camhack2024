import face_cat

import cv2

# WEBCAM_IP = "10.252.84.251"
# WEBCAM_PORT = "4747"

WINDOW_TITLE = "Video Feed blah blah"

# Initialize video capture
video = cv2.VideoCapture(0)
# address = f"https://{WEBCAM_IP}:{WEBCAM_PORT}/video"
address = 0
video.open(address)

while True:
    check, frame = video.read()
    if not check:
        break

    processedFrame = face_cat.process_frame(frame)

    # Show the frame
    cv2.imshow(WINDOW_TITLE, processedFrame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
video.release()
cv2.destroyAllWindows()