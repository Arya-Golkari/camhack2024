import cv2
from ghelper import *
from gtts import gTTS
from playsound import playsound

# Initialize video capture
video = cv2.VideoCapture(0)
#address = r"https://10.2#52.84.251:8080/video"
#video = cv2.VideoCapture(address)

hasWarned = False
warning = "Stranger Danger"
language = "en"
speech = gTTS(text=warning, lang=language, slow=True)
speech.save("Warning.mp3")

# Loop through video frames
while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    
    if not ret:
        break
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.putText(frame, "STRANGER DANGER", (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)

    # Show the frame
    cv2.imshow("Age Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if key == ord('s'):
        playsound("Warning.mp3")

# Release resources
video.release()
cv2.destroyAllWindows()