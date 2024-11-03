import inference
import cv2
import time
from playsound import playsound

from salt_and_pepper import salt_and_pepper as create_noise
from warning import WARNING_AUDIO, COOLDOWN_TIME, WARNING, GREEN

MODEL = r"detection-klr0p/1"

camera = cv2.VideoCapture(0)
camera.open(0)

model = inference.get_model(MODEL)

detected_orange = False
cooldown = False
detection_start = time.time()

while True:
    check, frame = camera.read()

    if not check:
        break

    inferences = model.infer(frame)
    predictions = inferences[0].predictions

    for object in predictions:
        if object.class_name == "choc":
            x = int(object.x)
            y = int(object.y)
            w = int(object.width)
            h = int(object.height)
            
            region = frame[y-(h//2):y+(h//2), x-(w//2):x+(w//2)]

            # Noise
            region = create_noise(region)

            # Gaussian blur
            try:
                frame[y-(h//2):y+(h//2), x-(w//2):x+(w//2)] = cv2.GaussianBlur(region, (33, 33), 0)
            except:
                pass

        elif object.class_name == "Oranges" and object.confidence > 0.8:
            cv2.putText(frame, WARNING, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        
            if not cooldown:
                detected_orange = True

    # Show the frame
    cv2.imshow("Arkangel", frame)
    
    # Break loop on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if detected_orange:
        playsound(WARNING_AUDIO)
        detected_orange = False
        cooldown = True
        detection_time = time.time()

    if cooldown:
        if time.time() - detection_time > COOLDOWN_TIME:
            cooldown = False

# Release resources
camera.release()
cv2.destroyAllWindows()