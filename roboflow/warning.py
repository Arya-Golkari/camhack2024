from gtts import gTTS
from playsound import playsound

WARNING = "BEWARE OF ORANGE"
LANGUAGE = "en"
WARNING_AUDIO = r"OrangeWarning.mp3"
MODEL = r"keras_model.h5"
LABELS = r"labels.txt"
COOLDOWN_TIME = 5

GREEN = (0, 255, 0)

# Save the warning audio
speech = gTTS(text=WARNING, lang=LANGUAGE, slow=True)
speech.save(WARNING_AUDIO)