import tkinter as tk
from PIL import ImageTk, Image
from win32api import GetMonitorInfo, MonitorFromPoint

# Get window size
monitorInfo = GetMonitorInfo(MonitorFromPoint((0,0)))
workArea = monitorInfo.get("Work")
windowWidth = workArea[2]
windowHeight = workArea[3]

# Startup window configs
root = tk.Tk()
root.title("Test")
root.geometry("{}x{}+0+0".format(windowWidth,windowHeight))

# Load in images (not buttons)
images = {}

def load_image(name, image_name):
    image = Image.open("Interface\\" + image_name).resize((100, 100))
    print("Interface\\" + image_name)
    img = ImageTk.PhotoImage(image)
    images[name] = img
    return img


# Connect to video stream after click eye
def click_eye():
    text.config(text= "You have clicked Me...")

def click_profile():
    pass


# Display images
history_img = load_image("history", "history_icon.png")
history_label = tk.Label(root, image=history_img)
history_label.place(x=windowWidth/4-100, y=windowHeight/2-100)

sightstream_img = load_image("sightstream", "sightstream_icon.png")
sightstream_label = tk.Label(root, image=sightstream_img)
sightstream_label.place(x=windowWidth/3-100, y=windowHeight*0.65)

gps_img = load_image("gps", "gps_icon.png")
gps_label = tk.Label(root, image=gps_img)
gps_label.place(x=windowWidth*(2/3), y=windowHeight*0.15)

settings_img = load_image("settings", "settings_icon.png")
settings_label = tk.Label(root, image=settings_img)
settings_label.place(x=windowWidth*0.75, y=windowHeight/2-100)

vitals_img = load_image("vitals", "vitals_icon.png")
vitals_label = tk.Label(root, image=vitals_img)
vitals_label.place(x=windowWidth*(2/3), y=windowHeight*0.65)

# Eye button
eye_image = Image.open("Interface\eye_icon.png").resize((200, 120))
eye_img = ImageTk.PhotoImage(eye_image)
eye_label = tk.Label(image=eye_img)
eye_button = tk.Button(root, image=eye_img, command=click_eye,
                       borderwidth=0)
eye_button.place(x=windowWidth/2-100, y=windowHeight/2-100)

# Profile button
profile_image = Image.open("Interface\profile_icon.png").resize((100, 100))
profile_img = ImageTk.PhotoImage(profile_image)
profile_label = tk.Label(image=profile_img)
profile_button = tk.Button(root, image=profile_img, command=click_profile,
                       borderwidth=0)
profile_button.place(x=windowWidth/3-100, y=windowHeight*0.15)


#
text= tk.Label(root, text= "")
text.pack(pady=30)

root.mainloop()
