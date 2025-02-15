import time, cv2


# frame = entire frame, rect = (leftx, topy, width, height)
# blurStrength is actually kernel size and has to be odd
# returns new frame
def blurRegion(frame, rect, blurStrength=63):
    (leftx, topy, width, height) = rect
    rightx = leftx+width
    bottomy = topy+height

    roi = frame[topy:bottomy, leftx:rightx]

    blurred_roi = cv2.GaussianBlur(roi, (blurStrength, blurStrength), 0)

    frame[topy:bottomy, leftx:rightx] = blurred_roi
    return frame


# for slaytastic rainbow effect (generated by chadGPT) (unused D:)
def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB.
    
    Args:
        h (float): Hue (0-360)
        s (float): Saturation (0-1)
        v (float): Value (0-1)
    
    Returns:
        tuple: RGB values (r, g, b) in the range [0, 255].
    """
    h = h / 60
    c = v * s
    x = c * (1 - abs(h % 2 - 1)) # crazy how mod works for floats
    m = v - c

    if 0 <= h < 1:
        r, g, b = c, x, 0
    elif 1 <= h < 2:
        r, g, b = x, c, 0
    elif 2 <= h < 3:
        r, g, b = 0, c, x
    elif 3 <= h < 4:
        r, g, b = 0, x, c
    elif 4 <= h < 5:
        r, g, b = x, 0, c
    elif 5 <= h < 6:
        r, g, b = c, 0, x

    r = (r + m) * 255
    g = (g + m) * 255
    b = (b + m) * 255

    return int(r), int(g), int(b)


def time_to_hue(t):
    return round(t*256)%360

def gen_rgb():
    t = time.time()
    hue = time_to_hue(t)
    rgb = hsv_to_rgb(hue, 1, 1)
    # print(t, hue, rgb)
    return rgb


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)