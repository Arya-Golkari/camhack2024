import numpy as np

def salt_and_pepper(image):
    output = np.zeros(image.shape, np.uint8)
    block = 3
    
    for i in range(image.shape[0] // block):
        for j in range(image.shape[1] // block):
            rdn = np.random.random()
            
            if rdn < 0.2:
                out = 0
            elif rdn > 0.8:
                out = 255
            else:
                out = image[i*block:(i+1)*block, j*block:(j+1)*block]
            
            output[i*block:(i+1)*block, j*block:(j+1)*block] = out
    
    return output