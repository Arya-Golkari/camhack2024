import numpy as np

def salt_and_pepper(image):
    # output = np.zeros(image.shape, np.uint8)
    output = image
    block = 5
    
    for i in range(image.shape[0] // block):
        for j in range(image.shape[1] // block):
            rdn = np.random.random()
            
            if rdn < 0.2:
                output[i*block:(i+1)*block, j*block:(j+1)*block] = 0
            elif rdn > 0.8:
                output[i*block:(i+1)*block, j*block:(j+1)*block] = 255
            
            
    
    return output