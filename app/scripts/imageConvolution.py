import cv2 
import numpy as np
from . import imageProcessing as imgProc

# In order to get the best result, the image used should be GREYSCALE
def convolve(image, kernel, padding=0, strides=1):
    # Apply cross correlation to our kernel
    kernel = np.flipud(np.fliplr(kernel))

    # Compute the matrix size of our outputted image
    # This is done by gathering the shapes of the kernel, image and padding
    xKernShape = kernel.shape[0] 
    yKernShape = kernel.shape[1] 
    xImgShape = image.shape[0] 
    yImgShape = image.shape[1]

    # We can then get the shape of the Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)

    # New matrix with deduced dimensions 
    output = np.zeros((xOutput, yOutput))

    # Apply padding to each side of the matrix
    # This is especially important, since the method relies on the padding being even
    # We check whether or not the padding is 0, and if it's not, we apply operations to correct it.
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else: 
        imagePadded = image

    # Iterate through the image
    for y in range(image.shape[1]):
        # Exit
        if y > image.shape[1] - yKernShape: 
            break
        # Check if end of image on y-axis is reached
        # Convolution is completed once we reach the bottom right of the image matrix
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Check if kernel is out of bounce (reached the very right of the image).
                # Once reached, break the x-loop and move down 1 on the y-axis to repeat the convolution process
                if x > image.shape[0] - xKernShape: 
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break
    return output