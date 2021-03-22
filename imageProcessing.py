import cv2 # works with images
from matplotlib import pyplot # for visualisation
import numpy as np

# visualisation function with color (original image)
def original(image, showPrint=True):
    # image view
    #pyplot.imshow(image)
    pyplot.imshow(image)# pixel view with grayscale
    pyplot.show()
    if showPrint == True:
        print("Image : Original")
        print('image size: ', image.shape)
        print('pixel matrix:\n', image)

# visualisation function with greyscale
def greyscale(image, showPrint=True):
    # image view
    #pyplot.imshow(image)
    pyplot.imshow(image, cmap="gray", vmin=0, vmax=255)# pixel view with grayscale
    pyplot.show()
    if showPrint == True:
        print("Image : Greyscale")
        print('image size: ', image.shape)
        print('pixel matrix:\n', image)