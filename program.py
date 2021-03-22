import skimage.measure
import numpy as np
import cv2

from files.scripts import imageProcessing as imgProc
from files.scripts import imageConvolution as imgConv


#from imageConvolution import test as test
img = cv2.imread('files/img/img.jpg', 0)


#===: Testing Functions :===#

# Show original image
print("\n=== Print Original Image ===")
imgProc.original(img)

# Show greyscale image
print("\n=== Print Greyscale Image ===")
imgProc.greyscale(img)
cv2.imwrite('files/out/img-greyscale.jpg', img)

# Resize the image
print("\n=== Print Resized Image ===")
SIZE = 32
img = cv2.resize(img, (SIZE,SIZE))
cv2.imwrite('files/out/img32x32.jpg', img)
imgProc.greyscale(img)


#===: ASSIGNMENT :===#

# Part A: Enter a grey scale pixel matrix [32, 32] with random colour values.
print("\n=== Part A ===")

# Take the resized image (32x32) and randomize it's color values.
imgRandom = cv2.randu(img, 0, 255)
imgProc.greyscale(imgRandom)
cv2.imwrite('files/out/img32x32-random.jpg', imgRandom)

# Part B: Filter it by convolutional multiplication with a sparse matrix for discovering vertical lines.
print("\n=== Part B ===")

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) # Edge Detection Kernel
img = cv2.imread('files/img/img.jpg', 0) # Resetting image
imgConvolved = imgConv.convolve(img, kernel, padding=1) # Convolve and Save Output
cv2.imwrite('files/out/img-convolved.jpg', imgConvolved) # Save image in the output folder
imgProc.greyscale(imgConvolved, False)

# Part C & D: Reshape it by applying max-pool method with size [2x2] and stride of 2
print("\n=== Part C ===")

img = cv2.imread('files/img/img.jpg', 0) # Resetting image
imgReshape = skimage.measure.block_reduce(img, (2,2), np.max) # Reduce the image with a max pool of 2-2
imgProc.original(imgReshape)
cv2.imwrite('files/out/img-reshaped.jpg', imgReshape) # Save image in the output folder