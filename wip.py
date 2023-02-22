import cv2 as cv
import numpy as np
import os

def rescaleFrame(frame, scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

images = []
path = 'C:/Users/Raffie/Desktop/Collagen'
for filename in os.listdir(path):
    img = cv.imread(os.path.join(path,filename))
    if img is not None:
        images.append(img)

hsv = []
masks = []
hsv_color1 = np.asarray([115, 50, 20])
hsv_color2 = np.asarray([170, 255, 255])
for image in images:
    hsv.append(cv.cvtColor(image, cv.COLOR_BGR2HSV))
for img in hsv:
    mask = cv.inRange(img, hsv_color1, hsv_color2)
    masks.append(mask)

# total pixels in the first image (assume all images are same resolution, 2048x1536)
x = images[0].shape[1]
y = images[0].shape[0]
total_pix = x*y

# number of collagen pixels
collagen = []
for mask in masks:
    collagen.append(cv.countNonZero(mask))

# amount of whitespace
hsv_color3 = np.asarray([0, 0, 170])
hsv_color4 = np.asarray([180, 45, 255])
whites = []
for img in hsv:
    whites.append(cv.inRange(img, hsv_color3, hsv_color4))
whitespace = []
for img in whites:
    whitespace.append(cv.countNonZero(img))

# calculating final percentages
percent_collagen = (collagen/(total_pix-whitespace))*100
print(percent_collagen)

image_resized = rescaleFrame(image)
mask_resized = rescaleFrame(mask)
white_resized = rescaleFrame(white)
cv.imshow('Input', image_resized)
cv.imshow('Collagen', mask_resized)
cv.imshow('Whitespace', white_resized)
cv.waitKey(0)
