import cv2 as cv
import numpy as np
import os

# If the images are too big for your screen, reduce "scale" in the function below
def rescaleFrame(frame, scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# IMPORTANT: Change this filepath to the image you want to analyse
image = cv.imread('C:/Users/Raffie/Desktop/Collagen/10x0124.tif', cv.IMREAD_COLOR)

hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
hsv_color1 = np.asarray([115, 50, 20])
hsv_color2 = np.asarray([170, 255, 255])

mask = cv.inRange(hsv, hsv_color1, hsv_color2)

# total pixels in the image
x = image.shape[1]
y = image.shape[0]
total_pix = x*y

# number of collagen pixels
collagen = cv.countNonZero(mask)

# amount of whitespace
hsv_color3 = np.asarray([0, 0, 170])
hsv_color4 = np.asarray([180, 45, 255])
white = cv.inRange(hsv, hsv_color3, hsv_color4)
whitespace = cv.countNonZero(white)

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
