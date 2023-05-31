import cv2 as cv
import numpy as np
import os

path = 'C:/Users/Raffie/Desktop/UA Kidneys/W34.70047.tif'

def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

## Load image, convert to HSV
image = cv.imread(path, cv.IMREAD_COLOR)
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

## Perform HSV threshold, return binary image with valid kidney pixels highlighted
hsv_color1 = np.asarray([150, 50, 50])
hsv_color2 = np.asarray([175, 192, 180])
color_thresh = cv.inRange(hsv, hsv_color1, hsv_color2)

## Find contours in the color thresh. Draw only the largest one and fill it. Reduces noise.
contour,hierarchy = cv.findContours(color_thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
validKidney = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
biggest_contour = max(contour, key=lambda item: cv.contourArea(item))
cv.drawContours(validKidney,[biggest_contour],-1,(255,255,255),cv.FILLED)

index = np.where(validKidney > 0)
coordinates = list(zip(index[0],index[1]))

gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
whitespace = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
for coordinate in coordinates:
    if gray[coordinate[0],coordinate[1]] >= 130:
        whitespace[coordinate[0],coordinate[1]] = 255

validKidney_pix = cv.countNonZero(validKidney)
whitespace_pix = cv.countNonZero(whitespace)
percent_whitespace = (whitespace_pix/validKidney_pix) * 100

## Alternatively, draw ALL contours and fill them.
#for cnt in contour:
    #output = cv.drawContours(validKidney,[cnt],0,255,-1)

## UNFINISHED --> maybe find whitespace by drawing contours of a smaller hierarchy.
#whitespace = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
#for cnt,hier in zip(contour,hierarchy):
    #if hier[1] == 1:
        #output2 = cv.drawContours(whitespace,[cnt],0,255,-1)

## Can also reduce noise by erosion followed by dilation. Unnecessary.
#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
#opening = cv.morphologyEx(validKidney, cv.MORPH_OPEN, kernel)

## UNFINISHED --> Color threshold for whitespace. Too much noise.
#hsv_color3 = np.asarray([0,0,165])
#hsv_color4 = np.asarray([40,25,180])
#color_thresh2 = cv.inRange(hsv, hsv_color3, hsv_color4)

## UNFINISHED --> methods of determining total volume without color.
#gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#adaptive_gauss = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,21,6)
#adaptive_mean = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,15,5)

## UNFINISHED --> Floodfill Method

empty_channel = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
figure = cv.merge((empty_channel, validKidney, whitespace))
figure[whitespace > 0] = (0,0,255)

cv.imshow('Original Image',rescaleFrame(image))
#cv.imshow('Color Threshold', rescaleFrame(color_thresh))
cv.imshow('Contours', rescaleFrame(validKidney))
cv.imshow('Whitespace',rescaleFrame(whitespace))
cv.imshow('Figure',rescaleFrame(figure))
cv.waitKey(0)

print(percent_whitespace)
