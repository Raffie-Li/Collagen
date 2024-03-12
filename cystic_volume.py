import cv2 as cv
import numpy as np
import time
import os

start_time = time.time()
path = 'C:/Users/Raffie/Desktop/UA Kidneys/W34.70047.tif' # Give path to your image on this line.

def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

## Loads image, converts to HSV
image = cv.imread(path, cv.IMREAD_COLOR)
filename = os.path.basename(path)
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

## Performs HSV threshold, returns binary image with valid kidney pixels highlighted
hsv_color1 = np.asarray([150, 50, 50]) # Lower HSV bound: hue, saturation, value
hsv_color2 = np.asarray([175, 192, 180]) # Upper HSV bound: hue, saturation, value
color_thresh = cv.inRange(hsv, hsv_color1, hsv_color2)

## Finds contours in the color thresh. Draws only the largest one and fill it. Eliminates noise.
contour,hierarchy = cv.findContours(color_thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
validKidney = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
biggest_contour = max(contour, key=lambda item: cv.contourArea(item))
cv.drawContours(validKidney,[biggest_contour],-1,(255,255,255),cv.FILLED)

## Create list of kidney pixel coordinates.
coordinates = list(zip(np.nonzero(validKidney)[0],np.nonzero(validKidney)[1]))

## Find whitespace but only if pixel is on kidney.
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
whitespace = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
for coordinate in coordinates:
    if gray[coordinate[0],coordinate[1]] >= 130: # Increase this value if too much whitespace, and vice versa
        whitespace[coordinate[0],coordinate[1]] = 255

## Calculate percentage cystic volume.
validKidney_pix = cv.countNonZero(validKidney)
whitespace_pix = cv.countNonZero(whitespace)
percent_whitespace = (whitespace_pix/validKidney_pix) * 100

## Generate Figures.
empty_channel = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
figure = cv.merge((empty_channel, validKidney, whitespace))
figure[whitespace > 0] = (0,0,255)

print("Time Elapsed: {:.2f}s".format(time.time() - start_time))
print('Filename: ' + filename)
print('Whitespace Pixels: ' + str(whitespace_pix))
print('Kidney Pixels: ' + str(validKidney_pix))
print('Percent Whitespace: ' + str(percent_whitespace))

cv.imshow(filename,rescaleFrame(image))
#cv.imshow('Color Threshold', rescaleFrame(color_thresh))
#cv.imshow('Contours', rescaleFrame(validKidney))
#cv.imshow('Whitespace',rescaleFrame(whitespace))
cv.imshow('Figure',rescaleFrame(figure))
cv.waitKey(0), cv.destroyAllWindows()

## Save results.
save = input('Save figure? (y/n): ')
if save == 'y':
    save_dir = os.path.splitext(path)[0] + 'FIGURE' + os.path.splitext(path)[1]
    cv.imwrite(save_dir, figure)
    print('SAVED TO: ' + save_dir)