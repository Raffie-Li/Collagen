import cv2 as cv
import numpy as np
import colorsys

def rescaleFrame(frame, scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('D:/Collagen/TRF/HA57.2/10x0006.tif', cv.IMREAD_COLOR)
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

hsv_color1 = np.asarray([0, 0, 235])
hsv_color2 = np.asarray([180, 255, 255])
brightest_pixel_mask = cv.inRange(hsv_img, hsv_color1, hsv_color2)
coordinates = list(zip(np.nonzero(brightest_pixel_mask)[0],np.nonzero(brightest_pixel_mask)[1]))

hsv_vectors = []
for coordinate in coordinates:
    hsv_vectors.append(hsv_img[coordinate])

a = np.array(hsv_vectors)
white_point = np.mean(a, axis=0)
rounded = np.around(white_point, decimals=0)

h,s,v = rounded[0]/180, rounded[1]/255, rounded[2]/255
rgb_fractional = colorsys.hsv_to_rgb(h,s,v)
r,g,b = rgb_fractional[0]*255,rgb_fractional[1]*255,rgb_fractional[2]*255

blank_image = np.zeros((100,100,3), np.uint8)
blank_image[:,:,0] = b
blank_image[:,:,1] = g
blank_image[:,:,2] = r
print('Average of brightest pixels (HSV):', rounded, '\n', 'Average of brightest pixels (BGR):', [b,g,r])

blue_ratio  = 255/b
green_ratio = 255/g
red_ratio = 255/r

blue=cv.multiply(img[...,0],blue_ratio)
green=cv.multiply(img[...,1],green_ratio)
red=cv.multiply(img[...,2],red_ratio)

balanced = cv.merge([blue,green,red])

cv.imshow("Balanced Image", rescaleFrame(balanced))
cv.imshow("White Sample", blank_image)
cv.imshow("Source Image", rescaleFrame(img))
cv.imshow("Brightest", rescaleFrame(brightest_pixel_mask))
cv.waitKey(0)


