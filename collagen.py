import cv2 as cv
import numpy as np
import os

def rescaleFrame(frame, scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

hsv_images = []
path = 'C:/Users/Raffie/Desktop/Collagen'
for filename in os.listdir(path):
    img = cv.imread(os.path.join(path,filename))
    if img is not None:
        hsv_images.append(cv.cvtColor(img, cv.COLOR_BGR2HSV))

masks = []
hsv_color1 = np.asarray([115, 50, 20])
hsv_color2 = np.asarray([170, 255, 255])
for img in hsv_images:
    mask = cv.inRange(img, hsv_color1, hsv_color2)
    masks.append(mask)

# total pixels in the first image (assume all images are same resolution, 2048x1536)
x = hsv_images[0].shape[1]
y = hsv_images[0].shape[0]
total_pix = x*y

# number of collagen pixels
collagen = []
for mask in masks:
    collagen.append(cv.countNonZero(mask))

# amount of whitespace
hsv_color3 = np.asarray([0, 0, 170])
hsv_color4 = np.asarray([180, 45, 255])
whitespace = []
for img in hsv_images:
    whitespace.append(cv.countNonZero(cv.inRange(img, hsv_color3, hsv_color4)))

percent_collagen = [100*(i/(total_pix-j)) for i,j in zip(collagen, whitespace)]
for name, percentage in zip(os.listdir(path), percent_collagen):
    print('Filename:', name, '\n', 'Percent Collagen:', percentage, '\n')

view = input('Would you like to see the results? (y/n): ')
if view == 'y':
    for img, mask, filename in zip(hsv_images, masks, os.listdir(path)):
        blend = cv.addWeighted(cv.cvtColor(img, cv.COLOR_HSV2BGR),
                0.6, cv.cvtColor(mask, cv.COLOR_GRAY2BGR), 0.4, 0.0)
        resized = rescaleFrame(blend)
        cv.imshow(filename, resized)
        cv.waitKey(0)
        cv.destroyWindow(filename)
else:
    quit()
