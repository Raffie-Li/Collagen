import cv2 as cv
import numpy as np

# rescaleFrame: a function that resizes the image window -- change "scale" to make larger/smaller.
def rescaleFrame(frame, scale=0.15):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# path: the file path to your image.
path = 'C:/Users/Raffie/Downloads/T86.3.tif'

# getImage: a function that loads the image from your harddrive.
def getImage():
    image = cv.imread(path, cv.IMREAD_COLOR)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return image, hsv_image

# liverArea: a function that isolate the liver from the rest of the image.
def liverArea(hsv_image):
    hsv_color1 = np.asarray([0, 0, 0])  # Lower HSV bound: hue, saturation, value
    hsv_color2 = np.asarray([179, 192, 200])  # Upper HSV bound: hue, saturation, value
    color_thresh = cv.inRange(hsv_image, hsv_color1, hsv_color2)

    contour, hierarchy = cv.findContours(color_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    liverAreaImage = np.zeros((hsv_image.shape[0], hsv_image.shape[1]), dtype=np.uint8)
    biggest_contour = max(contour, key=lambda item: cv.contourArea(item))
    cv.drawContours(liverAreaImage, [biggest_contour], -1, (255, 255, 255), cv.FILLED)

    liverPixelCount = cv.countNonZero(liverAreaImage)
    return liverAreaImage, liverPixelCount

# bileDuctArea: a function that isolates the bile ducts from the rest of the image
def bileDuctArea(hsv_image):
    hsv_color1 = np.asarray([120, 65, 70])
    hsv_color2 = np.asarray([180, 255, 170])
    color_thresh = cv.inRange(hsv_image, hsv_color1, hsv_color2)

    contours = cv.findContours(color_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    bileDuctAreaImage = np.zeros((hsv_image.shape[0], hsv_image.shape[1]), dtype=np.uint8)
    for c in contours:
        cv.drawContours(bileDuctAreaImage, [c], -1, (255, 255, 255), cv.FILLED)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10,10))
    opening = cv.morphologyEx(bileDuctAreaImage, cv.MORPH_OPEN, kernel)
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
    eroded = cv.erode(opening, kernel2, iterations = 2)
    return color_thresh, bileDuctAreaImage, eroded

# main: runs the program
def main():
    image, hsv_image = getImage()
    liverAreaImage, liverPixelCount = liverArea(hsv_image)
    color_thresh, bileDuctAreaImage, eroded = bileDuctArea(hsv_image)

    blend = cv.addWeighted(image, 0.95, cv.cvtColor(eroded, cv.COLOR_GRAY2BGR), 0.05, 0.0)
    blend[eroded > 0] = (0, 0, 255)

    cv.imshow('colorthresh', rescaleFrame(color_thresh))
    cv.imshow('opening', rescaleFrame(eroded))
    cv.imshow('blend', blend)
    cv.waitKey(0)

    savepath = 'C:/Users/Raffie/Downloads/result.tif'
    #cv.imwrite(savepath, blend)
main()