import cv2 as cv
import numpy as np
import os
path = 'C:/Users/tit/Desktop/SMA/L45.4 10x/Pos0'  # Change this to location of your images!

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def getImages():
    images = []
    filenames = []
    for filename in sorted(os.listdir(path)):
        img = cv.imread(os.path.join(path,filename))
        if img is not None:
            copy = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            copy8bit = cv.normalize(copy, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            images.append(copy8bit)
            filenames.append(filename)
    return images, filenames
    
def processImages(images, filenames):
    displayImages = True  # Change this to True if you want to see the raw images!!
    sma_pixels, tubule_pixels = [], []
    signals, tubules = [], []
    for image, filename in zip(images, filenames):
        signal = cv.inRange(image, 70, 190)
        sma = cv.countNonZero(signal)
        sma_pixels.append(sma)
        signals.append(signal)

        validKidney = cv.inRange(image, 10, 190)
        validKidneyCount = cv.countNonZero(validKidney)
        tubule_pixels.append(validKidneyCount)
        tubules.append(validKidney)

        if displayImages == True:
            cv.imshow(filename + ' Normalized', rescaleFrame(image))
            # cv.imshow('SMA', rescaleFrame(signal))
            # cv.imshow('Tubules', rescaleFrame(validKidney))
            cv.waitKey(0); cv.destroyAllWindows(); cv.waitKey(1)
            
    proceed = input('OK to Proceed (y/n): ')
    if proceed == 'y':
        pass
    else:
        exit()

    return sma_pixels, tubule_pixels, signals, tubules

def calcAvgs(sma_pix, tubule_pix):
    # Percent Formula: SMA pixels / Tubule Pixels
    percentages = []
    for sma, tubule in zip(sma_pix, tubule_pix):
        percent = 100*(sma/tubule)
        percentages.append(percent)
    return percentages

def generateFigs(signals, tubules, filenames):
    displayFigures = True
    figures = []
    for r, g, filename in zip(signals,tubules,filenames):
        empty_channel = np.zeros((1040,1392,1),dtype=np.uint8)
        figure = cv.merge((empty_channel, g, r))
        figure[r > 0] = (0, 0, 255)
        if displayFigures == True:
            cv.imshow(filename, rescaleFrame(figure))
            cv.waitKey(0), cv.destroyAllWindows(), cv.waitKey(1)
        figures.append(figure)
    return figures

def saveData(sma_pix, tubule_pix, percentages, figures, filenames):
    save_paths = []
    new_dir = path + '/Results'
    np.savetxt(path + '/Results.csv', np.column_stack((filenames, sma_pix, tubule_pix, percentages)),
               delimiter=',', fmt='%s', header='Filename')
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for filename in filenames:
        save_paths.append(new_dir + '/' + filename + '.tif')
    for save_path, figure in zip(save_paths, figures):
        cv.imwrite(save_path, figure)
    return()

def main():
    images, filenames = getImages()
    sma_pix, tubule_pix, signals, tubules = processImages(images,filenames)
    percentages = calcAvgs(sma_pix, tubule_pix)
    for name,percentage in zip(filenames,percentages):
        print('Filename: ', name, '\n', 'Percent SMA: ', percentage, '\n')
    figures = generateFigs(signals, tubules, filenames)
    save = input('Save results? (y/n): ')
    if save == 'y':
        saveData(sma_pix, tubule_pix, percentages, figures, filenames)

main()
