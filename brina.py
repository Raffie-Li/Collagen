import cv2 as cv
import numpy as np
import os

def rescaleFrame(frame, scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

hsv_images = []
filenames = []
path = 'C:/Users/Raffie/Desktop/Collagen'
for filename in sorted(os.listdir(path)):
    img = cv.imread(os.path.join(path,filename))
    if img is not None:
        hsv_images.append(cv.cvtColor(img, cv.COLOR_BGR2HSV))
        filenames.append(filename)

masks = []
whitespace = []
hsv_color1 = np.asarray([0, 0, 170])
hsv_color2 = np.asarray([180, 45, 255])
for img in hsv_images:
    mask = cv.inRange(img, hsv_color1, hsv_color2)
    masks.append(mask)
for mask in masks:
    whitespace.append(cv.countNonZero(mask))

# total pixels in the first image (assume all images are same resolution, 2048x1536)
x = hsv_images[0].shape[1]
y = hsv_images[0].shape[0]
total_pix = x*y

percentages = []
for i in whitespace:
    percentage = 100*(i/total_pix)
    percentages.append(percentage)
for name, percentage in zip(sorted(os.listdir(path)), percentages):
    print('Filename:', name, '\n', 'Percent Whitespace:', percentage, '\n')

results = []
for img, mask, filename in zip(hsv_images, masks, sorted(os.listdir(path))):
    blend = cv.addWeighted(cv.cvtColor(img, cv.COLOR_HSV2BGR),
            0.6, cv.cvtColor(mask, cv.COLOR_GRAY2BGR), 0.4, 0.0)
    results.append(blend)
view = input('Would you like to see the results? (y/n): ')
if view == 'y':
    for result, filename in zip(results, sorted(os.listdir(path))):
        resized = rescaleFrame(result)
        cv.imshow(filename, resized)
        cv.waitKey(0)
        cv.destroyWindow(filename)
else:
    pass

save = input('Shall I save the results? (y/n): ')
if save == 'y':
    save_paths = []
    new_dir = path + '/Results'
    np.savetxt(path + '/Results.csv', np.column_stack((filenames, whitespace, percentages)),
               delimiter=',', fmt='%s', header='Filename')
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for filename in sorted(os.listdir(path)):
        save_paths.append(new_dir + '/' + filename)
    for save_path, result in zip(save_paths, results):
        cv.imwrite(save_path, result)
