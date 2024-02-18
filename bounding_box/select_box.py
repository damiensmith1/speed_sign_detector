import json
import csv
import os
import argparse
import cv2
from tkinter import Tk, filedialog

keyPt = None
image = None
image_filename = None
clone = None
ptSelected = None
scale = None
imageList = []

def click_and_pick(event, x, y, flags, param):
    global keyPt, image, image_filename, clone, ptSelected, scale, imageList 

    if event == cv2.EVENT_LBUTTONDOWN:
        if not ptSelected:
            keyPt = [(x, y)]
            ptSelected = True
        else:
            keyPt.append((x, y))
            cv2.rectangle(image, keyPt[0], keyPt[1], (0, 255, 0), 1)

            cv2.putText(image, f"Box: {keyPt[0]} - {keyPt[1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            ptSelected = False
            coordinate1 = keyPt[0]
            coordinate2 = keyPt[1]
            keyPt = None 

            imageList.append((image_filename, coordinate1, coordinate2))


            dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow(image_filename, imageScaled)

    elif event == cv2.EVENT_RBUTTONDOWN:
        ptSelected = False
        image = clone.copy()
        dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow(image_filename, imageScaled)


def main():

    global keyPt, image, image_filename, clone, ptSelected, scale, imageList 

    theta = 0.0
    fileList = []
    scale = 1.0
    outFile = None

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    inImagePath = filedialog.askdirectory(title='Select input image directory')
    print(inImagePath)
    root.destroy()
    outFile = os.path.join(inImagePath, 'labels.txt')

    for filename in os.listdir(inImagePath):
        filepath = os.path.join(inImagePath, filename)
        if os.path.isfile(filepath) and (filename.endswith('.png') or filename.endswith('.bmp') or filename.endswith('.jpg')):
            fileList = fileList + [filename]

    print(fileList)

    ptList = [[]] * len(fileList)

    i = 0
    ptSelected = False

    print('press \'m\' for menu ...')
    while i < len(fileList):

        filename = fileList[i]
        filepath = os.path.join(inImagePath, filename)

        if os.path.isfile(filepath) and (filepath.endswith('.png') or filepath.endswith('.bmp') or filepath.endswith('.jpg')):

            image = cv2.imread(filepath)

            clone = image.copy()

            dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            cv2.destroyAllWindows()
            image_filename = filename
            cv2.imshow(image_filename, imageScaled)
            cv2.setMouseCallback(image_filename, click_and_pick)

            ptSelected = False

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("m"):
                print('--------- Menu ----------')
                print('\t\t<space> = save keypoint and load next image)')
                print('\t\t<left mouse click> = select keypoint')
                print('\t\t<right mouse click> = unselect keypoint')
                print('\t\tCTRL + <left mouse click> = move keypoint left')
                print('\t\tALT + <left mouse click> = move keypoint right')
                print('\t\tCTRL + SHIFT + <left mouse click> = move keypoint up')
                print('\t\tALT + SHIFT + <left mouse click> = move keypoint down')
                print('\t\tq = (q)uit')

            elif key == ord(">"):
                i += 1

            elif key == ord("<"): 
                i -= 1

            elif key == 32 :

                if ptSelected:
                    imageList = imageList + [[filename, (int(keyPt[0]/scale), int(keyPt[1]/scale))]]

                i += 1

    print('saving : ', imageList)
    filename = outFile

    with open(outFile, 'w', newline='') as csvfile:
        fieldnames = ['image', 'coordinate1', 'coordinate2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for item in imageList:
            writer.writerow(
                {'image': item[0], 'coordinate1': item[1], 'coordinate2': item[2]})

if __name__ == '__main__':

    main()
