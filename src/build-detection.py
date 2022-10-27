import sys

from bs4 import BeautifulSoup
import glob
import cv2
import numpy as np

from matplotlib import pyplot as plt
import random

bad = True


def main():
    recog_no = 1
    detect_no = 1

    # Loop through all the frames
    for filename in sorted(glob.glob('../data/OCR_images/default/*.PNG'), key=lambda k: random.random()):
        frame = cv2.imread(filename)
        good_frame = False

        # Read xml corresponding to each frame
        with open(filename[:-3]+'xml', 'r') as xml_file:
            data = xml_file.read()

        # Loop through each of the bounding boxes in the frame
        parsedData = BeautifulSoup(data, "xml")
        objects = parsedData.find_all('object')
        for object in objects:
            pts = object.find_all('pt')
            quadrangle = []
            for pt in pts:
                x = pt.find('x').text
                y = pt.find('y').text
                quadrangle.append([x, y])

            # Get bounding box and label in the right formats
            new_quadrangle = [quadrangle[3], quadrangle[0], quadrangle[1], quadrangle[2]]
            label = object.find("name").text

            # Crop the image in the bounding box
            cropped = fourPointsTransform(frame, new_quadrangle)

            # Get a metric for the quality of the bbox
            #gX = cv2.Sobel(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0)
            #gY = cv2.Sobel(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1)
            #height, width = cropped.shape[:2]
            #print(np.average(np.sqrt((gX ** 2) + (gY ** 2)))/(width*height))
            #print(cv2.Laplacian(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()/(width*height))

            # Display the images and decide if they are good (c) or bad (b) - terminate program with enter
            fig, ax = plt.subplots()
            fig.canvas.mpl_connect('key_press_event', on_press)
            ax.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            plt.show(block=False)
            plt.waitforbuttonpress()
            plt.close()

            detect_gt_file = open('../data/custom_detection/gts/img_' + str(detect_no) + '.txt', 'a')
            if not bad:
                string = str(quadrangle[0][0])+','+str(quadrangle[0][1])+','+str(quadrangle[1][0])+','+str(quadrangle[1][1])+','+str(quadrangle[2][0])+','+str(quadrangle[2][1])+','+str(quadrangle[3][0])+','+str(quadrangle[3][1])+',' + label + ',good' + '\n'
                detect_gt_file.write(string)
                good_frame = True
                print("Good")
            else:
                string = str(quadrangle[0][0]) + ',' + str(quadrangle[0][1]) + ',' + str(quadrangle[1][0]) + ',' + str(quadrangle[1][1]) + ',' + str(quadrangle[2][0]) + ',' + str(quadrangle[2][1]) + ',' + str(quadrangle[3][0]) + ',' + str(quadrangle[3][1]) + ',' + label + ',bad' + '\n'
                detect_gt_file.write(string)
                print("Bad")
            detect_gt_file.close()

        if good_frame:
            cv2.imwrite('../data/custom_detection/images/img_' + str(detect_no) + '.png', frame)
            detect_no += 1

            # Save recognition dataset
            #cv2.imwrite('../data/custom_recognition/img_' + str(recog_no) + '.png')
            #recog_gt_file = open('../data/custom_recognition/gts.txt', 'a')
            #recog_gt_file.write('img_' + str(recog_no) + ' ' + label + '\n')
            #recog_gt_file.close()
            #recog_no += 1


def on_press(event):
    global bad
    if event.key == 'c':
        bad = False
    elif event.key == 'b':
        bad = True
    elif event.key == 'enter':
        sys.exit()


def fourPointsTransform(frame, vertices):
    vertices = np.float32(vertices)

    width = np.sqrt((vertices[3][0] - vertices[1][0])**2 + (vertices[3][1] - vertices[1][1])**2)
    height = np.sqrt((vertices[1][0] - vertices[0][0])**2 + (vertices[1][1] - vertices[0][1])**2)

    outputSize = (int(width), int(height))

    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    return cv2.warpPerspective(frame, rotationMatrix, outputSize)


if __name__ == "__main__":
    main()
