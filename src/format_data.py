import os
import numpy as np
import cv2

import utils

def main():
    model = utils.EasyOCR()
    dataset = "IC15"

    file_path = "../data/evaluation_data_det/" + dataset + "/"
    test_images = open(os.path.join(file_path, "test_list.txt"), 'r').readlines()

    for image in test_images:
        frame = cv2.imread(os.path.join(file_path, "test_images/", image[:-1]))
        boxes = model.get_bboxes(frame)
        f = open("../data/formatted/detection/IC15/easyocr/" + image[:-5] + ".txt", "a")
        for box in boxes:
            f.write("placeholder 1.0 " + str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\n")
        f.close()


        #true_boxes = readTruths(os.path.join(file_path, "test_gts/", image[:-5] + ".txt"))
        #for box in true_boxes:
        #    f = open("../data/formatted/detection/IC15/test_gts/" + image[:-5] + ".txt", "a")
        #    f.write("placeholder " + str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\n")
        #    f.close()



def readTruths(filepath):
    true_boxes = []
    for line in open(filepath, "r").readlines():
        vertices = line.split(",")
        box = [int(vertices[0]), int(vertices[1]), int(vertices[4]), int(vertices[5])]
        true_boxes.append(box)

    return np.array(true_boxes)


if __name__ == "__main__":
    main()
