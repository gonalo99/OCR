import os
import utils
import cv2

import models

def main():
    model = models.mmocr()
    dataset = "IC15"

    file_path = "../data/blurred/" + dataset + "/"
    test_images = open(os.path.join(file_path, "test_list.txt"), 'r').readlines()

    for image in test_images:
        frame = cv2.imread(os.path.join(file_path, "test_images/", image[:-1]))
        boxes = model.get_bboxes(frame)
        f = open("../data/formatted/detection/"+dataset+"/easyocr1/" + image[:-5] + ".txt", "a")
        for box in boxes:
            f.write("placeholder " + str(box[4]) + " " + str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\n")
        f.close()

        # true_boxes = utils.readTruths(os.path.join(file_path, "test_gts/", image[:-5] + ".txt"))
        # for box in true_boxes:
        #     f = open("../data/formatted/detection/TD500/test_gts/" + image[:-5] + ".txt", "a")
        #     f.write("placeholder " + str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + "\n")
        #     f.close()


if __name__ == "__main__":
    main()
