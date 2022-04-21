import cv2
import os
import numpy as np

import models


def main():
    model = models.mmocr()

    dataset = "IC15"
    file_path = "../data/evaluation_data_det/" + dataset + "/"
    test_images = open(os.path.join(file_path, "test_list.txt"), 'r').readlines()

    for image in test_images:
        frame = cv2.imread(os.path.join(file_path, "test_images/", image[:-1]))
        blurred = frame.copy()

        for line in open(os.path.join(file_path, "test_gts/", image[:-5] + ".txt"), "r").readlines():
            vertices = line.split(",")
            if vertices[8][:-1] == "###":
                box = np.int32([[vertices[0], vertices[1]], [vertices[2], vertices[3]], [vertices[4], vertices[5]], [vertices[6], vertices[7]]])
                blurred = blur_portion(blurred, box)

            box = np.int32([[vertices[0], vertices[1]], [vertices[2], vertices[3]], [vertices[4], vertices[5]], [vertices[6], vertices[7]]])
            blurred = cv2.drawContours(blurred, [np.int32(box)], -1, (0, 0, 255), 2)

        #boxes = model.get_bboxes(blurred)
        #res = blurred.copy()
        #for box in boxes:
        #    new = [[box[0], box[3]], [box[2], box[3]], [box[2], box[1]], [box[0], box[1]]]
        #    res = cv2.drawContours(res, [np.int32(new)], -1, (0, 0, 255), 3)

        if not cv2.imwrite("../data/blurred/"+dataset+"/test_images/"+image[:-1], blurred):
            print("An error occurred saving image ", image[:-1])


def blur_portion(image, box):
    blurred_image = cv2.GaussianBlur(image, (43, 43), 30)
    mask = np.zeros(image.shape, dtype=np.uint8)
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, [box], ignore_mask_color)
    mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask
    return cv2.bitwise_and(blurred_image, mask) + cv2.bitwise_and(image, mask_inverse)


if __name__ == "__main__":
    main()

