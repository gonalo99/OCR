import cv2
import glob
import numpy as np

import utils


def main():
    recog_no = 1

    goods = 0
    bads = 0
    # Loop through all the gt files
    for filename in sorted(glob.glob('../data/custom_detection/gts/*.txt')):
        # Get the respective frame
        img_no = filename[33:-3]
        frame = cv2.imread('../data/custom_detection/images/img_' + img_no + 'png')

        for line in open(filename, "r").readlines():
            data = line.split(",")
            box = np.array([[float(data[0]), float(data[1])], [float(data[2]), float(data[3])], [float(data[4]), float(data[5])], [float(data[6]), float(data[7])]])
            label = data[8]
            quality = data[9]

            if quality[:-1] == 'good':
                # Crop the frame around each bbox
                new_box = [box[3], box[0], box[1], box[2]]
                cropped = utils.fourPointsTransform(frame, new_box, 1)

                # Save recognition dataset
                cv2.imwrite('../data/custom_recognition/images/img_' + str(recog_no) + '.png', cropped)
                recog_gt_file = open('../data/custom_recognition/gts.txt', 'a')
                recog_gt_file.write('img_' + str(recog_no) + ' ' + label + '\n')
                recog_gt_file.close()
                recog_no += 1

                goods += 1
            else:
                bads += 1

    print("Goods:", goods)
    print("Bads:", bads)


if __name__ == "__main__":
    main()