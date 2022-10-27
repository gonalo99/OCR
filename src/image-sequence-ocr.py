import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import time

import models
import utils


def main():
    detector = models.mmocr()
    recognizer = models.mmocr()

    img_array = []
    frame_count = 0

    case = "power_supply_multi"

    for filename in sorted(glob.glob('/home/goncalo/rafa_docs/Processing Hololens/New_test/DEmo/'+case+'/PV/*.png')):
        img = cv2.imread(filename)
        img_array.append(img)
        frame_count += 1

    frame_height, frame_width, _ = img_array[0].shape

    # Setup output video file with the processed frames
    fps = 5
    #out = cv2.VideoWriter('/home/goncalo/rafa_docs/Processing Hololens/New_test/DEmo/'+case+'_mmocr.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))
    i = 0

    start = time.time()
    for frame in img_array:
        if i % 3 == 0:
            newboxes = []

            found = False

            boxes = detector.get_bboxes(frame)
            for box in boxes:
                box = [box[3], box[0], box[1], box[2]]

                quadrangle_2f = []
                for j in range(4):
                    quadrangle_2f.append(box[j])

                print(quadrangle_2f)

                cropped = utils.fourPointsTransform(frame, quadrangle_2f, 20)
                found = True
                plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                plt.show()

                text = recognizer.get_text(cropped)
                #print(text)
                newboxes.append([box, text])

        for box in newboxes:
            cv2.putText(frame, box[1], (int(box[0][1][0]), int(box[0][1][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        cv2.polylines(frame, np.int32(boxes), True, (0, 255, 0), 2)

        #if found:
            #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #plt.show()

        # Write the frame into the output file
        #out.write(frame)
        i += 1

    end = time.time()
    print("Time is: "+str(end-start)+", about "+str(frame_count/(end-start))+" fps")
    #out.release()


if __name__ == "__main__":
    main()
