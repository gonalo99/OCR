import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

import models
import utils


def main():
    detector = models.EasyOCR()
    recognizer = models.mmocr()

    # Use video's local file path or 0 as argument to access the camera
    cap = cv2.VideoCapture('../data/my_videos/video1.mp4')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)
    frame_count = cap.get(7)

    # Setup output video file with the processed frames
    out = cv2.VideoWriter('../output2.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))
    i = 0
    newboxes = []
    boxes = []

    start = time.time()
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # if i % 20 == 0:
            newboxes = []
            boxes = detector.get_bboxes(frame)
            for box in boxes:
                box = [box[3], box[0], box[1], box[2]]

                quadrangle_2f = []
                for j in range(4):
                    quadrangle_2f.append(box[j])

                cropped = utils.fourPointsTransform(frame, quadrangle_2f, 20)

                text = recognizer.get_text(cropped)
                newboxes.append([box, text])

            for box in newboxes:
                cv2.putText(frame, box[1], (int(box[0][1][0]), int(box[0][1][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.polylines(frame, np.int32(boxes), True, (0, 255, 0), 2)

            # Write the frame into the output file
            out.write(frame)
            i += 1

        else:
            break

    end = time.time()
    print("Time is: ", end-start, ", about ", frame_count/(end-start), " fps")
    # When everything done, release the video capture object
    cap.release()
    out.release()


if __name__ == "__main__":
    main()
