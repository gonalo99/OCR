import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

import models
import time


def main():
    case = "PDT"
    fps = 3

    for (i, filename) in enumerate(sorted(glob.glob('/home/goncalo/rafa_docs/Processing Hololens/New_test/DEmo/'+case+'/PV/*.png'))):
        img = cv2.imread(filename)
        if i == 0:
            frame_height, frame_width, _ = img.shape
            out = cv2.VideoWriter('/home/goncalo/rafa_docs/Processing Hololens/New_test/DEmo/3fps_Video_' + case + '.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

        if i % 5 == 0:
            out.write(img)
    out.release()

if __name__ == "__main__":
    main()