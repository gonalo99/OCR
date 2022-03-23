import argparse
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import os.path

import utils

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--inputImage', help = "Path to an input image. Skip this argument to capture frames from a camera", required=True)
    parser.add_argument("--model", required=True, choices=["easyocr", "tesseract", "text_spotting"], help="Choose a model from Easyocr, Tesseract or TextSpotting")
    parser.add_argument("--detModelPath", help="Path to a binary .onnx model for detection", default="../data/DB_TD500_resnet50.onnx")
    parser.add_argument("--recModelPath", help="Path to a binary .onnx model for recognition", default="../data/crnn.onnx")
    parser.add_argument("--vocPath", default="../data/alphabet_36.txt", help="Path to benchmarks for evaluation")
    args = parser.parse_args()

    if args.model == "easyocr":
        model = utils.EasyOCR()
    elif args.model == "tesseract":
        model = utils.Tesseract()
    else:
        model = utils.TextSpotting(args.detModelPath, args.recModelPath, args.vocPath)

    file_path = "../data/evaluation_data_det/ic15/"
    test_images = open("../data/evaluation_data_det/ic15/test_list.txt", 'r').readlines()

    i = 0
    hits = 0
    threshold = 0.40

    for image in test_images:
        frame = cv2.imread(os.path.join(file_path, "test_images/", image[:-1]))
        true_boxes = readTruths(os.path.join(file_path, "test_gts/", image[:-5]+".txt"))
        boxes, texts = model.process_frame(frame)
        i += 1
        if IoU(boxes, true_boxes) >= threshold:
            hits += 1
    
    print("Threshold ", threshold, " hits: ", hits/i*100, "%")

def readTruths(filepath):
    true_boxes = []
    for line in open(filepath, "r").readlines():
        vertices = line.split(",")
        box = np.int32([[vertices[0], vertices[1]], [vertices[2], vertices[3]], [vertices[4], vertices[5]], [vertices[6], vertices[7]]])
        true_boxes.append(box)

    return true_boxes

def IoU(boxes, true_boxes):
    true_polygon = unary_union([Polygon(box) for box in true_boxes])
    detected_polygon = unary_union([Polygon(box) for box in boxes])

    intersection = detected_polygon.intersection(true_polygon)
    return intersection.area / (true_polygon.area + detected_polygon.area)

if __name__ == "__main__":
    main()