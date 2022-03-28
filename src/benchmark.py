import argparse
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import os.path
import time

import utils


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--inputImage', help = "Path to an input image. Skip this argument to capture frames from a camera", required=True)
    parser.add_argument("--model", required=True, choices=["easyocr", "tesseract", "text_spotting"], help="Choose a model from Easyocr, Tesseract or TextSpotting")
    parser.add_argument("--detModel", help="Path to a binary .onnx model for detection", default="DB_TD500_resnet50.onnx", choices=["DB_IC15_resnet50.onnx", "DB_TD500_resnet50.onnx", "DB_IC15_resnet18.onnx", "DB_TD500_resnet18.onnx"])
    parser.add_argument("--recModelPath", help="Path to a binary .onnx model for recognition", default="../data/crnn.onnx")
    parser.add_argument("--vocPath", default="../data/alphabet_36.txt", help="Path to benchmarks for evaluation")
    args = parser.parse_args()

    if args.model == "easyocr":
        model = utils.EasyOCR()
    elif args.model == "tesseract":
        model = utils.Tesseract()
    else:
        model = utils.TextSpotting(args.detModel, args.recModelPath, args.vocPath)
        args.model += " " + args.detModel

    # evaluate_recognition(model, args.model)
    # evaluate_detection(model, args.model, "TD500")


def evaluate_recognition(model, name):
    file_path = "../data/evaluation_data_rec/"
    test_texts = open(file_path + "test_gts.txt", 'r').readlines()

    hits = 0
    i = 0
    progress = 0
    total_time = 0
    for text in test_texts:
        info = text.split(" ")
        frame = cv2.imread(file_path + info[0])
        start = time.time()
        boxes, texts = model.process_frame(frame)
        end = time.time()
        total_time += end - start

        if len(texts) > 0:
            if texts[0].lower() == info[1][:-1].lower():
                hits += 1

        if i / len(test_texts) > progress:
            print("Processing frames: ", "{:.2f}".format(progress * 100), "% completed")
            progress += 0.2

        i += 1

    print("Recognition results for model ", name)
    print("Recognition accuracy: ", "{:.2f}".format(hits / len(test_texts) * 100), "%")
    print("Time per frame: ", total_time / len(test_texts), " - FPS: ", len(test_texts) / total_time)
    

def evaluate_detection(model, name, dataset):
    file_path = "../data/evaluation_data_det/" + dataset + "/"
    test_images = open(os.path.join(file_path, "test_list.txt"), 'r').readlines()

    i, t40, t50, t60, t70, t80, t90, progress, total_time = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for image in test_images:
        frame = cv2.imread(os.path.join(file_path, "test_images/", image[:-1]))
        true_boxes = readTruths(os.path.join(file_path, "test_gts/", image[:-5] + ".txt"))

        start = time.time()
        boxes, texts = model.process_frame(frame)
        end = time.time()
        total_time += end - start

        score = IoU(boxes, true_boxes)
        if score > 0.40:
            t40 += 1
            if score > 0.50:
                t50 += 1
                if score > 0.60:
                    t60 += 1
                    if score > 0.70:
                        t70 += 1
                        if score > 0.80:
                            t80 += 1
                            if score > 0.90:
                                t90 += 1

        if i / len(test_images) > progress:
            print("Processing frames: ", "{:.2f}".format(progress * 100), "% completed")
            progress += 0.2

        i += 1

    print("Detection results for model ", name)
    print("Threshold 40: ", "{:.2f}".format(t40 / len(test_images) * 100), "%")
    print("Threshold 50: ", "{:.2f}".format(t50 / len(test_images) * 100), "%")
    print("Threshold 60: ", "{:.2f}".format(t60 / len(test_images) * 100), "%")
    print("Threshold 70: ", "{:.2f}".format(t70 / len(test_images) * 100), "%")
    print("Threshold 80: ", "{:.2f}".format(t80 / len(test_images) * 100), "%")
    print("Threshold 90: ", "{:.2f}".format(t90 / len(test_images) * 100), "%")
    print("Time per frame: ", total_time / len(test_images), " - FPS: ", len(test_images) / total_time)


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

    union = unary_union([true_polygon, detected_polygon])

    if union.area == 0:
        return 0
    else:
        intersection = detected_polygon.intersection(true_polygon)
        return intersection.area / union.area


if __name__ == "__main__":
    main()
