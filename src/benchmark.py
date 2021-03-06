import argparse
import cv2
import os.path
import time

import models
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["easyocr", "tesseract", "text_spotting", "mmocr"], help="Choose a model from Easyocr, Tesseract, MMOCR or TextSpotting")
    parser.add_argument("--metric", help="Choose the detection evaluation metric (1to1 or union)", choices=["1to1", "union"], default="union")
    args = parser.parse_args()

    if args.model == "easyocr":
        model = models.EasyOCR()
    elif args.model == "tesseract":
        model = models.Tesseract()
    elif args.model == "mmocr":
        model = models.mmocr()
    else:
        model = models.TextSpotting()

    # evaluate_recognition(model, args.model)
    evaluate_detection(model, args.model, "IC15", args.metric)


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
        texts = model.get_text(frame)
        end = time.time()
        total_time += end - start

        if len(texts) > 0:
            if texts == info[1][:-1]:
                hits += 1

        if i / len(test_texts) > progress:
            print("Processing frames: ", "{:.2f}".format(progress * 100), "% completed")
            progress += 0.2

        i += 1

    print("Recognition results for model ", name)
    print("Recognition accuracy: ", "{:.2f}".format(hits / len(test_texts) * 100), "%")
    print("Time per frame: ", total_time / len(test_texts), " - FPS: ", len(test_texts) / total_time)
    

def evaluate_detection(model, name, dataset, metric):
    file_path = "../data/blurred/" + dataset + "/"
    test_images = open(os.path.join(file_path, "test_list.txt"), 'r').readlines()

    i, t40, t60, t80, progress, total_time, number_true = [0, 0, 0, 0, 0, 0, 0]

    if metric == "union":
        for image in test_images:
            frame = cv2.imread(os.path.join(file_path, "test_images/", image[:-1]))
            true_boxes = utils.readTruths(os.path.join(file_path, "test_gts/", image[:-5] + ".txt"))

            start = time.time()
            boxes = model.get_bboxes(frame)
            end = time.time()
            total_time += end - start

            score = utils.union_iou(boxes, true_boxes)
            if score > 0.40:
                t40 += 1
                if score > 0.60:
                    t60 += 1
                    if score > 0.80:
                        t80 += 1

            if i / len(test_images) > progress:
                print("Processing frames: ", "{:.2f}".format(progress * 100), "% completed")
                progress += 0.2

            i += 1

        print("Detection results for model ", name)
        print("Threshold 40: ", "{:.2f}".format(t40 / len(test_images) * 100), "%")
        print("Threshold 60: ", "{:.2f}".format(t60 / len(test_images) * 100), "%")
        print("Threshold 80: ", "{:.2f}".format(t80 / len(test_images) * 100), "%")
        print("Time per frame: ", total_time / len(test_images), " - FPS: ", len(test_images) / total_time)

    else:
        for image in test_images:
            frame = cv2.imread(os.path.join(file_path, "test_images/", image[:-1]))
            true_boxes = utils.readTruths(os.path.join(file_path, "test_gts/", image[:-5] + ".txt"))
            number_true += len(true_boxes)

            start = time.time()
            boxes = model.get_bboxes(frame)
            end = time.time()
            total_time += end - start

            if len(boxes) > 0:
                _, _, ious, _ = utils.match_bboxes(true_boxes, boxes, 0.4)
                t40 += len(ious)
                t60 += len([hit for hit in ious if hit > 0.6])
                t80 += len([hit for hit in ious if hit > 0.8])

            if i / len(test_images) > progress:
                print("Processing frames: ", "{:.2f}".format(progress * 100), "% completed")
                progress += 0.2

            i += 1

        print("Detection results for model ", name)
        print("Threshold 40: ", "{:.2f}".format(t40 / number_true * 100), "%")
        print("Threshold 60: ", "{:.2f}".format(t60 / number_true * 100), "%")
        print("Threshold 80: ", "{:.2f}".format(t80 / number_true * 100), "%")
        print("Time per frame: ", total_time / len(test_images), " - FPS: ", len(test_images) / total_time)


if __name__ == "__main__":
    main()
