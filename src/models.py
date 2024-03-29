import cv2
import numpy as np
import easyocr
import pytesseract
from mmocr.utils.ocr import MMOCR


class mmocr():
    def get_bboxes(self, frame):
        ocr = MMOCR(config_dir="../../../Software/mmocr/configs", recog=None, det="MaskRCNN_IC17")
        results = ocr.readtext(frame)

        boxes = []
        for box in results[0]['boundary_result']:
            boxes.append([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]])

        return np.array(boxes)

    def get_text(self, frame):
        ocr = MMOCR(config_dir="../../../Software/mmocr/configs", det=None, recog="CRNN")
        results = ocr.readtext(frame)
        return results[0]['text']


class EasyOCR():
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def get_bboxes(self, frame):
        ths = 0.1
        results = self.reader.detect(frame, ycenter_ths=ths, height_ths=ths, width_ths=ths, text_threshold=ths)
        boxes = []

        for box in results[0][0]:
            boxes.append([[box[0], box[2]], [box[1], box[2]], [box[1], box[3]], [box[0], box[3]]])

        for box in results[1][0]:
            boxes.append([box[0], box[1], box[2], box[3]])

        return np.array(boxes)

    def get_text(self, frame):
        texts = self.reader.recognize(frame, batch_size=2, allowlist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.")
        return texts[0][1]


class Tesseract():
    def get_bboxes(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = pytesseract.image_to_data(frame, config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 11')

        boxes = []
        for id, line in enumerate(results.splitlines()):
            if id != 0:
                line = line.split()
                if len(line) == 12:
                    x, y, w, h = int(line[6]), int(line[7]), int(line[8]), int(line[9])
                    boxes.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])

        return np.array(boxes)

    def get_text(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = pytesseract.image_to_data(frame, config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ. --psm 8')

        texts = []
        for id, line in enumerate(results.splitlines()):
            if id != 0:
                line = line.split()
                if len(line) == 12:
                    texts.append(line[11])
        if len(texts) > 0:
            return texts[0]
        else:
            return texts


class TextSpotting():
    def __init__(self):
        recModelPath = "../data/crnn.onnx"
        vocPath = "../data/alphabet_36.txt"
        detModel = "DB_TD500_resnet18.onnx"

        if detModel == "DB_IC15_resnet18.onnx":
            inputHeight = 736
            inputWidth = 1280
        if detModel == "DB_TD500_resnet18.onnx":
            inputHeight = 736
            inputWidth = 736

        detModelPath = "../data/"+detModel

        # Load networks
        self.detector = cv2.dnn_TextDetectionModel_DB(detModelPath)
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.recognizer = cv2.dnn_TextRecognitionModel(recModelPath)
        self.recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Load vocabulary
        if vocPath:
            with open(vocPath, 'rt') as f:
                vocabulary = f.read().rstrip('\n').split('\n')
        self.recognizer.setVocabulary(vocabulary)
        self.recognizer.setDecodeType("CTC-greedy")

        # Parameters for Detection
        detScale = 1.0 / 255.0
        detInputSize = (inputWidth, inputHeight)
        detMean = (122.67891434, 116.66876762, 104.00698793)
        self.detector.setInputParams(detScale, detInputSize, detMean)

        # Parameters for Recognition
        recScale = 1.0 / 127.5
        recMean = (127.5)
        recInputSize = (100, 32)
        self.recognizer.setInputParams(recScale, recInputSize, recMean)


    def get_bboxes(self, frame):
        detResults = self.detector.detect(frame)
        boxes = []
        if len(detResults[0]) > 0:
            for i in range(len(detResults[0])):
                quadrangle = detResults[0][i].astype('float32')
                boxes.append([quadrangle[1], quadrangle[2], quadrangle[3], quadrangle[0]])

        return np.array(boxes)

    def get_text(self, frame):
        texts = self.recognizer.recognize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return texts

