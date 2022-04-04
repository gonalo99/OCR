import cv2
import numpy as np
import easyocr
import pytesseract

class EasyOCR():
    def __init__(self):
        self.reader = easyocr.Reader(['en'])


    def get_bboxes(self, frame):
        ths = 0.1
        results = self.reader.detect(frame, ycenter_ths=ths, height_ths=ths, width_ths=ths, text_threshold=ths)
        boxes = []
        for result in results[0][0]:
            boxes.append([result[0], result[2], result[1], result[3]])

        return np.array(boxes)

    def get_text(self, frame):
        texts = self.reader.recognize(frame, batch_size=2, allowlist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        return texts[0][1]


    def process_frame(self, frame):
        results = reader.readtext(frame)

        boxes = []
        texts = []
        for result in results:
            #cv2.putText(frame, text=result[1], org=(int(result[0][0][0]), int(result[0][0][1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
            #cv2.rectangle(frame, (int(result[0][0][0]), int(result[0][0][1])), (int(result[0][2][0]), int(result[0][2][1])), (0,255,0),3)
            boxes.append(np.array(result[0]))
            texts.append(result[1])

        return [boxes, texts]

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
                    boxes.append([x, y, w + x, h + y])

        return np.array(boxes)

    def get_text(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = pytesseract.image_to_data(frame, config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8')

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


    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        results = pytesseract.image_to_data(frame, config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 11')

        boxes = []
        texts = []
        for id, line in enumerate(results.splitlines()):
            if id != 0:
                line = line.split()
                if len(line) == 12:
                    x, y, w, h = int(line[6]), int(line[7]), int(line[8]), int(line[9])
                    # cv2.rectangle(frame, (x, y), (w+x, h+y), (0, 255, 0), 2)
                    # cv2.putText(frame, line[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                    boxes.append([x, y, w+x, h+y])
                    texts.append(line[11])

        return [np.array(boxes), texts]

class TextSpotting():
    def __init__(self, detModel, recModelPath, vocPath):
        if detModel == "DB_IC15_resnet50.onnx" or detModel == "DB_IC15_resnet18.onnx":
            inputHeight = 736
            inputWidth = 1280
        if detModel == "DB_TD500_resnet50.onnx" or detModel == "DB_TD500_resnet18.onnx":
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
                x1, y1, x2, y2 = [quadrangle[0][0], quadrangle[2][1], quadrangle[2][0], quadrangle[0][1]]
                boxes.append([x1, y1, x2, y2])

        return np.array(boxes)

    def get_text(self, frame):
        texts = self.recognizer.recognize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return texts

    def process_frame(self, frame):
        detResults = self.detector.detect(frame)
        boxes = []
        texts = []
        if len(detResults[0]) > 0:
            # Text Recognition
            for i in range(len(detResults[0])):
                quadrangle = detResults[0][i].astype('float32')
                boxes.append(quadrangle)

                quadrangle_2f = []
                for j in range(4):
                    quadrangle_2f.append(quadrangle[j])

                # Transform and Crop
                cropped = self.fourPointsTransform(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), quadrangle_2f)
                recognitionResult = self.recognizer.recognize(cropped)
                #cv2.putText(frame, recognitionResult, (int(quadrangle[3][0]), int(quadrangle[3][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                texts.append(recognitionResult)

            #cv2.polylines(frame, np.int32(contours), True, (0, 255, 0), 2)

        return [boxes, texts]

    def fourPointsTransform(self, frame, vertices):
        vertices = np.asarray(vertices)
        
        outputSize = (100, 32)

        targetVertices = np.array([
            [0, outputSize[1] - 1],
            [0, 0],
            [outputSize[0] - 1, 0],
            [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

        rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
        return cv2.warpPerspective(frame, rotationMatrix, outputSize)