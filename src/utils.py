import cv2
from matplotlib.pyplot import text
import numpy as np
import easyocr
import pytesseract

class EasyOCR():
    def process_frame(self, frame):
        reader = easyocr.Reader(['en'])
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
    def process_frame(self, frame):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        results = pytesseract.image_to_data(frame)
        
        boxes = []
        texts = []
        for id, line in enumerate(results.splitlines()):
            if id != 0:
                line = line.split()
                if len(line) == 12:
                    x, y, w, h = int(line[6]), int(line[7]), int(line[8]), int(line[9])
                    #cv2.rectangle(frame, (x, y), (w+x, h+y), (0, 255, 0), 2)
                    #cv2.putText(frame, line[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                    boxes.append(np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]))
                    texts.append(line[11])
        
        return [boxes, texts]

class TextSpotting():
    def __init__(self, detModelPath, recModelPath, vocPath):
        inputHeight = 736
        inputWidth = 736
        binaryThreshold = 0.3
        polygonThreshold = 0.5
        maxCandidates = 200
        unclipRatio = 2.0

        # Load networks
        self.detector = cv2.dnn_TextDetectionModel_DB(detModelPath)
        self.detector.setBinaryThreshold(binaryThreshold)
        self.detector.setPolygonThreshold(polygonThreshold)
        self.detector.setUnclipRatio(unclipRatio)
        self.detector.setMaxCandidates(maxCandidates)

        self.recognizer = cv2.dnn_TextRecognitionModel(recModelPath)

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