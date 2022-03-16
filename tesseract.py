from matplotlib import pyplot
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('images/image1.jpg')
pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

results = pytesseract.image_to_data(img)
for id, line in enumerate(results.splitlines()):

    if id != 0:
        line = line.split()
        print(float(line[10]) > 30)
        if len(line) == 12:
            x, y, w, h = int(line[6]), int(line[7]), int(line[8]), int(line[9])
            res = cv2.rectangle(img, (x, y), (w+x, h+y), (0, 255, 0), 2)
            res = cv2.putText(img, line[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            pyplot.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

pyplot.show()