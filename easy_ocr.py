from matplotlib import pyplot
import easyocr
import cv2

img = cv2.imread('images/image1.jpg')
reader = easyocr.Reader(['en'])
results = reader.readtext(img)
pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

for result in results:
    res = cv2.putText(img, text=result[1], org=(int(result[0][0][0]), int(result[0][0][1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, (int(result[0][0][0]), int(result[0][0][1])), (int(result[0][2][0]), int(result[0][2][1])), (0,255,0),3)
    pyplot.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

pyplot.show()