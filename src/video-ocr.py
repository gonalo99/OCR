import cv2
import pytesseract
import easyocr

# Point to the installed tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Setup easyocr reader object
reader = easyocr.Reader(['en'])

# Use video's local file path or 0 as argument to access the camera
cap = cv2.VideoCapture('videos/video1.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(5)
frame_count = cap.get(7)

# Setup output video file with the processed frames
out = cv2.VideoWriter('videos/output1.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))
i = 0

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    if i % 20 == 0: # To proccess only every 20th frame
      # Choose which OCR library to use
      results = pytesseract.image_to_data(frame)
      #results = reader.readtext(frame)

      print("Processing frame ", i," out of ", frame_count)
      

    # - This is the block to process the frames using Tesseract -
    for id, line in enumerate(results.splitlines()):
        if id != 0:
            line = line.split()
            if len(line) == 12 and float(line[10]) > 10: # line[10] is the output confidence, up to 100
                x, y, w, h = int(line[6]), int(line[7]), int(line[8]), int(line[9])
                cv2.rectangle(frame, (x, y), (w+x, h+y), (0, 255, 0), 2)
                cv2.putText(frame, line[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)

    # - This is the block to process the frames using EasyOCR -
    #for result in results:
    #    cv2.putText(frame, text=result[1], (int(result[0][0][0]), int(result[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    #    cv2.rectangle(frame, (int(result[0][0][0]), int(result[0][0][1])), (int(result[0][2][0]), int(result[0][2][1])), (0,255,0),3)


    # Display the resulting frame
    #cv2.imshow('Frame', frame)
    # Write the frame into the output file
    out.write(frame)
    i += 1

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
