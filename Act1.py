import cv2
import os

cap=cv2.VideoCapture("C:/Users/mbnrs/OneDrive/Documents/Jetlearn/OpenCV-11/cars.opdownload")

plate_cascade=cv2.CascadeClassifier("C:/Users/mbnrs/OneDrive/Documents/Jetlearn/OpenCV-11/haarcascade_russian_plate_number.xml")

while True:
    ret,frames=cap.read()
    gray=cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    plates=plate_cascade.detectMultiScale(gray,1.1,1)
    print(plates)

    for (x,y,w,h) in plates:
        cv2.rectangle(frames,(x,y),(x+w,y+h), (255,0,0),2)
    cv2.imshow("OpenCV",frames)
    if cv2.waitKey(33) ==27:
        break
cv2.destroyAllWindows()