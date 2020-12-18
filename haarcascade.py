import cv2
import numpy as np

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detect = cv2.CascadeClassifier('haarcascade_eye.xml')

camera = cv2.VideoCapture(0)

while 1:
    ret,goruntu = camera.read()
    gray = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray,1.3,5)
    
    for x,y,w,h in faces:
        cv2.rectangle(goruntu, (x,y), (x+w,y+h), (255,0,0,2))
        roi_gray = gray[y:y+h, x:x+w]
        roi_img =  goruntu[y:y+h, x:x+w]
        
        eyes =eye_detect.detectMultiScale(roi_gray)
        
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roi_img, (ex,ey), (ex+w,ey+h),(0,255,0), 2)
            
    cv2.imshow('img',goruntu)
    k = cv2.waitKey(30) & 0xff 
    if k == 27:
        break

camera.release ()
cv2.destroyAllWindows()