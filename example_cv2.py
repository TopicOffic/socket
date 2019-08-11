import cv2
import threading
import numpy as np



class Extractor:

    orb=cv2.ORB_create()

    def kp(self,image):
        kp=cv2.goodFeaturesToTrack(image,maxCorners=1000, qualityLevel=0.01, minDistance=1)
        for k in kp:
            x,y=map(lambda x: int(round(x)), k.pt)
            print(f'({x}:{y})')
            cv2.circle(image, (x,y), 3, (0,255,0), 1)
            return image 


#webcam = cv2.VideoCapture(0)

while True:

   # ret, frame = webcam.read()
    frame=cv2.imread('car.jpg',0)
    extractor=Extractor()
    frame1=extractor.kp(frame)
    cv2.imshow('img',frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#webcam.release()
cv2.destroyAllWindows()
