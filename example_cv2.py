import cv2
import threading
import numpy as np



class Extractor:

    def kp(self,image):
        grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kp=cv2.goodFeaturesToTrack(grey,maxCorners=700, qualityLevel=0.01, minDistance=3)
        kp=kp.tolist()
        
        for k in kp:
            x,y=map(lambda x: int(round(x)), k[0])
            print(f'({x}:{y})')
          #  x,y=355,245
            cv2.circle(image, (x,y), 3, (0,255,0), 1)
        return image 


webcam = cv2.VideoCapture(0)

while True:

    ret, frame = webcam.read()
    extractor=Extractor()
    frame1=extractor.kp(frame)
    cv2.imshow('img',frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
