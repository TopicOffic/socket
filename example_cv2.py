import cv2
import numpy
import threading
import time


img = cv2.VideoCapture(0)


n=0


orb = cv2.ORB_create(100)

while True:
    ret, frame = img.read()
    kp, des = orb.detectAndCompute(frame, None)
    
    
    cv2.imshow('img2', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def foro(kp):
    for k in kp:
        kx = int(k.pt[0])
        ky =int(k.pt[1])     
        n +=1
        cv2.circle(frame, (kx,ky),3, (0,255,0), 1)




img.release()
cv2.destroyAllWindows()

