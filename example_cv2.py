import cv2
import threading
import numpy as np



class Extractor:

    def kp(self,image):
        grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kp=cv2.goodFeaturesToTrack(grey,maxCorners=500, qualityLevel=0.01, minDistance=3)
        kp=kp.tolist()
        
        for k in kp:
            x,y=map(lambda x: int(round(x)), k[0])
            print(f'(x:{x},y:{y})')
          #  x,y=355,245
            cv2.circle(image, (x,y), 2, (0,255,0), -1)
        return image 


class Matcher:

    def match(self,image):
        orb=cv2.ORB_create(300)
        kp,des=orb.detectAndCompute(image,None)
        return (des,kp)
    
    def matcher(self,des,image):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        orb=cv2.ORB_create(300)
        kp,des1=orb.detectAndCompute(image,None)
        bf=cv2.BFMatcher()
        matches=bf.match(des,des1)
        matches_1=[]
        for n in matches:
            x=n.distance
            matches_1.append(x)
        dictionary=dict(zip(matches_1,kp))
        return dictionary
        

webcam = cv2.VideoCapture(0)
image=cv2.imread('cacao.jpg',1)
des=(Matcher().match(image))[0]

while True:

    ret, frame = webcam.read()
    #extractor=Extractor()
    #frame1=extractor.kp(frame)
    matches=Matcher().matcher(des,frame)
    sorted_good=sorted(matches.keys())
    first_10=sorted_good[:10]
    for x in first_10:
        k=matches.get(x)        
        x,y=int(k.pt[0]),int(k.pt[1])
        cv2.circle(frame, (x,y), 2, (0,255,0), -1)
        print(f'x:{x};y:{y}')

    cv2.imshow('img',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
