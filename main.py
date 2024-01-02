import cv2
from matplotlib import pyplot as plt 

FaceCascade = cv2.CascadeClassifier('Cascade/face.xml')
EyeCascade = cv2.CascadeClassifier('Cascade/haarcascade_eye.xml')
SmileCascade = cv2.CascadeClassifier('Cascade/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    ret , img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = FaceCascade.detectMultiScale(gray, 1.3, 5,minSize=(30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        print('face', x, y, w, h)
        cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 255, 78), 2)
        
        # detect eyes
        eyes = EyeCascade.detectMultiScale(roi_gray,1.5,5,minSize=(5, 5))
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            print('eye', ex, ey, ew, eh)
            cv2.putText(img, "Eye", (ex,ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 255, 78), 2)
        
        smile = SmileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            )
        
        for (xx, yy, ww, hh) in smile:
            print('smile', xx, yy, ww, hh)
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
            cv2.putText(img, "Smile", (xx, yy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 255, 78), 2)
    cv2.imshow('video', img) 
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
