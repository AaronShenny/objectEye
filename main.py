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
    print(ret)
    if ret==False:
        print('Camera is not connected!!!')
        break
    faces = FaceCascade.detectMultiScale(gray, 1.3, 5,minSize=(30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        print('face', x, y, w, h)
        cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 255, 78), 2)
        print('Face Detected')
        # detect eyes
        eyes = EyeCascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            # cv2.circle(roi_color, (ex+(ew//2), ey+(eh//2)), (ew+eh)//4, (0, 0, 255), 2)
            print('eye', ex, ey, ew, eh)
            cv2.putText(roi_color, "Eye", (ex,ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 255, 78), 2)
        
        smile = SmileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.2,
            minNeighbors=20,
            minSize=(25, 25),
            )
        if(len(smile)>0):
            for (xx, yy, ww, hh) in smile:
                print('smile', xx, yy, ww, hh)
                cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
                cv2.putText(roi_color, "Smile", (xx, yy+hh+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 255, 78), 2)
    cv2.imshow('video', img) 
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
