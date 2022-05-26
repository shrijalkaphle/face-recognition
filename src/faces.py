import cv2
import pickle
import numpy as np
from datetime import datetime

def writeToSheets(name):
    with open('Sheet.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime('%D %H:%M:%S')
            f.writelines(f'\n{name},{dateString}')

face_cascade = cv2.CascadeClassifier('src\cascades\data\haarcascade_frontalface_alt2.xml')
reconizer = cv2.face.LBPHFaceRecognizer_create()
reconizer.read("trainner.yml")

face_detected = False

labels = {}
with open("labels.pickle", 'rb') as f:
    label = pickle.load(f)
    labels = {v:k for k,v in label.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # reconize the face
        id_, conf = reconizer.predict(roi_gray)
        if conf>=45 and conf <=85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,0)
            stroke = 2
            cv2.putText(frame,name,(x,y), font, 0.5, color, stroke, cv2.LINE_AA)
            writeToSheets(labels[id_])

        # Draw reactangle around the face
        color = (255, 0, 0) #BGR from 0-255
        stroke = 2
        cv2.rectangle(frame,(x,y), ((x+w), (y+h)), color, stroke)
   
    # Display the resulting frame
    cv2.imshow('Image Recognition',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()