import cv2
import face_recognition as face
import os
import numpy as np
from datetime import datetime


path = 'Faces'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])



def encod (images):
    encodeList = []
    for img in images:
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = encod(images)


def attend (name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')




cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    frame = cv2.flip(img, 1)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face.face_locations(imgS)
    encodesCurFrame = face.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        matches = face.compare_faces(encodeListKnown, encodeFace)
        faceDis = face.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = (y1 * 4)-40 , (x2 * 4) +25, (y2 * 4)+20, (x1 * 4) -20
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 45), (x2, y2), (0, 255, 0), cv2.FILLED)

        if matches[matchIndex]:


            name = classNames[matchIndex].upper()


            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, .8 , (255, 255, 255), 2)
            attend(name)

    cv2.imshow('camera', img)
    if cv2.waitKey(1) == ord('q'):
        break



