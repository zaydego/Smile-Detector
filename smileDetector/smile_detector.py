import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_capture, frame = webcam.read()

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_grayscale)
    smiles = smile_detector.detectMultiScale(frame_grayscale, scaleFactor = 1.7, minNeighbors = 20)
    #print(faces)

    cv2.imshow("Isaiah Jones Smile Detector", frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100,150,0), 5)

        face = frame[y:y+h, x:x+w]
        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 20)

        if len(smiles) > 0:
            cv2.putText(frame, 'smiling!', (x, y+h+40), fontScale = 1,
            fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL, color = (0, 255, 0))



    #for (x, y, w, h) in smiles:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Isaiah Jones Smile Detector", frame)

    key = cv2.waitKey(1)
    if key == 81 or key == 113 or not successful_frame_capture:
        break
    pass

webcam.release()
cv2.destroyAllWindows()

print("Code completed")
