from utils import *
import keras
import numpy as np
import cv2 

model_fer = FacialExpressionRecogn()
model_gender = GenderRecogn()
emotions = cv2.imread('images/emotions.png')
print(emotions.size)
emotions = cv2.resize(emotions, (300,480))
emotions = np.array(emotions)

webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Could not open webcam")
    exit()

if __name__ == '__main__':

    while webcam.isOpened():
        timer = cv2.getTickCount()
        status, frame = webcam.read()
        if not status:
            print("Could not read frame")
            exit()   
        face, confidence = FaceDetector.detect_faces(frame)
        for idx, f in enumerate(face):
            """ crop face for fer """
            startX, startY , endX, endY = f[0], f[1], f[2], f[3]
            fer_face_crop = crop_image(frame, startX, startY, endX, endY) 
            """ crop face for gender recogn """
            startX,startY,endX,endY = adjust_face_detect(startX,startY,endX,endY)
            gender_face_crop = crop_image(frame, startX, startY, endX, endY)
            """ draw """
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
            
            
            fer_label, fer_percent = model_fer.recognize(fer_face_crop)
            fer_text = "emotion: {}".format(fer_label)
            cv2.putText(frame, fer_text, (startX, startY-25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print("{} - {:.2f}%".format(fer_text, fer_percent))

            gender_label, gender_percent = model_gender.recognize(gender_face_crop)
            gender_text = "gender: {}".format(gender_label)
            cv2.putText(frame, gender_text, (startX, startY-5),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print("{} - {:.2f}%".format(gender_label, gender_percent))
            
        fps = int(cv2.getTickFrequency()//(cv2.getTickCount()-timer))
        fps_text = "FPS: {}".format(fps)
        cv2.putText(frame, fps_text, (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        window = np.hstack((frame, emotions))
        cv2.imshow("ProPTIT AI Force Gen 2", window)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
