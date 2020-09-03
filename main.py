from utils import *
import keras
import numpy as np
import cv2 


model_fer = FacialExpressionRecogn()
model_gender = GenderRecogn()


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
            startX, startY = f[0], f[1]
            endX, endY = f[2], f[3]
            startX,startY,endX,endY = adjust_face_detect(startX,startY,endX,endY)
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
            face_crop = crop_image(frame, startX, startY, endX, endY)

            fer_label, fer_percent = model_fer.recognize(face_crop)
            gender_label, gender_percent = model_gender.recognize(face_crop)
            fer_text = "emotion: {}".format(fer_label)
            gender_text = "{}: {:.2f}%".format(gender_label, gender_percent)
            print("{}\n{}".format(fer_text,gender_text))
            cv2.putText(frame, gender_text, (startX, startY),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, fer_text, (startX, startY-20),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        fps = int(cv2.getTickFrequency()//(cv2.getTickCount()-timer))
        fps_text = "FPS: {}".format(fps)
        cv2.putText(frame, fps_text, (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("ProPTIT AI Force Gen 2", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
