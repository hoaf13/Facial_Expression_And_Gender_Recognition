from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import cv2
import cvlib as cv
from config import (LINK_TO_FACE_DETECT,LINK_TO_FACIAL_EXPRESSION_RECOGN, LINK_TO_GENDER_RECOGN,
                    CROP_SIZE, WEBCAM_HEIGHT, WEBCAM_WIDTH, GENDER_CLASSES, FACIAL_EXPRESSION_CLASSES)

def adjust_face_detect(startX,startY,endX,endY):
    if startX > 20: startX -= 40
    if startY > 20: startY -= 40
    if endX < WEBCAM_WIDTH - 20 : endX += 10
    if endY < WEBCAM_HEIGHT - 20: endY += 10
    return startX, startY, endX, endY

def crop_image(image, startX,startY,endX,endY):
    face_crop = np.copy(image[startY:endY,startX:endX])
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_crop = cv2.resize(face_crop, (CROP_SIZE,CROP_SIZE))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = np.array(face_crop)
    face_crop = np.reshape(face_crop,(1,CROP_SIZE,CROP_SIZE,1))
    return face_crop

class FaceDetector:
    
    def __init__(self):
        pass

    @staticmethod
    def detect_faces(image):
        faces, confidence = cv.detect_face(image)
        return faces, confidence


class FacialExpressionRecogn:
    
    def __init__(self):
        self.model = load_model(LINK_TO_FACIAL_EXPRESSION_RECOGN)
        self.model.summary()
        self.classes = FACIAL_EXPRESSION_CLASSES
        
        
    def recognize(self, image):
        probabilities = self.model.predict(image)[0]
        index = np.argmax(probabilities)
        label = self.classes[index]
        percent = probabilities[index] * 100
        return label, percent

    
class GenderRecogn:
    
    def __init__(self):
        self.model = load_model(LINK_TO_GENDER_RECOGN) 
        self.model.summary()
        self.classes = GENDER_CLASSES

    def recognize(self, image):
        probabilities = self.model.predict(image)[0]
        index = np.argmax(probabilities)
        label = self.classes[index]
        percent = probabilities[index] * 100
        return label, percent

