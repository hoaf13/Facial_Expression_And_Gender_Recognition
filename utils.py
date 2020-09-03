from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import cv2
import cvlib as cv
from config import *

def adjust_face_detect(startX,startY,endX,endY):
    if startX >= 10: startX -= 10
    if startY >= 40: startY -= 40
    if endX <= WEBCAM_WIDTH - 10 : endX += 10
    if endY <= WEBCAM_HEIGHT - 10: endY += 10
    return startX, startY, endX, endY

def crop_image(image, startX,startY,endX,endY):
    face_crop = np.copy(image[startY:endY,startX:endX])
    gray_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray_face_crop = cv2.resize(gray_face_crop, (CROP_SIZE,CROP_SIZE))
    gray_face_crop = gray_face_crop.astype("float") / 255.0
    gray_face_crop = np.array(gray_face_crop)
    gray_face_crop = np.reshape(gray_face_crop,(1,CROP_SIZE,CROP_SIZE,1))
    return gray_face_crop


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

